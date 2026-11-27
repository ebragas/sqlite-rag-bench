import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createConnection } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  readCorpusDirectory,
  type CorpusFileRecord,
  type CorpusReader
} from "../../src/adapters/corpus-reader.js";
import type { EmbedBatchResult, Embedder } from "../../src/adapters/embedder.js";
import type {
  LlmProvider,
  LlmSynthesizeInput,
  LlmSynthesizeOutput
} from "../../src/adapters/llm-provider.js";
import {
  AskResponseSchema,
  ProviderError,
  RetrievalError
} from "../../src/models/corpus.js";
import { ConfigError, ContractError } from "../../src/models/errors.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import {
  createBrainServer,
  parseProfileOverridesEnv,
  parseServerEnv,
  serializeBoundedAskResponse,
  type BrainServerHandle
} from "../../src/services/server.js";
import { resolveCachedDbPath } from "../../src/usecases/ingest.js";
import { ServerLifecycleController } from "../../src/usecases/server-lifecycle.js";

interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(error: Error): void;
}

function createDeferred<T>(): Deferred<T> {
  let resolveFn: (value: T) => void = () => undefined;
  let rejectFn: (error: Error) => void = () => undefined;
  const promise = new Promise<T>((resolve, reject) => {
    resolveFn = resolve;
    rejectFn = reject;
  });
  return {
    promise,
    resolve: resolveFn,
    reject: rejectFn
  };
}

class GatedCorpusReader implements CorpusReader {
  private readonly corpusDir: string;
  private readonly gate: Promise<void>;

  public constructor(corpusDir: string, gate: Promise<void>) {
    this.corpusDir = corpusDir;
    this.gate = gate;
  }

  public async *readCorpus(signal?: AbortSignal): AsyncGenerator<CorpusFileRecord> {
    await this.gate;
    yield* readCorpusDirectory(this.corpusDir, signal);
  }
}

class ControlledLlmProvider implements LlmProvider {
  private readonly handler: (
    input: LlmSynthesizeInput,
    signal?: AbortSignal
  ) => Promise<LlmSynthesizeOutput>;

  public constructor(
    handler: (
      input: LlmSynthesizeInput,
      signal?: AbortSignal
    ) => Promise<LlmSynthesizeOutput>
  ) {
    this.handler = handler;
  }

  public synthesize(
    input: LlmSynthesizeInput,
    signal?: AbortSignal
  ): Promise<LlmSynthesizeOutput> {
    return this.handler(input, signal);
  }
}

const tempDirs: string[] = [];
const handles: BrainServerHandle[] = [];

function createFixtureCorpus(): { root: string; corpusDir: string; dbPath: string } {
  const root = mkdtempSync(join(tmpdir(), "srb-server-"));
  tempDirs.push(root);
  const corpusDir = join(root, "corpus");
  const dbPath = join(root, "cache", "brain.sqlite");

  mkdirSync(join(corpusDir, "confluence", "product"), { recursive: true });
  writeFileSync(
    join(corpusDir, "confluence", "product", "pricing.json"),
    JSON.stringify({
      dataset_doc_uuid: "dsid_pricing_1",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Enterprise Pricing Tier",
      status: "approved",
      body: "Enterprise plan includes SSO, SAML, and dedicated SLA support."
    }),
    "utf8"
  );

  return { root, corpusDir, dbPath };
}

afterEach(async () => {
  while (handles.length > 0) {
    const handle = handles.pop();
    if (handle !== undefined) {
      await handle.shutdown(50);
    }
  }
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("parseServerEnv and serializeBoundedAskResponse", () => {
  it("parses valid environment config and rejects invalid config", () => {
    const parsed = parseServerEnv(
      {
        CORPUS_DIR: "/data/corpus",
        PORT: "8080",
        GEMINI_API_KEY: "secret-key"
      },
      "/workspace"
    );

    expect(parsed).toEqual({
      corpusDir: "/data/corpus",
      port: 8080,
      geminiApiKey: "secret-key",
      brainProfile: "prod",
      cacheDir: "/workspace/.cache"
    });

    const parsedWithDb = parseServerEnv(
      {
        CORPUS_DIR: "/data/corpus",
        PORT: "8080",
        GEMINI_API_KEY: "secret-key",
        DB_PATH: "/custom/path.sqlite",
        BRAIN_PROFILE_OVERRIDES: 'topK=20;enabledTools=["search","fetch_docs"],maxAgentTurns=4'
      },
      "/workspace"
    );
    expect(parsedWithDb.dbPath).toBe("/custom/path.sqlite");
    expect(parsedWithDb.brainProfileOverrides).toEqual([
      "topK=20",
      'enabledTools=["search","fetch_docs"]',
      "maxAgentTurns=4"
    ]);
    expect(parseProfileOverridesEnv(undefined)).toEqual([]);
    expect(parseProfileOverridesEnv("  ")).toEqual([]);

    expect(() => parseServerEnv({})).toThrow(ConfigError);
  });

  it("bounds oversized AskResponse payloads strictly to 1 MiB valid JSON", () => {
    const hugeAnswer = "A".repeat(1_300_000);
    const serialized = serializeBoundedAskResponse({
      answer: hugeAnswer,
      document_ids: ["dsid_pricing_1"]
    });

    expect(Buffer.byteLength(serialized, "utf8")).toBeLessThanOrEqual(
      1024 * 1024
    );
    const parsed = AskResponseSchema.parse(JSON.parse(serialized));
    expect(parsed.answer.endsWith("...")).toBe(true);
    expect(parsed.document_ids).toEqual(["dsid_pricing_1"]);
  });
});

describe("ServerLifecycleController state transitions", () => {
  it("enforces bounded concurrency, startup failure, and drain timeout abort", async () => {
    const lifecycle = new ServerLifecycleController({ maxConcurrent: 2 });
    expect(lifecycle.getHealthStatusCode()).toBe(503);
    expect(lifecycle.getPhase().kind).toBe("booting");

    lifecycle.startIngest();
    expect(lifecycle.getPhase().kind).toBe("ingesting");
    lifecycle.markIngestSucceeded();
    expect(lifecycle.getHealthStatusCode()).toBe(200);
    expect(lifecycle.isIngestDone()).toBe(true);

    const slot1 = createDeferred<void>();
    const slot2 = createDeferred<void>();
    let observedAbort = false;

    const req1 = lifecycle.runAsk(async (signal) => {
      signal.addEventListener("abort", () => {
        observedAbort = true;
      });
      await slot1.promise;
      return "one";
    });
    const req2 = lifecycle.runAsk(async () => {
      await slot2.promise;
      return "two";
    });
    const queuedReq3 = lifecycle.runAsk(async () => "three");

    expect(lifecycle.getInFlightCount()).toBe(2);

    const drainDone = lifecycle.requestShutdown(0);
    await expect(queuedReq3).rejects.toThrow(ContractError);
    await drainDone;

    expect(lifecycle.getPhase().kind).toBe("stopped");
    expect(lifecycle.getInFlightCount()).toBe(0);
    expect(observedAbort).toBe(true);

    slot1.resolve();
    slot2.resolve();
    await Promise.all([req1, req2]);
    expect(lifecycle.getInFlightCount()).toBe(0);

    const failingLifecycle = new ServerLifecycleController();
    failingLifecycle.startIngest();
    failingLifecycle.markIngestFailed("corrupt corpus");
    expect(failingLifecycle.getPhase()).toEqual({
      kind: "failed",
      reason: "corrupt corpus"
    });
    expect(failingLifecycle.getHealthStatusCode()).toBe(503);
    expect(failingLifecycle.isIngestDone()).toBe(false);

    const startupAbort = new ServerLifecycleController();
    startupAbort.startIngest();
    await startupAbort.requestShutdown();
    expect(startupAbort.getStartupSignal().aborted).toBe(true);
    expect(startupAbort.getPhase().kind).toBe("stopped");
    startupAbort.markIngestSucceeded();
    expect(startupAbort.getHealthStatusCode()).toBe(503);

    const idleShutdown = new ServerLifecycleController();
    idleShutdown.startIngest();
    idleShutdown.markIngestSucceeded();
    await idleShutdown.requestShutdown();
    expect(idleShutdown.getPhase().kind).toBe("stopped");

    expect(() => new ServerLifecycleController({ maxConcurrent: 0 })).toThrow(
      ConfigError
    );
    expect(new RetrievalError("fts error").code).toBe("RETRIEVAL_ERROR");
    expect(new ProviderError("api error").code).toBe("PROVIDER_ERROR");
  });
});

describe("createBrainServer HTTP contract", () => {
  const profile = RunProfileSchema.parse({ name: "l0-fts" });

  it("serves /health 503 during ingestion, flips to 200 when ready, validates /ask, and enforces 1 MiB cap", async () => {
    const { corpusDir, dbPath } = createFixtureCorpus();
    const ingestGate = createDeferred<void>();

    const llm = new ControlledLlmProvider(async (input) => {
      if (input.prompt.includes("HUGE_RESPONSE")) {
        return {
          result: {
            answer: "B".repeat(1_250_000),
            abstained: false,
            supporting_doc_ids: ["dsid_pricing_1"]
          },
          usage: { inputTokens: 10, outputTokens: 10 },
          model: input.model
        };
      }
      return {
        result: {
          answer: "Enterprise plan includes SSO, SAML, and SLA support.",
          abstained: false,
          supporting_doc_ids: ["dsid_pricing_1"]
        },
        usage: { inputTokens: 50, outputTokens: 20 },
        model: input.model
      };
    });

    const handle = await createBrainServer({
      corpusDir,
      port: 0,
      dbPath,
      profile,
      llmProvider: llm,
      corpusReader: new GatedCorpusReader(corpusDir, ingestGate.promise)
    });
    handles.push(handle);

    const healthDuringIngest = await fetch(`${handle.url}/health`);
    expect(healthDuringIngest.status).toBe(503);

    const askDuringIngest = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Enterprise SSO?" })
    });
    expect(askDuringIngest.status).toBe(503);

    ingestGate.resolve();
    const summary = await handle.ready;
    expect(summary.inserted).toBe(1);

    const healthReady = await fetch(`${handle.url}/health`);
    expect(healthReady.status).toBe(200);

    const badJson = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{bad"
    });
    expect(badJson.status).toBe(400);

    const badPayload = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "   " })
    });
    expect(badPayload.status).toBe(400);

    const notFound = await fetch(`${handle.url}/unknown`);
    expect(notFound.status).toBe(404);

    const askOk = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "What does the Enterprise plan include?" })
    });
    expect(askOk.status).toBe(200);
    expect(await askOk.json()).toEqual({
      answer: "Enterprise plan includes SSO, SAML, and SLA support.",
      document_ids: ["dsid_pricing_1"]
    });

    const askHuge = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Enterprise HUGE_RESPONSE" })
    });
    expect(askHuge.status).toBe(200);
    const hugeRaw = await askHuge.text();
    expect(Buffer.byteLength(hugeRaw, "utf8")).toBeLessThanOrEqual(1024 * 1024);
    const hugeParsed = AskResponseSchema.parse(JSON.parse(hugeRaw));
    expect(hugeParsed.document_ids).toEqual(["dsid_pricing_1"]);
  });

  it("handles 4 concurrent /ask requests without state leakage and drains cleanly on shutdown", async () => {
    const { corpusDir, dbPath } = createFixtureCorpus();

    const allFourArrived = createDeferred<void>();
    const releaseFour = createDeferred<void>();
    let concurrentActive = 0;

    const drainStarted = createDeferred<void>();
    const releaseDrainAsk = createDeferred<void>();

    const llm = new ControlledLlmProvider(async (input) => {
      if (input.prompt.includes("DRAIN_PROBE")) {
        drainStarted.resolve();
        await releaseDrainAsk.promise;
        return {
          result: {
            answer: "Completed before drain exit.",
            abstained: false,
            supporting_doc_ids: ["dsid_pricing_1"]
          },
          usage: { inputTokens: 10, outputTokens: 10 },
          model: input.model
        };
      }

      const match = /CONCURRENT_TOKEN_(\d)/.exec(input.prompt);
      const token = match?.[1] ?? "unknown";
      concurrentActive += 1;
      if (concurrentActive === 4) {
        allFourArrived.resolve();
      }
      await releaseFour.promise;
      return {
        result: {
          answer: `Isolated answer for token ${token}`,
          abstained: false,
          supporting_doc_ids: ["dsid_pricing_1"]
        },
        usage: { inputTokens: 20, outputTokens: 10 },
        model: input.model
      };
    });

    const handle = await createBrainServer({
      corpusDir,
      port: 0,
      dbPath,
      profile,
      llmProvider: llm
    });
    handles.push(handle);
    await handle.ready;

    const fourRequests = [1, 2, 3, 4].map((num) =>
      fetch(`${handle.url}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: `Enterprise SSO CONCURRENT_TOKEN_${num}`
        })
      }).then(async (res) => ({
        status: res.status,
        body: (await res.json()) as { answer: string; document_ids: string[] }
      }))
    );

    await allFourArrived.promise;
    expect(handle.lifecycle.getInFlightCount()).toBe(4);
    releaseFour.resolve();

    const results = await Promise.all(fourRequests);
    expect(results.map((item) => item.status)).toEqual([200, 200, 200, 200]);
    expect(results.map((item) => item.body.answer)).toEqual([
      "Isolated answer for token 1",
      "Isolated answer for token 2",
      "Isolated answer for token 3",
      "Isolated answer for token 4"
    ]);

    const inFlightDrainAsk = fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Enterprise DRAIN_PROBE" })
    });
    await drainStarted.promise;

    const shutdownPromise = handle.shutdown(5000);
    expect(handle.lifecycle.getPhase().kind).toBe("draining");

    const healthDraining = await fetch(`${handle.url}/health`);
    expect(healthDraining.status).toBe(503);

    const askWhileDraining = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Enterprise rejected during drain" })
    });
    expect(askWhileDraining.status).toBe(503);

    releaseDrainAsk.resolve();
    const drainedResponse = await inFlightDrainAsk;
    expect(drainedResponse.status).toBe(200);
    expect(await drainedResponse.json()).toEqual({
      answer: "Completed before drain exit.",
      document_ids: ["dsid_pricing_1"]
    });

    await shutdownPromise;
    expect(handle.lifecycle.getPhase().kind).toBe("stopped");
  });

  it("sets Connection: close on 500 during drain and force-closes lingering sockets on grace expiry", async () => {
    const { corpusDir, dbPath } = createFixtureCorpus();
    const askStarted = createDeferred<void>();
    const releaseError = createDeferred<void>();

    const llm = new ControlledLlmProvider(async () => {
      askStarted.resolve();
      await releaseError.promise;
      throw new ProviderError("Simulated upstream failure during drain");
    });

    const handle = await createBrainServer({
      corpusDir,
      port: 0,
      dbPath,
      profile,
      llmProvider: llm
    });
    handles.push(handle);
    await handle.ready;

    // Open a lingering socket with an incomplete HTTP request to verify force close on grace expiry.
    const lingeringSocket = createConnection({
      host: "127.0.0.1",
      port: handle.port
    });
    const socketClosed = createDeferred<void>();
    lingeringSocket.on("close", () => {
      socketClosed.resolve();
    });
    await new Promise<void>((resolveConnect) => {
      lingeringSocket.once("connect", resolveConnect);
    });
    lingeringSocket.write(
      "POST /ask HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 1000\r\n\r\n{"
    );

    const failingAsk = fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Enterprise SSO?" })
    });
    await askStarted.promise;

    const shutdownPromise = handle.shutdown(50);
    expect(handle.lifecycle.getPhase().kind).toBe("draining");

    releaseError.resolve();
    const errorRes = await failingAsk;
    expect(errorRes.status).toBe(500);
    expect(errorRes.headers.get("connection")).toBe("close");

    await shutdownPromise;
    await socketClosed.promise;
    expect(handle.lifecycle.getPhase().kind).toBe("stopped");
  });

  it("defaults DB path via resolveCachedDbPath and wires embedder for L2 profiles", async () => {
    const { root, corpusDir } = createFixtureCorpus();
    const cacheDir = join(root, "hash-cache");
    const l2Profile = RunProfileSchema.parse({
      name: "l2-server-test",
      enableEmbeddings: true,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: 4
    });

    const fakeEmbedder: Embedder = {
      model: "gemini-embedding-001",
      dims: 4,
      async embedBatch(texts: readonly string[]): Promise<EmbedBatchResult> {
        return {
          vectors: texts.map(() => new Float32Array([1, 0, 0, 0])),
          inputTokens: texts.length * 10
        };
      }
    };

    const llm = new ControlledLlmProvider(async (input) => ({
      result: {
        answer: "Enterprise plan includes SSO and dedicated SLA.",
        abstained: false,
        supporting_doc_ids: ["dsid_pricing_1"]
      },
      usage: { inputTokens: 30, outputTokens: 15 },
      model: input.model
    }));

    const handle = await createBrainServer({
      corpusDir,
      port: 0,
      cacheDir,
      profile: l2Profile,
      llmProvider: llm,
      embedder: fakeEmbedder
    });
    handles.push(handle);

    const summary = await handle.ready;
    expect(handle.dbPath).toBe(
      resolveCachedDbPath(cacheDir, corpusDir, l2Profile)
    );
    expect(summary.embeddedChunks).toBe(1);
    expect(handle.store.getVectorCacheStats().vectorCount).toBe(1);

    const askRes = await fetch(`${handle.url}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "Tell me about enterprise plans" })
    });
    expect(askRes.status).toBe(200);
    expect(await askRes.json()).toEqual({
      answer: "Enterprise plan includes SSO and dedicated SLA.",
      document_ids: ["dsid_pricing_1"]
    });
  });
});
