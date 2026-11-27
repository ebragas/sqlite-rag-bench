import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createFilesystemCorpusReader } from "../../src/adapters/corpus-reader.js";
import {
  GeminiEmbedder,
  parseRetryDelayMs,
  type EmbedBatchResult,
  type Embedder
} from "../../src/adapters/embedder.js";
import { KnowledgeStore } from "../../src/adapters/knowledge-store.js";
import type {
  LlmAgentStepInput,
  LlmAgentStepOutput,
  LlmProvider,
  LlmSynthesizeInput,
  LlmSynthesizeOutput
} from "../../src/adapters/llm-provider.js";
import { RetrievalError } from "../../src/models/corpus.js";
import { ConfigError } from "../../src/models/errors.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import { answerQuestion } from "../../src/usecases/answer.js";
import {
  capEmbeddingText,
  ingestCorpus,
  type EmbeddingUsageEvent
} from "../../src/usecases/ingest.js";
import { retrieveEvidence } from "../../src/usecases/retrieve.js";
import { l2Normalize } from "../../src/utils/vector.js";

/**
 * Deterministic 4-axis semantic fake embedder for real SQLite tests:
 * axis 0: identity / access / login / sso / federation / credentials
 * axis 1: billing / invoice / renewal / pricing / finance
 * axis 2: database / replication / storage / postgres
 * axis 3: incident / outage / rollback / pager
 */
class DeterministicSemanticEmbedder implements Embedder {
  public readonly model: string;
  public readonly dims: number;
  public readonly batches: string[][] = [];
  public totalTextsEmbedded = 0;

  public constructor(model = "gemini-embedding-001", dims = 4) {
    this.model = model;
    this.dims = dims;
  }

  public async embedBatch(texts: readonly string[]): Promise<EmbedBatchResult> {
    this.batches.push([...texts]);
    this.totalTextsEmbedded += texts.length;

    const vectors = texts.map((text) => this.embedSingle(text));
    const inputTokens = texts.reduce(
      (sum, text) => sum + Math.max(1, Math.ceil(text.length / 4)),
      0
    );
    return { vectors, inputTokens };
  }

  private embedSingle(text: string): Float32Array {
    const lower = text.toLowerCase();
    const raw = new Float32Array(this.dims);

    const scoreAxis = (keywords: readonly string[]): number => {
      let count = 0;
      for (const word of keywords) {
        if (lower.includes(word)) {
          count += 1;
        }
      }
      return count;
    };

    raw[0] = scoreAxis([
      "sso",
      "saml",
      "federated",
      "identity",
      "login",
      "sign-in",
      "credentials",
      "authentication"
    ]);
    raw[1] = scoreAxis([
      "billing",
      "invoice",
      "pricing",
      "renewal",
      "subscription",
      "contract"
    ]);
    raw[2] = scoreAxis([
      "postgres",
      "database",
      "replication",
      "storage",
      "snapshot"
    ]);
    raw[3] = scoreAxis(["incident", "outage", "rollback", "pager", "alert"]);

    const sum = (raw[0] ?? 0) + (raw[1] ?? 0) + (raw[2] ?? 0) + (raw[3] ?? 0);
    if (sum === 0) {
      raw[0] = 0.25;
      raw[1] = 0.25;
      raw[2] = 0.25;
      raw[3] = 0.25;
    }

    return l2Normalize(raw);
  }
}

class ScriptedHybridLlm implements LlmProvider {
  public readonly stepInputs: LlmAgentStepInput[] = [];
  public readonly synthesizeInputs: LlmSynthesizeInput[] = [];
  private stepCount = 0;

  public async synthesize(
    input: LlmSynthesizeInput
  ): Promise<LlmSynthesizeOutput> {
    this.synthesizeInputs.push(input);
    return {
      result: {
        answer: "Federated sign-in uses SAML identity assertion certificates.",
        abstained: false,
        supporting_doc_ids: ["dsid_identity_guide"]
      },
      usage: { inputTokens: 80, outputTokens: 20 },
      model: input.model
    };
  }

  public async runAgentStep(
    input: LlmAgentStepInput
  ): Promise<LlmAgentStepOutput> {
    this.stepInputs.push({
      model: input.model,
      system: input.system,
      messages: structuredClone(input.messages),
      tools: [...input.tools]
    });
    this.stepCount += 1;

    if (this.stepCount === 1) {
      return {
        text: "Searching with hybrid retrieval.",
        toolCalls: [
          {
            toolCallId: "call_search_1",
            toolName: "search",
            input: { query: "enterprise login credentials", topK: 5 }
          }
        ],
        usage: { inputTokens: 40, outputTokens: 12 },
        model: input.model
      };
    }

    if (this.stepCount === 2) {
      return {
        text: "Fetching top semantic match.",
        toolCalls: [
          {
            toolCallId: "call_fetch_1",
            toolName: "fetch_docs",
            input: { doc_ids: ["dsid_identity_guide"] }
          }
        ],
        usage: { inputTokens: 50, outputTokens: 10 },
        model: input.model
      };
    }

    return {
      text: "Evidence gathered.",
      toolCalls: [],
      usage: { inputTokens: 30, outputTokens: 8 },
      model: input.model
    };
  }
}

const tempDirs: string[] = [];
const openStores: KnowledgeStore[] = [];
const openServers: Server[] = [];

function createTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function writeJson(filePath: string, payload: unknown): void {
  mkdirSync(join(filePath, ".."), { recursive: true });
  writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf8");
}

afterEach(async () => {
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolveClose) => {
        server.close(() => resolveClose());
        server.closeAllConnections();
      });
    }
  }
  while (openStores.length > 0) {
    const store = openStores.pop();
    store?.close();
  }
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("hybrid dense retrieval, incremental embedding ingest, and RRF", () => {
  it("finds semantic matches missed by BM25, boosts dual hits via RRF, re-embeds only changed docs, and removes deleted vectors from SQLite and RAM", async () => {
    const root = createTempDir("srb-l2-retrieve-");
    const corpusDir = join(root, "corpus");
    const dbPath = join(root, "cache", "brain.sqlite");

    const identityPath = join(corpusDir, "confluence", "identity.json");
    const hybridBothPath = join(corpusDir, "confluence", "billing-login.json");
    const dbPathDoc = join(corpusDir, "jira", "postgres.json");

    writeJson(identityPath, {
      dataset_doc_uuid: "dsid_identity_guide",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Federated SAML Architecture",
      body: "SAML identity federation uses signed XML assertions for workforce access."
    });
    writeJson(hybridBothPath, {
      dataset_doc_uuid: "dsid_billing_access",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Billing Portal Login Policy",
      body: "Enterprise billing renewal portal requires authentication and invoice approval."
    });
    writeJson(dbPathDoc, {
      dataset_doc_uuid: "dsid_postgres_ops",
      title_field_name: "summary",
      content_field_names: ["description"],
      summary: "Postgres Storage Snapshot",
      description: "Automated database replication snapshot runs every hour."
    });

    const profile = RunProfileSchema.parse({
      name: "l2-test",
      enableEmbeddings: true,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: 4,
      enabledTools: ["search", "fetch_docs", "filter_metadata"],
      topK: 5
    });

    const embedder = new DeterministicSemanticEmbedder("gemini-embedding-001", 4);
    const usageEvents: EmbeddingUsageEvent[] = [];
    const store = new KnowledgeStore(dbPath);
    openStores.push(store);

    const reader = createFilesystemCorpusReader(corpusDir);
    const first = await ingestCorpus(reader, store, profile, {
      embedder,
      embeddingBatchSize: 2,
      onEmbeddingUsage: (event) => usageEvents.push(event)
    });

    expect(first.inserted).toBe(3);
    expect(first.embeddedChunks).toBe(3);
    expect(first.embeddingInputTokens).toBeGreaterThan(0);
    expect(embedder.batches).toHaveLength(2);
    expect(usageEvents).toHaveLength(2);
    expect(store.getVectorCacheStats()).toEqual({
      loaded: true,
      model: "gemini-embedding-001",
      dims: 4,
      vectorCount: 3,
      byteLength: 3 * 4 * Float32Array.BYTES_PER_ELEMENT
    });

    // Query with zero lexical overlap with dsid_identity_guide ("credentials" is absent from its text).
    expect(store.searchFts("credentials", 5)).toEqual([]);

    const semanticOnly = await retrieveEvidence(
      store,
      profile,
      "credentials",
      { embedder, topK: 3 }
    );
    expect(semanticOnly.mode).toBe("hybrid");
    expect(semanticOnly.ftsHitCount).toBe(0);
    expect(semanticOnly.denseHitCount).toBe(3);
    expect(semanticOnly.hits[0]?.datasetDocUuid).toBe("dsid_identity_guide");
    expect(semanticOnly.hits[0]?.ftsRank).toBeNull();
    expect(semanticOnly.hits[0]?.denseRank).toBe(1);
    expect(semanticOnly.hits[0]?.denseScore).toBeCloseTo(1, 5);

    // Query where dsid_billing_access matches BOTH FTS5 ("invoice") and dense axis 1 (billing/invoice).
    const dualHit = await retrieveEvidence(
      store,
      profile,
      "invoice billing renewal",
      { embedder, topK: 3 }
    );
    expect(dualHit.hits[0]?.datasetDocUuid).toBe("dsid_billing_access");
    expect(dualHit.hits[0]?.ftsRank).toBe(1);
    expect(dualHit.hits[0]?.denseRank).toBe(1);
    expect(dualHit.hits[0]?.rrfScore).toBeCloseTo(2 / 61, 8);

    // Re-ingest unchanged corpus: zero additional chunks embedded.
    const embedsBeforeSecond = embedder.totalTextsEmbedded;
    const second = await ingestCorpus(reader, store, profile, { embedder });
    expect(second.unchanged).toBe(3);
    expect(second.embeddedChunks).toBe(0);
    expect(second.embeddingInputTokens).toBe(0);
    expect(embedder.totalTextsEmbedded).toBe(embedsBeforeSecond);

    // Update one document: only the updated document is re-embedded.
    writeJson(dbPathDoc, {
      dataset_doc_uuid: "dsid_postgres_ops",
      title_field_name: "summary",
      content_field_names: ["description"],
      summary: "Outage Rollback Pager Procedure",
      description: "Incident alert escalation and automated rollback runbook."
    });
    const third = await ingestCorpus(reader, store, profile, { embedder });
    expect(third.updated).toBe(1);
    expect(third.unchanged).toBe(2);
    expect(third.embeddedChunks).toBe(1);
    expect(embedder.totalTextsEmbedded).toBe(embedsBeforeSecond + 1);

    // Direct deleteDocument removes vectors from SQLite AND RAM cache immediately.
    expect(store.deleteDocument("dsid_identity_guide")).toBe(true);
    expect(store.getVectorCacheStats().vectorCount).toBe(2);
    const afterDeleteHits = store.searchDense(
      new Float32Array([1, 0, 0, 0]),
      5
    );
    expect(
      afterDeleteHits.some((hit) => hit.datasetDocUuid === "dsid_identity_guide")
    ).toBe(false);

    // Reloading vector cache from SQLite confirms CASCADE removed the embedding row permanently.
    const reloadedStats = store.loadVectorCache("gemini-embedding-001", 4);
    expect(reloadedStats.vectorCount).toBe(2);

    // Removing the file on disk and re-ingesting keeps vector count clean.
    unlinkSync(identityPath);
    const fourth = await ingestCorpus(reader, store, profile, { embedder });
    expect(fourth.total).toBe(2);
    expect(fourth.embeddedChunks).toBe(0);
    expect(store.getVectorCacheStats().vectorCount).toBe(2);
  });

  it("skips embeddings completely when enableEmbeddings is false and routes L2 agent search through retrieveEvidence", async () => {
    const root = createTempDir("srb-l2-gating-");
    const corpusDir = join(root, "corpus");

    writeJson(join(corpusDir, "confluence", "identity.json"), {
      dataset_doc_uuid: "dsid_identity_guide",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Federated SAML Architecture",
      body: "SAML identity federation uses signed XML assertions."
    });

    const ftsOnlyProfile = RunProfileSchema.parse({
      name: "l1-fts-only",
      enableEmbeddings: false,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: 4,
      enabledTools: ["search", "fetch_docs"]
    });

    const embedder = new DeterministicSemanticEmbedder("gemini-embedding-001", 4);
    const store = new KnowledgeStore(":memory:");
    openStores.push(store);

    const reader = createFilesystemCorpusReader(corpusDir);
    const ftsIngest = await ingestCorpus(reader, store, ftsOnlyProfile, {
      embedder
    });
    expect(ftsIngest.embeddedChunks).toBe(0);
    expect(ftsIngest.embeddingInputTokens).toBe(0);
    expect(embedder.totalTextsEmbedded).toBe(0);
    expect(store.getVectorCacheStats().loaded).toBe(false);
    expect(() => store.searchDense(new Float32Array([1, 0, 0, 0]), 5)).toThrow(
      RetrievalError
    );

    const ftsRetrieve = await retrieveEvidence(
      store,
      ftsOnlyProfile,
      "SAML federation",
      { embedder }
    );
    expect(ftsRetrieve.mode).toBe("fts");
    expect(ftsRetrieve.denseHitCount).toBe(0);
    expect(ftsRetrieve.hits[0]?.datasetDocUuid).toBe("dsid_identity_guide");
    expect(embedder.totalTextsEmbedded).toBe(0);

    // Enable embeddings and verify ConfigError when embedder is omitted.
    const hybridProfile = RunProfileSchema.parse({
      name: "l2-hybrid-test",
      enableEmbeddings: true,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: 4,
      enabledTools: ["search", "fetch_docs"],
      topK: 5
    });
    await expect(ingestCorpus(reader, store, hybridProfile)).rejects.toThrow(
      ConfigError
    );

    await ingestCorpus(reader, store, hybridProfile, { embedder });
    const llm = new ScriptedHybridLlm();
    const outcome = await answerQuestion(
      store,
      llm,
      hybridProfile,
      "How does enterprise login work?",
      { embedder }
    );

    expect(outcome.abstained).toBe(false);
    expect(outcome.embeddingInputTokens).toBeGreaterThan(0);
    expect(outcome.response.document_ids).toEqual(["dsid_identity_guide"]);

    const secondStepMessages = llm.stepInputs[1]?.messages ?? [];
    const toolMessage = secondStepMessages.find((msg) => msg.role === "tool");
    expect(toolMessage).toMatchObject({
      role: "tool",
      results: [
        {
          toolName: "search",
          output: {
            mode: "hybrid",
            totalHits: 1,
            hits: [
              expect.objectContaining({
                dataset_doc_uuid: "dsid_identity_guide",
                dense_rank: 1
              })
            ]
          }
        }
      ]
    });
  });

  it("GeminiEmbedder passes outputDimensionality, batches requests, retries transient 429, and L2-normalizes vectors", async () => {
    const receivedBodies: Array<Record<string, unknown>> = [];
    let attemptCount = 0;

    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        attemptCount += 1;
        const bodyText = Buffer.concat(chunks).toString("utf8");
        const parsed = JSON.parse(bodyText) as Record<string, unknown>;
        receivedBodies.push(parsed);

        res.setHeader("Content-Type", "application/json");
        if (attemptCount === 1) {
          res.statusCode = 429;
          res.end(
            JSON.stringify({
              error: {
                code: 429,
                message: "Rate limit",
                status: "RESOURCE_EXHAUSTED"
              }
            })
          );
          return;
        }

        res.statusCode = 200;
        if (Array.isArray(parsed.requests)) {
          const requests = parsed.requests as unknown[];
          res.end(
            JSON.stringify({
              embeddings: requests.map(() => ({ values: [3, 4, 0, 0] }))
            })
          );
          return;
        }

        res.end(
          JSON.stringify({
            embedding: { values: [0, 0, 3, 4] }
          })
        );
      });
    });

    await new Promise<void>((resolveListen) => {
      server.listen(0, "127.0.0.1", resolveListen);
    });
    openServers.push(server);
    const address = server.address() as AddressInfo;

    const embedder = new GeminiEmbedder({
      apiKey: "test-key",
      model: "gemini-embedding-001",
      dims: 4,
      batchSize: 2,
      maxAttempts: 3,
      baseDelayMs: 1,
      maxDelayMs: 5,
      baseURL: `http://127.0.0.1:${address.port}/v1beta`
    });

    // Starts unthrottled (fast path for Tier 2/3 keys) until a 429 engages Tier-1 pacing.
    expect(embedder.effectiveTokensPerMinute).toBeNull();

    const result = await embedder.embedBatch([
      "alpha text",
      "beta text",
      "gamma text"
    ]);
    expect(embedder.effectiveTokensPerMinute).toBe(900_000);
    expect(result.vectors).toHaveLength(3);
    expect(result.inputTokens).toBeGreaterThan(0);
    expect(result.vectors[0]?.[0]).toBeCloseTo(0.6, 5);
    expect(result.vectors[0]?.[1]).toBeCloseTo(0.8, 5);
    expect(result.vectors[2]?.[2]).toBeCloseTo(0.6, 5);
    expect(result.vectors[2]?.[3]).toBeCloseTo(0.8, 5);

    // 1st call failed (429), 2nd call succeeded for batch [0..1], 3rd call succeeded for batch [2].
    expect(attemptCount).toBe(3);
    const batchRequests = receivedBodies[1]?.requests as Array<
      Record<string, unknown>
    >;
    expect(batchRequests).toHaveLength(2);
    expect(batchRequests[0]?.outputDimensionality).toBe(4);
    expect(receivedBodies[2]?.outputDimensionality).toBe(4);
  });

  it("parses server retry delays and caps oversized embedding chunks using head+tail", () => {
    expect(parseRetryDelayMs(new Error("Please retry in 44.2s."))).toBe(45_200);
    expect(parseRetryDelayMs(new Error('{"retryDelay":"12s"}'))).toBe(13_000);
    expect(parseRetryDelayMs(new Error("Generic error"))).toBeNull();

    const shortText = "Short header and body.";
    expect(capEmbeddingText(shortText, 1200)).toBe(shortText);

    const longText = `HEAD_MARKER ${"a".repeat(1500)} TAIL_MARKER`;
    const capped = capEmbeddingText(longText, 120);
    expect(capped.length).toBeLessThanOrEqual(120);
    expect(capped).toContain("HEAD_MARKER");
    expect(capped).toContain("TAIL_MARKER");
    expect(capped).toContain("\n...\n");
  });
});
