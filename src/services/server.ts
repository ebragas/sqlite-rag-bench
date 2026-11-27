import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse
} from "node:http";
import type { AddressInfo } from "node:net";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  createFilesystemCorpusReader,
  type CorpusReader
} from "../adapters/corpus-reader.js";
import { GeminiEmbedder, type Embedder } from "../adapters/embedder.js";
import { KnowledgeStore } from "../adapters/knowledge-store.js";
import {
  GeminiLlmProvider,
  type LlmProvider
} from "../adapters/llm-provider.js";
import { loadRunProfile } from "../adapters/profile-loader.js";
import {
  createNoopTelemetryController,
  createTelemetryController,
  type TelemetryController
} from "../adapters/telemetry.js";
import { AskRequestSchema, type AskResponse } from "../models/corpus.js";
import { ConfigError, ContractError, ValidationError } from "../models/errors.js";
import type { RunProfile } from "../models/profile.js";
import { answerQuestion } from "../usecases/answer.js";
import {
  ingestCorpus,
  resolveCachedDbPath,
  type IngestSummary
} from "../usecases/ingest.js";
import { ServerLifecycleController } from "../usecases/server-lifecycle.js";

const MAX_BODY_BYTES = 1024 * 1024;
const MAX_RESPONSE_BYTES = 1024 * 1024;
const DEFAULT_DRAIN_GRACE_MS = 8000;
const DEFAULT_HOST = "127.0.0.1";

const ServerEnvSchema = z.object({
  CORPUS_DIR: z.string().min(1),
  PORT: z.coerce.number().int().min(1).max(65535),
  GEMINI_API_KEY: z.string().min(1),
  BRAIN_PROFILE: z.string().min(1).default("prod"),
  BRAIN_PROFILE_OVERRIDES: z.string().optional(),
  DB_PATH: z.string().min(1).optional()
});

export interface ServerEnvConfig {
  corpusDir: string;
  port: number;
  geminiApiKey: string;
  brainProfile: string;
  brainProfileOverrides?: readonly string[];
  cacheDir: string;
  dbPath?: string;
}

export function parseProfileOverridesEnv(raw: string | undefined): string[] {
  if (raw === undefined || raw.trim().length === 0) {
    return [];
  }

  const entries: string[] = [];
  let current = "";
  let inQuotes = false;
  let escaped = false;
  let bracketDepth = 0;
  let braceDepth = 0;

  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    if (ch === undefined) {
      continue;
    }

    if (inQuotes) {
      current += ch;
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inQuotes = false;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      current += ch;
      continue;
    }
    if (ch === "[") {
      bracketDepth += 1;
      current += ch;
      continue;
    }
    if (ch === "]") {
      bracketDepth = Math.max(0, bracketDepth - 1);
      current += ch;
      continue;
    }
    if (ch === "{") {
      braceDepth += 1;
      current += ch;
      continue;
    }
    if (ch === "}") {
      braceDepth = Math.max(0, braceDepth - 1);
      current += ch;
      continue;
    }

    if ((ch === "," || ch === ";") && bracketDepth === 0 && braceDepth === 0) {
      const trimmed = current.trim();
      if (trimmed.length > 0) {
        entries.push(trimmed);
      }
      current = "";
      continue;
    }

    current += ch;
  }

  const tail = current.trim();
  if (tail.length > 0) {
    entries.push(tail);
  }
  return entries;
}

export interface CreateBrainServerOptions {
  corpusDir: string;
  port: number;
  host?: string;
  dbPath?: string;
  cacheDir?: string;
  profile: RunProfile;
  llmProvider: LlmProvider;
  embedder?: Embedder;
  corpusReader?: CorpusReader;
  maxConcurrent?: number;
  drainGraceMs?: number;
  telemetry?: TelemetryController;
}

export interface BrainServerHandle {
  readonly port: number;
  readonly url: string;
  readonly dbPath: string;
  readonly lifecycle: ServerLifecycleController;
  readonly store: KnowledgeStore;
  readonly ready: Promise<IngestSummary>;
  shutdown(graceMs?: number): Promise<void>;
}

export function parseServerEnv(
  env: NodeJS.ProcessEnv,
  cwd = process.cwd()
): ServerEnvConfig {
  const result = ServerEnvSchema.safeParse({
    CORPUS_DIR: env.CORPUS_DIR,
    PORT: env.PORT,
    GEMINI_API_KEY: env.GEMINI_API_KEY,
    BRAIN_PROFILE:
      env.BRAIN_PROFILE !== undefined && env.BRAIN_PROFILE.trim().length > 0
        ? env.BRAIN_PROFILE
        : undefined,
    BRAIN_PROFILE_OVERRIDES:
      env.BRAIN_PROFILE_OVERRIDES !== undefined &&
      env.BRAIN_PROFILE_OVERRIDES.trim().length > 0
        ? env.BRAIN_PROFILE_OVERRIDES
        : undefined,
    DB_PATH:
      env.DB_PATH !== undefined && env.DB_PATH.trim().length > 0
        ? env.DB_PATH
        : undefined
  });

  if (!result.success) {
    throw new ConfigError(`Invalid server environment: ${result.error.message}`);
  }

  const brainProfileOverrides = parseProfileOverridesEnv(
    result.data.BRAIN_PROFILE_OVERRIDES
  );

  return {
    corpusDir: result.data.CORPUS_DIR,
    port: result.data.PORT,
    geminiApiKey: result.data.GEMINI_API_KEY,
    brainProfile: result.data.BRAIN_PROFILE,
    ...(brainProfileOverrides.length > 0 ? { brainProfileOverrides } : {}),
    cacheDir: resolve(cwd, ".cache"),
    ...(result.data.DB_PATH !== undefined ? { dbPath: result.data.DB_PATH } : {})
  };
}

function sliceUtf8Safe(text: string, maxUnits: number): string {
  if (maxUnits >= text.length) {
    return text;
  }
  if (maxUnits <= 0) {
    return "";
  }
  const code = text.charCodeAt(maxUnits - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    return text.slice(0, maxUnits - 1);
  }
  return text.slice(0, maxUnits);
}

export function serializeBoundedAskResponse(
  response: AskResponse,
  maxBytes = MAX_RESPONSE_BYTES
): string {
  const direct = JSON.stringify(response);
  if (Buffer.byteLength(direct, "utf8") <= maxBytes) {
    return direct;
  }

  const documentIds = [...response.document_ids];
  const suffix = "...";
  while (
    documentIds.length > 0 &&
    Buffer.byteLength(
      JSON.stringify({ answer: suffix, document_ids: documentIds }),
      "utf8"
    ) > maxBytes
  ) {
    documentIds.pop();
  }

  let low = 1;
  let high = response.answer.length;
  let best = JSON.stringify({ answer: suffix, document_ids: documentIds });

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const candidateAnswer = `${sliceUtf8Safe(response.answer, mid)}${suffix}`;
    const candidateJson = JSON.stringify({
      answer: candidateAnswer,
      document_ids: documentIds
    });
    if (Buffer.byteLength(candidateJson, "utf8") <= maxBytes) {
      best = candidateJson;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return best;
}

function sendJson(
  res: ServerResponse,
  statusCode: number,
  body: string,
  closeConnection = false
): Promise<void> {
  if (res.writableEnded || res.destroyed) {
    return Promise.resolve();
  }
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Content-Length", Buffer.byteLength(body, "utf8"));
  if (closeConnection) {
    res.setHeader("Connection", "close");
  }
  return new Promise<void>((resolveEnd) => {
    res.end(body, () => {
      resolveEnd();
    });
  });
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise<string>((resolveBody, rejectBody) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;

    req.on("data", (chunk: Buffer | string) => {
      const buf = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
      totalBytes += buf.length;
      if (totalBytes > MAX_BODY_BYTES) {
        rejectBody(new ValidationError("Request body exceeds 1 MiB limit"));
        req.destroy();
        return;
      }
      chunks.push(buf);
    });

    req.on("end", () => {
      resolveBody(Buffer.concat(chunks).toString("utf8"));
    });

    req.on("error", (error) => {
      rejectBody(error);
    });
  });
}

function closeHttpServer(
  server: Server,
  forceCloseAfterMs = 0
): Promise<void> {
  return new Promise<void>((resolveClose) => {
    let forceTimer: ReturnType<typeof setTimeout> | null = null;
    const finish = (): void => {
      if (forceTimer !== null) {
        clearTimeout(forceTimer);
        forceTimer = null;
      }
      resolveClose();
    };

    server.close(finish);
    server.closeIdleConnections();

    if (forceCloseAfterMs <= 0) {
      server.closeAllConnections();
      return;
    }

    forceTimer = setTimeout(() => {
      forceTimer = null;
      server.closeAllConnections();
    }, forceCloseAfterMs);
    forceTimer.unref();
  });
}

export async function createBrainServer(
  options: CreateBrainServerOptions
): Promise<BrainServerHandle> {
  const host = options.host ?? DEFAULT_HOST;
  const drainGraceMs = options.drainGraceMs ?? DEFAULT_DRAIN_GRACE_MS;
  const resolvedDbPath =
    options.dbPath ??
    resolveCachedDbPath(
      options.cacheDir ?? resolve(process.cwd(), ".cache"),
      options.corpusDir,
      options.profile
    );
  const telemetry = options.telemetry ?? createNoopTelemetryController();
  const store = new KnowledgeStore(resolvedDbPath);
  const reader =
    options.corpusReader ?? createFilesystemCorpusReader(options.corpusDir);
  const lifecycle = new ServerLifecycleController(
    options.maxConcurrent !== undefined
      ? { maxConcurrent: options.maxConcurrent }
      : {}
  );

  const handleRequest = async (
    req: IncomingMessage,
    res: ServerResponse
  ): Promise<void> => {
    const urlPath = (req.url ?? "/").split("?")[0] ?? "/";

    if (req.method === "GET" && urlPath === "/health") {
      const status = lifecycle.getHealthStatusCode();
      await sendJson(
        res,
        status,
        JSON.stringify({ status: status === 200 ? "ok" : "not_ready" }),
        status !== 200
      );
      return;
    }

    if (req.method === "POST" && urlPath === "/ask") {
      if (lifecycle.getHealthStatusCode() !== 200) {
        await sendJson(
          res,
          503,
          JSON.stringify({ error: "Server is not ready to accept requests" }),
          true
        );
        return;
      }

      let rawBody: string;
      try {
        rawBody = await readRequestBody(req);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to read request body";
        await sendJson(res, 400, JSON.stringify({ error: message }));
        return;
      }

      let parsedJson: unknown;
      try {
        parsedJson = JSON.parse(rawBody);
      } catch {
        await sendJson(res, 400, JSON.stringify({ error: "Invalid JSON body" }));
        return;
      }

      const validated = AskRequestSchema.safeParse(parsedJson);
      if (!validated.success) {
        await sendJson(
          res,
          400,
          JSON.stringify({ error: `Invalid request: ${validated.error.message}` })
        );
        return;
      }

      try {
        await telemetry.runWithIncomingTraceContext(req.headers, async () => {
          await lifecycle.runAsk(async (signal) => {
            const result = await answerQuestion(
              store,
              options.llmProvider,
              options.profile,
              validated.data.question,
              {
                signal,
                telemetry,
                ...(options.embedder !== undefined
                  ? { embedder: options.embedder }
                  : {})
              }
            );
            const body = serializeBoundedAskResponse(result.response);
            const isDraining = lifecycle.getPhase().kind !== "ready";
            await sendJson(res, 200, body, isDraining);
          });
        });
      } catch (error) {
        if (error instanceof ContractError) {
          await sendJson(res, 503, JSON.stringify({ error: error.message }), true);
          return;
        }
        if (error instanceof ValidationError) {
          await sendJson(res, 400, JSON.stringify({ error: error.message }));
          return;
        }
        const message =
          error instanceof Error ? error.message : "Internal server error";
        console.error("[server] /ask failed:", error);
        const isDraining = lifecycle.getPhase().kind !== "ready";
        await sendJson(res, 500, JSON.stringify({ error: message }), isDraining);
      }
      return;
    }

    await sendJson(res, 404, JSON.stringify({ error: "Not found" }));
  };

  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((error: unknown) => {
      console.error("[server] unhandled request error:", error);
      void sendJson(res, 500, JSON.stringify({ error: "Internal server error" }), true);
    });
  });

  await new Promise<void>((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(options.port, host, () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });

  const address = server.address() as AddressInfo | null;
  if (address === null) {
    await closeHttpServer(server);
    store.close();
    throw new ContractError("Failed to determine bound HTTP server address");
  }

  lifecycle.startIngest();
  const ready = ingestCorpus(reader, store, options.profile, {
    signal: lifecycle.getStartupSignal(),
    ...(options.embedder !== undefined ? { embedder: options.embedder } : {})
  }).then(
    (summary) => {
      lifecycle.markIngestSucceeded();
      return summary;
    },
    (error: unknown) => {
      const message =
        error instanceof Error ? error.message : "Corpus ingestion failed";
      lifecycle.markIngestFailed(message);
      throw error;
    }
  );
  ready.catch((error: unknown) => {
    if (!lifecycle.getStartupSignal().aborted) {
      console.error("[server] background ingestion error:", error);
    }
  });

  let shutdownPromise: Promise<void> | null = null;
  const shutdown = (graceMs = drainGraceMs): Promise<void> => {
    if (shutdownPromise !== null) {
      return shutdownPromise;
    }
    shutdownPromise = (async () => {
      const startedAtMs = Date.now();
      await lifecycle.requestShutdown(graceMs);
      const elapsedMs = Date.now() - startedAtMs;
      const remainingGraceMs = Math.max(0, graceMs - elapsedMs);
      await closeHttpServer(server, remainingGraceMs);
      store.close();
      await telemetry.shutdown();
    })();
    return shutdownPromise;
  };

  return {
    port: address.port,
    url: `http://${host}:${address.port}`,
    dbPath: resolvedDbPath,
    lifecycle,
    store,
    ready,
    shutdown
  };
}

async function runMain(): Promise<void> {
  const config = parseServerEnv(process.env);
  const profile = loadRunProfile(
    config.brainProfile,
    config.brainProfileOverrides ?? []
  );
  const dbPath =
    config.dbPath ??
    resolveCachedDbPath(config.cacheDir, config.corpusDir, profile);
  const telemetry = await createTelemetryController(process.env, {
    serviceName: "sqlite-rag-bench-server"
  });

  let handle: BrainServerHandle;
  try {
    const llmProvider = new GeminiLlmProvider({
      apiKey: config.geminiApiKey,
      telemetry
    });
    const embedder = profile.enableEmbeddings
      ? new GeminiEmbedder({
          apiKey: config.geminiApiKey,
          model: profile.embeddingModel,
          dims: profile.embeddingDims
        })
      : undefined;
    handle = await createBrainServer({
      corpusDir: config.corpusDir,
      port: config.port,
      dbPath,
      profile,
      llmProvider,
      ...(embedder !== undefined ? { embedder } : {}),
      telemetry
    });
  } catch (error) {
    await telemetry.shutdown();
    throw error;
  }

  let shuttingDown = false;
  const onSignal = (signalName: string): void => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    console.error(`[server] received ${signalName}, shutting down`);
    void handle
      .shutdown(DEFAULT_DRAIN_GRACE_MS)
      .then(() => {
        process.exit(0);
      })
      .catch((error: unknown) => {
        console.error("[server] error during shutdown:", error);
        process.exit(1);
      });
  };

  process.on("SIGTERM", () => onSignal("SIGTERM"));
  process.on("SIGINT", () => onSignal("SIGINT"));

  try {
    const summary = await handle.ready;
    console.error(
      `[server] ready on ${handle.url} (docs=${summary.total}, inserted=${summary.inserted}, updated=${summary.updated}, unchanged=${summary.unchanged}, deleted=${summary.deleted})`
    );
  } catch (error) {
    if (shuttingDown) {
      return;
    }
    console.error("[server] startup ingestion failed:", error);
    await handle.shutdown(0);
    process.exit(1);
  }
}

const currentFilePath = fileURLToPath(import.meta.url);
if (
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === currentFilePath
) {
  void runMain().catch((error: unknown) => {
    console.error("[server] fatal startup error:", error);
    process.exit(1);
  });
}
