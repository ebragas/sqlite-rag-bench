import { performance } from "node:perf_hooks";
import { resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { createFilesystemCorpusReader } from "../adapters/corpus-reader.js";
import { loadDotEnvIfPresent } from "../adapters/dotenv.js";
import { GeminiEmbedder, type Embedder } from "../adapters/embedder.js";
import { KnowledgeStore } from "../adapters/knowledge-store.js";
import {
  GeminiLlmProvider,
  type LlmProvider
} from "../adapters/llm-provider.js";
import { loadRunProfile } from "../adapters/profile-loader.js";
import {
  createTelemetryController,
  type TelemetryController
} from "../adapters/telemetry.js";
import { ConfigError } from "../models/errors.js";
import {
  computeIndexConfigHash,
  type RunProfile
} from "../models/profile.js";
import {
  answerQuestion,
  type AnswerResult,
  type AnswerTraceEvent
} from "../usecases/answer.js";
import {
  ingestCorpus,
  resolveCachedDbPath
} from "../usecases/ingest.js";
import { estimateTokenCostUsd } from "../utils/scoring.js";

export interface AskCliArgs {
  readonly profile: string;
  readonly setOverrides: readonly string[];
  readonly corpusDir: string;
  readonly dbPath?: string;
  readonly quiet: boolean;
  readonly json: boolean;
  readonly question?: string;
}

export interface AskStartupSummaryInput {
  readonly profile: RunProfile;
  readonly indexConfigHash: string;
  readonly dbPath: string;
  readonly documentCount: number;
  readonly chunkCount: number;
  readonly sourceCount: number;
  readonly graphEdgeCount: number;
  readonly graphResolvedEdgeCount: number;
  readonly vectorCount: number;
  readonly langfuseEnabled: boolean;
  readonly langfuseBaseUrl?: string;
}

export interface FormatTraceEventOptions {
  readonly color?: boolean;
}

export interface AskAnswerSummaryInput {
  readonly question: string;
  readonly profile: RunProfile;
  readonly result: AnswerResult;
  readonly latencyMs: number;
  readonly quiet?: boolean;
  readonly color?: boolean;
}

export interface AskCliOverrides {
  readonly env?: NodeJS.ProcessEnv;
  readonly cwd?: string;
  readonly stdout?: Pick<NodeJS.WriteStream, "write"> & { isTTY?: boolean };
  readonly stderr?: Pick<NodeJS.WriteStream, "write">;
  readonly llmProvider?: LlmProvider;
  readonly embedder?: Embedder;
  readonly telemetry?: TelemetryController;
}

const ANSI_RESET = "\u001B[0m";
const ANSI_DIM = "\u001B[2m";
const ANSI_CYAN = "\u001B[36m";
const ANSI_YELLOW = "\u001B[33m";
const ANSI_RED = "\u001B[31m";
const ANSI_GREEN = "\u001B[32m";
const ANSI_BOLD = "\u001B[1m";
const MAX_COMPACT_JSON_CHARS = 240;

function style(text: string, code: string, color: boolean): string {
  return color ? `${code}${text}${ANSI_RESET}` : text;
}

function compactJson(value: unknown, maxChars = MAX_COMPACT_JSON_CHARS): string {
  let serialized: string;
  try {
    serialized = JSON.stringify(value);
  } catch {
    serialized = String(value);
  }
  if (serialized.length <= maxChars) {
    return serialized;
  }
  return `${serialized.slice(0, maxChars - 3)}...`;
}

function indentLines(text: string, prefix = "  "): string {
  return text
    .split(/\r?\n/)
    .map((line) => `${prefix}${line}`)
    .join("\n");
}

export function parseAskCliArgs(argv: readonly string[]): AskCliArgs {
  let parsed: ReturnType<typeof parseArgs>;
  try {
    parsed = parseArgs({
      args: [...argv],
      strict: true,
      allowPositionals: true,
      options: {
        profile: { type: "string", default: "prod" },
        set: { type: "string", multiple: true, default: [] },
        corpus: { type: "string", default: "data/corpus" },
        db: { type: "string" },
        quiet: { type: "boolean", default: false },
        json: { type: "boolean", default: false }
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ConfigError(`Invalid ask CLI arguments: ${message}`, { cause: error });
  }

  const profile = String(parsed.values.profile ?? "prod").trim();
  if (profile.length === 0) {
    throw new ConfigError("Flag --profile must be non-empty.");
  }

  const corpusDir = String(parsed.values.corpus ?? "data/corpus").trim();
  if (corpusDir.length === 0) {
    throw new ConfigError("Flag --corpus must be non-empty.");
  }

  const rawDb = typeof parsed.values.db === "string" ? parsed.values.db.trim() : undefined;
  const dbPath = rawDb !== undefined && rawDb.length > 0 ? rawDb : undefined;
  const rawSet = Array.isArray(parsed.values.set)
    ? parsed.values.set.map((item) => String(item))
    : [];
  const joinedQuestion = parsed.positionals.join(" ").trim();

  return {
    profile,
    setOverrides: rawSet,
    corpusDir,
    ...(dbPath !== undefined ? { dbPath } : {}),
    quiet: Boolean(parsed.values.quiet),
    json: Boolean(parsed.values.json),
    ...(joinedQuestion.length > 0 ? { question: joinedQuestion } : {})
  };
}

export function estimateAnswerCostUsd(
  profile: RunProfile,
  result: AnswerResult
): number {
  const llmCost = estimateTokenCostUsd(
    result.model,
    result.usage.inputTokens,
    result.usage.outputTokens
  );
  const embedCost = estimateTokenCostUsd(
    profile.embeddingModel,
    result.embeddingInputTokens,
    0
  );
  return llmCost + embedCost;
}

export function formatStartupSummary(
  input: AskStartupSummaryInput,
  color = false
): string {
  const langfuseLabel = input.langfuseEnabled
    ? input.langfuseBaseUrl !== undefined
      ? `ON (${input.langfuseBaseUrl})`
      : "ON"
    : "OFF";
  const lines = [
    style("Enterprise Knowledge CLI", ANSI_BOLD, color),
    `  profile:  ${input.profile.name} (index=${input.indexConfigHash.slice(0, 12)})`,
    `  models:   routing=${input.profile.routingModel} | synthesis=${input.profile.synthesisModel}`,
    `  db:       ${input.dbPath}`,
    `  corpus:   docs=${input.documentCount} chunks=${input.chunkCount} sources=${input.sourceCount} graph_edges=${input.graphResolvedEdgeCount}/${input.graphEdgeCount} vectors=${input.vectorCount}`,
    `  langfuse: ${langfuseLabel}`
  ];
  return lines.join("\n");
}

export function formatTraceEvent(
  event: AnswerTraceEvent,
  options?: FormatTraceEventOptions
): string | null {
  const color = options?.color ?? false;

  switch (event.type) {
    case "pre_search_done":
      return style(
        `[search] pre-search (${event.mode}) -> ${event.hitCount} ${event.hitCount === 1 ? "hit" : "hits"} (${Math.round(event.durationMs)}ms)`,
        ANSI_CYAN,
        color
      );

    case "agent_turn_start":
      return `\n${style(`--- Turn ${event.turn}/${event.maxTurns} (${event.model}) ---`, ANSI_BOLD, color)}`;

    case "agent_thinking":
      return style(`[thinking]\n${indentLines(event.text)}`, ANSI_DIM, color);

    case "tool_call":
      return style(
        `[tool] ${event.toolName}(${compactJson(event.input)})`,
        ANSI_CYAN,
        color
      );

    case "tool_result": {
      const prefix = event.isError ? "[error] " : "";
      const line = `  -> ${prefix}${event.summary} (${Math.round(event.durationMs)}ms)`;
      return style(line, event.isError ? ANSI_RED : ANSI_DIM, color);
    }

    case "deadline_hit":
      return style(
        `[deadline] soft deadline (${event.softDeadlineMs}ms) reached after ${event.turnsCompleted} turns`,
        ANSI_YELLOW,
        color
      );

    case "synthesis_start":
      return `\n${style(`--- Synthesis (${event.model}, docs=${event.fetchedDocCount}, filters=${event.filterSnapshotCount}) ---`, ANSI_BOLD, color)}`;

    case "synthesis_thinking":
      return style(`[thinking]\n${indentLines(event.text)}`, ANSI_DIM, color);

    case "citations_validated":
      if (event.strippedDocIds.length === 0) {
        return null;
      }
      return style(
        `[citations] kept ${event.validDocIds.length}, stripped ${event.strippedDocIds.length}: ${event.strippedDocIds.join(", ")}`,
        ANSI_YELLOW,
        color
      );

    case "answer_complete":
      return null;
  }
}

export function buildAnswerJsonPayload(input: {
  readonly question: string;
  readonly profile: RunProfile;
  readonly result: AnswerResult;
  readonly latencyMs: number;
}): Record<string, unknown> {
  const estimatedCostUsd = estimateAnswerCostUsd(input.profile, input.result);
  return {
    question: input.question,
    profile: input.profile.name,
    answer: input.result.response.answer,
    abstained: input.result.abstained,
    document_ids: input.result.response.document_ids,
    stripped_doc_ids: input.result.strippedDocIds,
    agent_turns: input.result.agentTurns,
    deadline_hit: input.result.deadlineHit,
    latency_ms: Math.round(input.latencyMs),
    usage: input.result.usage,
    embedding_input_tokens: input.result.embeddingInputTokens,
    estimated_cost_usd: Number(estimatedCostUsd.toFixed(6)),
    ...(input.result.traceId !== undefined ? { trace_id: input.result.traceId } : {})
  };
}

export function formatAnswerSummary(input: AskAnswerSummaryInput): string {
  const color = input.color ?? false;
  const citations =
    input.result.response.document_ids.length > 0
      ? input.result.response.document_ids.join(", ")
      : "(none)";

  if (input.quiet) {
    return `${input.result.response.answer}\nCitations: ${citations}`;
  }

  const estimatedCostUsd = estimateAnswerCostUsd(input.profile, input.result);
  const lines: string[] = [
    "",
    style("Answer:", ANSI_GREEN, color),
    input.result.response.answer,
    "",
    `Abstained: ${input.result.abstained ? "yes" : "no"}`,
    `Citations (${input.result.response.document_ids.length}): ${citations}`
  ];

  if (input.result.strippedDocIds.length > 0) {
    lines.push(
      style(
        `Stripped citations (${input.result.strippedDocIds.length}): ${input.result.strippedDocIds.join(", ")}`,
        ANSI_YELLOW,
        color
      )
    );
  }

  lines.push(
    `Stats: latency=${Math.round(input.latencyMs)}ms | turns=${input.result.agentTurns} | tokens in=${input.result.usage.inputTokens} out=${input.result.usage.outputTokens} embed=${input.result.embeddingInputTokens} | est_cost=$${estimatedCostUsd.toFixed(4)}`
  );

  if (input.result.traceId !== undefined) {
    lines.push(`Langfuse traceId: ${input.result.traceId}`);
  }

  return lines.join("\n");
}

async function readAllStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk);
  }
  return Buffer.concat(chunks).toString("utf8").trim();
}

interface ActiveStoreSession {
  profile: RunProfile;
  indexHash: string;
  dbPath: string;
  store: KnowledgeStore;
  embedder: Embedder | undefined;
}

async function openStoreSession(params: {
  profile: RunProfile;
  corpusDir: string;
  explicitDbPath?: string;
  cacheDir: string;
  geminiApiKey?: string;
  embedderOverride?: Embedder;
}): Promise<ActiveStoreSession> {
  const indexHash = computeIndexConfigHash(params.profile);
  const dbPath =
    params.explicitDbPath !== undefined
      ? resolve(params.explicitDbPath)
      : resolveCachedDbPath(params.cacheDir, params.corpusDir, params.profile);
  const store = new KnowledgeStore(dbPath);

  let embedder: Embedder | undefined = params.embedderOverride;
  if (embedder === undefined && params.profile.enableEmbeddings) {
    if (params.geminiApiKey === undefined || params.geminiApiKey.trim().length === 0) {
      store.close();
      throw new ConfigError(
        "GEMINI_API_KEY is required when profile.enableEmbeddings is true."
      );
    }
    embedder = new GeminiEmbedder({
      apiKey: params.geminiApiKey,
      model: params.profile.embeddingModel,
      dims: params.profile.embeddingDims
    });
  }

  try {
    const reader = createFilesystemCorpusReader(params.corpusDir);
    await ingestCorpus(
      reader,
      store,
      params.profile,
      embedder !== undefined ? { embedder } : {}
    );
  } catch (error) {
    store.close();
    throw error;
  }

  return {
    profile: params.profile,
    indexHash,
    dbPath,
    store,
    embedder
  };
}

function writeStartupBanner(
  session: ActiveStoreSession,
  telemetry: TelemetryController,
  stdout: Pick<NodeJS.WriteStream, "write">,
  color: boolean
): void {
  const stats = session.store.getStats();
  const graphStats = session.store.getGraphStats();
  const vectorStats = session.store.getVectorCacheStats();
  const summary = formatStartupSummary(
    {
      profile: session.profile,
      indexConfigHash: session.indexHash,
      dbPath: session.dbPath,
      documentCount: stats.documentCount,
      chunkCount: stats.chunkCount,
      sourceCount: stats.sourceCount,
      graphEdgeCount: graphStats.edgeCount,
      graphResolvedEdgeCount: graphStats.resolvedEdgeCount,
      vectorCount: vectorStats.vectorCount,
      langfuseEnabled: telemetry.enabled,
      ...(telemetry.config?.baseUrl !== undefined
        ? { langfuseBaseUrl: telemetry.config.baseUrl }
        : {})
    },
    color
  );
  stdout.write(`${summary}\n`);
}

export async function runAskCli(
  argv: readonly string[],
  overrides?: AskCliOverrides
): Promise<void> {
  const args = parseAskCliArgs(argv);
  const env = overrides?.env ?? process.env;
  const cwd = overrides?.cwd ?? process.cwd();
  const stdout = overrides?.stdout ?? process.stdout;
  const stderr = overrides?.stderr ?? process.stderr;
  const color = Boolean(stdout.isTTY) && !args.json;
  const geminiApiKey = env.GEMINI_API_KEY?.trim();

  if (
    overrides?.llmProvider === undefined &&
    (geminiApiKey === undefined || geminiApiKey.length === 0)
  ) {
    throw new ConfigError("Missing GEMINI_API_KEY in environment or .env.");
  }

  const telemetry =
    overrides?.telemetry ??
    (await createTelemetryController(env, {
      serviceName: "sqlite-rag-bench-ask",
      exportMode: "immediate"
    }));

  const llmProvider =
    overrides?.llmProvider ??
    new GeminiLlmProvider({
      apiKey: geminiApiKey ?? "",
      telemetry
    });

  const corpusDir = resolve(cwd, args.corpusDir);
  const cacheDir = resolve(cwd, ".cache");
  const initialProfile = loadRunProfile(args.profile, args.setOverrides);

  let session: ActiveStoreSession | null = null;
  try {
    session = await openStoreSession({
      profile: initialProfile,
      corpusDir,
      ...(args.dbPath !== undefined ? { explicitDbPath: resolve(cwd, args.dbPath) } : {}),
      cacheDir,
      ...(geminiApiKey !== undefined ? { geminiApiKey } : {}),
      ...(overrides?.embedder !== undefined ? { embedderOverride: overrides.embedder } : {})
    });

    if (!args.quiet && !args.json) {
      writeStartupBanner(session, telemetry, stdout, color);
    }

    const executeQuestion = async (question: string): Promise<void> => {
      if (session === null) {
        return;
      }
      const startedAt = performance.now();
      const result = await answerQuestion(
        session.store,
        llmProvider,
        session.profile,
        question,
        {
          ...(session.embedder !== undefined ? { embedder: session.embedder } : {}),
          telemetry,
          ...(!args.quiet && !args.json
            ? {
                onEvent(event: AnswerTraceEvent): void {
                  const formatted = formatTraceEvent(event, { color });
                  if (formatted !== null) {
                    stdout.write(`${formatted}\n`);
                  }
                }
              }
            : {})
        }
      );
      const latencyMs = Math.max(0, performance.now() - startedAt);
      await telemetry.flush();

      if (args.json) {
        stdout.write(
          `${JSON.stringify(
            buildAnswerJsonPayload({
              question,
              profile: session.profile,
              result,
              latencyMs
            })
          )}\n`
        );
        return;
      }

      stdout.write(
        `${formatAnswerSummary({
          question,
          profile: session.profile,
          result,
          latencyMs,
          quiet: args.quiet,
          color
        })}\n`
      );
    };

    if (args.question !== undefined) {
      await executeQuestion(args.question);
      return;
    }

    if (!process.stdin.isTTY) {
      const pipedQuestion = await readAllStdin();
      if (pipedQuestion.length === 0) {
        throw new ConfigError(
          "Provide a question argument or run in an interactive TTY."
        );
      }
      await executeQuestion(pipedQuestion);
      return;
    }

    if (!args.quiet && !args.json) {
      stdout.write("\nInteractive mode. Commands: /profile <name>, /quit, /exit\n");
    }

    const rl = createInterface({
      input: process.stdin,
      output: process.stdout
    });
    try {
      for (;;) {
        let line: string;
        try {
          line = await rl.question("\nask> ");
        } catch {
          break;
        }
        const trimmed = line.trim();
        if (trimmed.length === 0) {
          continue;
        }
        if (trimmed === "/quit" || trimmed === "/exit") {
          break;
        }
        if (trimmed.startsWith("/profile")) {
          const nextName = trimmed.slice("/profile".length).trim();
          if (nextName.length === 0) {
            stderr.write("Usage: /profile <name>\n");
            continue;
          }
          try {
            const nextProfile = loadRunProfile(nextName, args.setOverrides);
            const nextHash = computeIndexConfigHash(nextProfile);
            if (nextHash !== session.indexHash) {
              const previousSession = session;
              const nextSession = await openStoreSession({
                profile: nextProfile,
                corpusDir,
                ...(args.dbPath !== undefined
                  ? { explicitDbPath: resolve(cwd, args.dbPath) }
                  : {}),
                cacheDir,
                ...(geminiApiKey !== undefined ? { geminiApiKey } : {}),
                ...(overrides?.embedder !== undefined
                  ? { embedderOverride: overrides.embedder }
                  : {})
              });
              previousSession.store.close();
              session = nextSession;
            } else {
              session.profile = nextProfile;
            }
            if (!args.quiet && !args.json) {
              writeStartupBanner(session, telemetry, stdout, color);
            }
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            stderr.write(`Failed to switch profile: ${message}\n`);
          }
          continue;
        }

        try {
          await executeQuestion(trimmed);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          stderr.write(`Error: ${message}\n`);
        }
      }
    } finally {
      rl.close();
    }
  } finally {
    session?.store.close();
    await telemetry.flush();
    await telemetry.shutdown();
  }
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  loadDotEnvIfPresent();
  runAskCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`ask failed: ${message}\n`);
    process.exitCode = 1;
  });
}
