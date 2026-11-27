import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { createFilesystemCorpusReader } from "../adapters/corpus-reader.js";
import { loadDotEnvIfPresent } from "../adapters/dotenv.js";
import { GeminiEmbedder } from "../adapters/embedder.js";
import {
  DEFAULT_FRONTIER_BASELINE_MODEL,
  FrontierOneShotResponder
} from "../adapters/frontier-baseline.js";
import { AiSdkJudgeAdapter, type JudgeAdapter } from "../adapters/judge.js";
import { readJsonlFile } from "../adapters/jsonl.js";
import { KnowledgeStore } from "../adapters/knowledge-store.js";
import { LangfuseExperimentExporter } from "../adapters/langfuse-exporter.js";
import { GeminiLlmProvider } from "../adapters/llm-provider.js";
import { loadRunProfile } from "../adapters/profile-loader.js";
import { LiveHttpResponder, type ResponderAdapter } from "../adapters/responder.js";
import { SpendLedgerStore } from "../adapters/spend-ledger.js";
import {
  DEFAULT_SYSTEMONE_BASE_URL,
  DEFAULT_SYSTEMONE_MODEL,
  SystemOneDecisionJudgeAdapter
} from "../adapters/systemone-judge.js";
import { createTelemetryController } from "../adapters/telemetry.js";
import {
  TerminalScorecardExporter,
  type ObservabilityExporter
} from "../adapters/terminal-scorecard.js";
import { ConfigError, ValidationError } from "../models/errors.js";
import {
  DevQuestionSchema,
  QuestionStratumSchema,
  SplitManifestSchema,
  type QuestionStratum
} from "../models/eval.js";
import {
  CalibrationReportSchema,
  EvalSplitNameSchema,
  JudgeTierSchema,
  type CalibrationReport,
  type EvalSplitName,
  type JudgeTier,
  type SpendBucket
} from "../models/eval-harness.js";
import { answerQuestion } from "../usecases/answer.js";
import { DEFAULT_EVAL_BUDGET_USD, runEvaluation } from "../usecases/eval-runner.js";
import { ingestCorpus, resolveCachedDbPath } from "../usecases/ingest.js";
import { estimateTokenCostUsd, normalizeModelIdentifier } from "../utils/scoring.js";

const KNOWN_FRONTIER_JUDGES = new Set([
  "gemini-4-neon",
  "gemini-4-carbon",
  "gemini-4-argon",
  "gemini-3.1-pro-preview",
  "gemini-3.1-pro-preview-customtools",
  "gemini-2.5-pro",
  "gpt-4.1",
  "gpt-4o",
  "o3"
]);

function parsePositiveInteger(raw: string | undefined, flagName: string): number | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new ConfigError(`Flag ${flagName} must be a positive integer, got "${raw}".`);
  }
  return value;
}

function parsePositiveNumber(raw: string | undefined, flagName: string): number | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new ConfigError(`Flag ${flagName} must be a positive number, got "${raw}".`);
  }
  return value;
}

export function inferJudgeTier(modelId: string): JudgeTier {
  const normalized = normalizeModelIdentifier(modelId);
  if (normalized.includes("clef") || normalized.includes("jev")) {
    return "decision";
  }
  if (KNOWN_FRONTIER_JUDGES.has(normalized)) {
    return "frontier";
  }
  return "fast";
}

function readGitSha(): string {
  const result = spawnSync("git", ["rev-parse", "--short", "HEAD"], { encoding: "utf8" });
  if (result.status === 0 && typeof result.stdout === "string" && result.stdout.trim().length > 0) {
    return result.stdout.trim();
  }
  return "unknown";
}

function loadJsonFile<T>(filePath: string, parse: (input: unknown) => T): T {
  const raw = readFileSync(filePath, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new ValidationError(`Invalid JSON in ${filePath}`, { cause: error });
  }
  return parse(parsed);
}

function loadCorpusUuidSet(filePath: string): ReadonlySet<string> | undefined {
  if (!existsSync(filePath)) {
    return undefined;
  }
  return loadJsonFile(filePath, (raw) => {
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new ValidationError(`Expected object map in ${filePath}`);
    }
    return new Set(Object.keys(raw));
  });
}

function parseSplit(raw: string): EvalSplitName {
  const result = EvalSplitNameSchema.safeParse(raw);
  if (!result.success) {
    throw new ConfigError(`Invalid --split "${raw}". Expected dev, val, or holdout.`);
  }
  return result.data;
}

function parseStratum(raw: string | undefined): QuestionStratum | undefined {
  if (raw === undefined) {
    return undefined;
  }
  const result = QuestionStratumSchema.safeParse(raw);
  if (!result.success) {
    throw new ConfigError(
      `Invalid --stratum "${raw}". Expected unanswerable, single_doc, two_doc, or multi_doc.`
    );
  }
  return result.data;
}

function parseJudgeTier(raw: string | undefined, modelId: string): JudgeTier {
  if (raw === undefined) {
    return inferJudgeTier(modelId);
  }
  const result = JudgeTierSchema.safeParse(raw);
  if (!result.success) {
    throw new ConfigError(`Invalid --judge-tier "${raw}". Expected fast, frontier, or decision.`);
  }
  return result.data;
}

type EvalJudgeProvider = "ai-sdk" | "systemone";
type EvalResponderMode = "http" | "frontier-oneshot";

function parseJudgeProvider(raw: string): EvalJudgeProvider {
  if (raw === "ai-sdk" || raw === "systemone") {
    return raw;
  }
  throw new ConfigError(`Invalid --judge-provider "${raw}". Expected ai-sdk or systemone.`);
}

function parseResponderMode(raw: string): EvalResponderMode {
  if (raw === "http" || raw === "frontier-oneshot") {
    return raw;
  }
  throw new ConfigError(`Invalid --responder "${raw}". Expected http or frontier-oneshot.`);
}

export interface EvalCliOverrides {
  readonly inProcessResponder?: ResponderAdapter;
  readonly env?: NodeJS.ProcessEnv;
}

export async function runEvalCli(
  argv: readonly string[],
  overrides?: EvalCliOverrides
): Promise<void> {
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    options: {
      profile: { type: "string", default: "l0-fts" },
      set: { type: "string", multiple: true, default: [] },
      split: { type: "string", default: "dev" },
      only: { type: "string" },
      stratum: { type: "string" },
      concurrency: { type: "string" },
      url: { type: "string", default: "http://127.0.0.1:8080" },
      responder: { type: "string", default: "http" },
      "baseline-model": { type: "string" },
      budget: { type: "string" },
      "max-judge-usd": { type: "string" },
      "max-responder-usd": { type: "string" },
      judge: { type: "string" },
      "judge-tier": { type: "string" },
      "judge-provider": { type: "string", default: "ai-sdk" },
      "systemone-url": { type: "string" },
      "systemone-key": { type: "string" },
      "responder-model": { type: "string" },
      "unlock-holdout": { type: "boolean", default: false },
      "calibration-file": { type: "string" },
      "in-process": { type: "boolean", default: false },
      corpus: { type: "string", default: "data/corpus" },
      db: { type: "string" },
      questions: { type: "string", default: "data/questions.jsonl" },
      "splits-file": { type: "string", default: "eval/splits.json" },
      "uuid-index": { type: "string", default: "data/uuid_index.json" },
      ledger: { type: "string", default: "eval/spend-ledger.jsonl" },
      "runs-dir": { type: "string", default: "eval/runs" }
    }
  });

  const profile = loadRunProfile(values.profile, values.set);
  const split = parseSplit(values.split);
  const stratumFilter = parseStratum(values.stratum);
  const judgeProvider = parseJudgeProvider(values["judge-provider"]);
  const responderMode = parseResponderMode(values.responder);
  const onlyIds =
    values.only !== undefined
      ? values.only
          .split(",")
          .map((item) => item.trim())
          .filter((item) => item.length > 0)
      : undefined;

  const env = overrides?.env ?? process.env;
  const budgetUsd =
    parsePositiveNumber(values.budget ?? env.EVAL_BUDGET_USD, "--budget") ??
    DEFAULT_EVAL_BUDGET_USD;
  const concurrency = parsePositiveInteger(values.concurrency, "--concurrency") ?? 1;
  const maxJudgeUsd = parsePositiveNumber(values["max-judge-usd"], "--max-judge-usd");
  const maxResponderUsd = parsePositiveNumber(values["max-responder-usd"], "--max-responder-usd");

  const maxBucketUsd: Partial<Record<SpendBucket, number>> = {};
  if (maxJudgeUsd !== undefined) {
    maxBucketUsd.judge = maxJudgeUsd;
  }
  if (maxResponderUsd !== undefined) {
    maxBucketUsd.responder = maxResponderUsd;
  }

  if (values["in-process"] && responderMode !== "http") {
    throw new ConfigError(
      `Cannot combine --in-process with --responder "${responderMode}".`
    );
  }

  const telemetry = await createTelemetryController(env, {
    serviceName: "sqlite-rag-bench-eval",
    exportMode: "immediate"
  });
  let langfuseExporter: LangfuseExperimentExporter | undefined;
  let inProcessStore: KnowledgeStore | undefined;

  try {
    const spendLedger = new SpendLedgerStore(resolve(values.ledger));
    let responder: ResponderAdapter;
    if (values["in-process"]) {
      if (overrides?.inProcessResponder !== undefined) {
        responder = overrides.inProcessResponder;
      } else {
        const geminiApiKey = env.GEMINI_API_KEY?.trim();
        if (geminiApiKey === undefined || geminiApiKey.length === 0) {
          throw new ConfigError(
            "Flag --in-process requires GEMINI_API_KEY in the environment."
          );
        }
        const corpusDir = resolve(values.corpus);
        const dbPath =
          values.db !== undefined
            ? resolve(values.db)
            : resolveCachedDbPath(resolve(process.cwd(), ".cache"), corpusDir, profile);
        inProcessStore = new KnowledgeStore(dbPath);
        const embedder = profile.enableEmbeddings
          ? new GeminiEmbedder({
              apiKey: geminiApiKey,
              model: profile.embeddingModel,
              dims: profile.embeddingDims
            })
          : undefined;
        if (profile.enableEmbeddings) {
          spendLedger.assertWithinBudget({
            maxTotalUsd: budgetUsd,
            ...(Object.keys(maxBucketUsd).length > 0 ? { maxBucketUsd } : {})
          });
        }
        await ingestCorpus(
          createFilesystemCorpusReader(corpusDir),
          inProcessStore,
          profile,
          embedder !== undefined
            ? {
                embedder,
                onEmbeddingUsage: (event) => {
                  spendLedger.record({
                    timestamp: new Date().toISOString(),
                    runId: `ingest_${profile.name}`,
                    split,
                    bucket: "embedding",
                    model: event.model,
                    inputTokens: event.inputTokens,
                    outputTokens: 0,
                    usd: estimateTokenCostUsd(event.model, event.inputTokens, 0)
                  });
                }
              }
            : {}
        );
        const llmProvider = new GeminiLlmProvider({
          apiKey: geminiApiKey,
          telemetry
        });
        const activeStore = inProcessStore;
        responder = {
          modelId: values["responder-model"] ?? profile.synthesisModel,
          async answer(question, callOptions) {
            const startedAt = performance.now();
            const result = await answerQuestion(activeStore, llmProvider, profile, question, {
              ...(callOptions?.signal !== undefined ? { signal: callOptions.signal } : {}),
              ...(embedder !== undefined ? { embedder } : {}),
              telemetry
            });
            const latencyMs = Math.max(0, performance.now() - startedAt);
            const responseBytes = Buffer.byteLength(
              JSON.stringify(result.response),
              "utf8"
            );
            return {
              answer: result.response.answer,
              documentIds: result.response.document_ids,
              latencyMs,
              responseBytes,
              usage: result.usage
            };
          }
        };
      }
    } else if (responderMode === "frontier-oneshot") {
      const baselineModel =
        values["baseline-model"] ??
        values["responder-model"] ??
        env.EVAL_BASELINE_MODEL ??
        DEFAULT_FRONTIER_BASELINE_MODEL;
      responder = new FrontierOneShotResponder({
        modelId: baselineModel
      });
    } else {
      responder = new LiveHttpResponder({
        baseUrl: values.url,
        modelId: values["responder-model"] ?? profile.synthesisModel
      });
    }

    let judge: JudgeAdapter;
    if (judgeProvider === "systemone") {
      if (values["judge-tier"] !== undefined && values["judge-tier"] !== "decision") {
        throw new ConfigError(
          `Judge provider "systemone" requires tier "decision", got --judge-tier "${values["judge-tier"]}".`
        );
      }
      const judgeModel = values.judge ?? env.EVAL_JUDGE_MODEL ?? DEFAULT_SYSTEMONE_MODEL;
      const systemOneUrl =
        values["systemone-url"] ?? env.SYSTEMONE_BASE_URL ?? DEFAULT_SYSTEMONE_BASE_URL;
      const systemOneKey = values["systemone-key"] ?? env.SYSTEMONE_API_KEY;
      judge = new SystemOneDecisionJudgeAdapter({
        modelId: judgeModel,
        baseUrl: systemOneUrl,
        ...(systemOneKey !== undefined ? { apiKey: systemOneKey } : {})
      });
    } else {
      const judgeModel = values.judge ?? env.EVAL_JUDGE_MODEL ?? "gemini-2.5-pro";
      const judgeTier = parseJudgeTier(values["judge-tier"], judgeModel);
      judge = new AiSdkJudgeAdapter({
        modelId: judgeModel,
        tier: judgeTier,
        telemetry
      });
    }

    let calibrationReport: CalibrationReport | undefined;
    if (values["calibration-file"] !== undefined) {
      calibrationReport = loadJsonFile(resolve(values["calibration-file"]), (raw) =>
        CalibrationReportSchema.parse(raw)
      );
    }

    const questions = readJsonlFile(resolve(values.questions), DevQuestionSchema);
    const splitManifest = loadJsonFile(resolve(values["splits-file"]), (raw) =>
      SplitManifestSchema.parse(raw)
    );
    const validCorpusIds = loadCorpusUuidSet(resolve(values["uuid-index"]));

    const exporters: ObservabilityExporter[] = [new TerminalScorecardExporter()];
    if (telemetry.enabled) {
      langfuseExporter = new LangfuseExperimentExporter({ env, telemetry });
      exporters.push(langfuseExporter);
    }

    await runEvaluation(
      {
        responder,
        judge,
        spendLedger,
        exporters,
        ...(langfuseExporter !== undefined ? { experimentWrapper: langfuseExporter } : {}),
        runsDir: resolve(values["runs-dir"]),
        gitSha: readGitSha(),
        ...(validCorpusIds !== undefined ? { validCorpusIds } : {})
      },
      {
        profile,
        split,
        splitManifest,
        questions,
        budgetUsd,
        concurrency,
        unlockHoldout: values["unlock-holdout"],
        ...(Object.keys(maxBucketUsd).length > 0 ? { maxBucketUsd } : {}),
        ...(calibrationReport !== undefined ? { calibrationReport } : {}),
        ...(onlyIds !== undefined && onlyIds.length > 0 ? { onlyIds } : {}),
        ...(stratumFilter !== undefined ? { stratumFilter } : {})
      }
    );
  } finally {
    inProcessStore?.close();
    await langfuseExporter?.flush();
    await langfuseExporter?.shutdown();
    await telemetry.flush();
    await telemetry.shutdown();
  }
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  loadDotEnvIfPresent();
  runEvalCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`eval failed: ${message}\n`);
    process.exitCode = 1;
  });
}
