import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { loadDotEnvIfPresent } from "../adapters/dotenv.js";
import { AiSdkJudgeAdapter, type JudgeAdapter, type JudgeInput } from "../adapters/judge.js";
import { readJsonlFile } from "../adapters/jsonl.js";
import { LiveHttpResponder } from "../adapters/responder.js";
import { SpendLedgerStore } from "../adapters/spend-ledger.js";
import {
  DEFAULT_SYSTEMONE_BASE_URL,
  DEFAULT_SYSTEMONE_MODEL,
  SystemOneDecisionJudgeAdapter
} from "../adapters/systemone-judge.js";
import { ConfigError, ValidationError } from "../models/errors.js";
import { DevQuestionSchema, SplitManifestSchema } from "../models/eval.js";
import {
  EvalRunManifestSchema,
  EvalSplitNameSchema,
  HoldoutLockedError,
  JudgeTierSchema,
  type JudgeTier
} from "../models/eval-harness.js";
import {
  calibrateJudge,
  DEFAULT_EVAL_BUDGET_USD,
  DEFAULT_MIN_CALIBRATION_AGREEMENT
} from "../usecases/eval-runner.js";

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

type CalibrationSplitName = "dev" | "val";

function parseSplit(raw: string): CalibrationSplitName {
  const result = EvalSplitNameSchema.safeParse(raw);
  if (!result.success) {
    throw new ConfigError(`Invalid --split "${raw}". Expected dev or val.`);
  }
  if (result.data === "holdout") {
    throw new HoldoutLockedError(
      'Judge calibration cannot run on the locked "holdout" split. Use --split dev or --split val.'
    );
  }
  return result.data;
}

function parseJudgeTier(raw: string): JudgeTier {
  const result = JudgeTierSchema.safeParse(raw);
  if (!result.success) {
    throw new ConfigError(`Invalid --judge-tier "${raw}". Expected fast, frontier, or decision.`);
  }
  return result.data;
}

type CalibrateJudgeProvider = "ai-sdk" | "systemone";

function parseJudgeProvider(raw: string): CalibrateJudgeProvider {
  if (raw === "ai-sdk" || raw === "systemone") {
    return raw;
  }
  throw new ConfigError(`Invalid --judge-provider "${raw}". Expected ai-sdk or systemone.`);
}

export async function runCalibrateJudgeCli(argv: readonly string[]): Promise<void> {
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    options: {
      judge: { type: "string" },
      "judge-tier": { type: "string" },
      "judge-provider": { type: "string", default: "ai-sdk" },
      "systemone-url": { type: "string" },
      "systemone-key": { type: "string" },
      "frontier-judge": { type: "string", default: "gemini-2.5-pro" },
      url: { type: "string", default: "http://127.0.0.1:8080" },
      "run-file": { type: "string" },
      split: { type: "string", default: "dev" },
      only: { type: "string" },
      limit: { type: "string", default: "20" },
      budget: { type: "string" },
      questions: { type: "string", default: "data/questions.jsonl" },
      "splits-file": { type: "string", default: "eval/splits.json" },
      ledger: { type: "string", default: "eval/spend-ledger.jsonl" },
      output: { type: "string" }
    }
  });

  const split = parseSplit(values.split);
  const judgeProvider = parseJudgeProvider(values["judge-provider"]);
  if (
    judgeProvider === "systemone" &&
    values["judge-tier"] !== undefined &&
    values["judge-tier"] !== "decision"
  ) {
    throw new ConfigError(
      `Judge provider "systemone" requires tier "decision", got --judge-tier "${values["judge-tier"]}".`
    );
  }

  const candidateModel =
    values.judge ??
    (judgeProvider === "systemone" ? DEFAULT_SYSTEMONE_MODEL : "gemini-3.5-flash-lite");
  const judgeTier =
    judgeProvider === "systemone"
      ? "decision"
      : parseJudgeTier(values["judge-tier"] ?? "fast");
  const limit = parsePositiveNumber(values.limit, "--limit") ?? 20;
  const budgetUsd =
    parsePositiveNumber(values.budget ?? process.env.EVAL_BUDGET_USD, "--budget") ??
    DEFAULT_EVAL_BUDGET_USD;
  const spendLedger = new SpendLedgerStore(resolve(values.ledger));
  spendLedger.assertWithinBudget({ maxTotalUsd: budgetUsd });

  const questions = readJsonlFile(resolve(values.questions), DevQuestionSchema);
  const questionsById = new Map(questions.map((question) => [question.question_id, question]));

  const onlySet =
    values.only !== undefined
      ? new Set(
          values.only
            .split(",")
            .map((item) => item.trim())
            .filter((item) => item.length > 0)
        )
      : undefined;

  const judgeInputs: JudgeInput[] = [];
  if (values["run-file"] !== undefined) {
    const manifest = loadJsonFile(resolve(values["run-file"]), (raw) =>
      EvalRunManifestSchema.parse(raw)
    );
    if (manifest.split === "holdout") {
      throw new HoldoutLockedError(
        'Judge calibration cannot read a run manifest from the locked "holdout" split.'
      );
    }
    for (const item of manifest.items) {
      if (onlySet !== undefined && !onlySet.has(item.questionId)) {
        continue;
      }
      const question = questionsById.get(item.questionId);
      if (question === undefined) {
        throw new ValidationError(
          `Question "${item.questionId}" from run file is missing in ${values.questions}.`
        );
      }
      judgeInputs.push({
        question: question.question,
        goldAnswer: question.gold_answer,
        answerFacts: question.answer_facts,
        expectedDocIds: question.expected_doc_ids,
        responderAnswer: item.answer,
        responderDocIds: item.returnedDocumentIds
      });
      if (judgeInputs.length >= limit) {
        break;
      }
    }
  } else {
    const splitManifest = loadJsonFile(resolve(values["splits-file"]), (raw) =>
      SplitManifestSchema.parse(raw)
    );
    const responder = new LiveHttpResponder({
      baseUrl: values.url,
      modelId: "live-http-responder"
    });
    for (const questionId of splitManifest[split]) {
      if (onlySet !== undefined && !onlySet.has(questionId)) {
        continue;
      }
      const question = questionsById.get(questionId);
      if (question === undefined) {
        throw new ValidationError(`Question "${questionId}" from split "${split}" is missing.`);
      }
      const response = await responder.answer(question.question);
      judgeInputs.push({
        question: question.question,
        goldAnswer: question.gold_answer,
        answerFacts: question.answer_facts,
        expectedDocIds: question.expected_doc_ids,
        responderAnswer: response.answer,
        responderDocIds: response.documentIds
      });
      if (judgeInputs.length >= limit) {
        break;
      }
    }
  }

  const safeJudgeName = candidateModel.replace(/[^a-zA-Z0-9._-]+/g, "_");
  const outputPath = resolve(values.output ?? `eval/calibrations/${safeJudgeName}.json`);

  let candidateJudge: JudgeAdapter;
  if (judgeProvider === "systemone") {
    const systemOneUrl =
      values["systemone-url"] ?? process.env.SYSTEMONE_BASE_URL ?? DEFAULT_SYSTEMONE_BASE_URL;
    const systemOneKey = values["systemone-key"] ?? process.env.SYSTEMONE_API_KEY;
    candidateJudge = new SystemOneDecisionJudgeAdapter({
      modelId: candidateModel,
      baseUrl: systemOneUrl,
      ...(systemOneKey !== undefined ? { apiKey: systemOneKey } : {})
    });
  } else {
    candidateJudge = new AiSdkJudgeAdapter({
      modelId: candidateModel,
      tier: judgeTier
    });
  }

  const frontierJudge = new AiSdkJudgeAdapter({
    modelId: values["frontier-judge"],
    tier: "frontier"
  });

  const report = await calibrateJudge(candidateJudge, frontierJudge, judgeInputs, {
    spendLedger,
    budgetUsd,
    split,
    minAgreement: DEFAULT_MIN_CALIBRATION_AGREEMENT,
    outputPath
  });

  const statusLabel = report.passed ? "PASS" : "FAIL";
  process.stdout.write(
    [
      `Judge calibration ${statusLabel}: ${report.judgeModel} vs ${report.frontierJudgeModel}`,
      `Sample size: ${report.sampleSize}`,
      `Contradiction agreement: ${(report.contradictionAgreement * 100).toFixed(1)}%`,
      `Saved report: ${outputPath}`
    ].join("\n") + "\n"
  );

  if (!report.passed) {
    process.exitCode = 1;
  }
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  loadDotEnvIfPresent();
  runCalibrateJudgeCli(process.argv.slice(2)).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`calibrate-judge failed: ${message}\n`);
    process.exitCode = 1;
  });
}
