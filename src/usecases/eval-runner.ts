import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { JudgeAdapter, JudgeInput } from "../adapters/judge.js";
import type { ResponderAdapter, ResponderCallOptions } from "../adapters/responder.js";
import type { BudgetCheckOptions, SpendLedgerStore } from "../adapters/spend-ledger.js";
import type {
  ExperimentItemExecutionContext,
  ExperimentRunContext,
  ExperimentRunnerWrapper,
  ObservabilityExporter
} from "../adapters/terminal-scorecard.js";
import { ValidationError } from "../models/errors.js";
import type { DevQuestion, QuestionStratum, SplitManifest } from "../models/eval.js";
import {
  CalibrationReportSchema,
  EvalRunManifestSchema,
  HoldoutLockedError,
  JudgePolicyError,
  PromotionDecisionSchema,
  type CalibrationReport,
  type EvalItemResult,
  type EvalRunManifest,
  type EvalSplitName,
  type PromotionDecision,
  type SpendBucket
} from "../models/eval-harness.js";
import { computeIndexConfigHash, type RunProfile } from "../models/profile.js";
import {
  aggregateSuiteMetrics,
  computeEffectiveAnswerScore,
  estimateTokenCostUsd,
  normalizeModelIdentifier,
  scoreCitations,
  scoreComposite
} from "../utils/scoring.js";
import { classifyQuestionStratum } from "./splits.js";

export const DEFAULT_EVAL_BUDGET_USD = 40;
export const DEFAULT_MIN_CALIBRATION_AGREEMENT = 0.9;
export const DEFAULT_PROMOTION_NOISE_BAND = 0.02;
export const DEFAULT_MAX_LATENCY_P95_MS = 60_000;

export interface EvalRunnerDeps {
  readonly responder: ResponderAdapter;
  readonly judge: JudgeAdapter;
  readonly spendLedger: SpendLedgerStore;
  readonly exporters: readonly ObservabilityExporter[];
  readonly experimentWrapper?: ExperimentRunnerWrapper;
  readonly getItemRequestHeaders?: (
    question: DevQuestion
  ) => Readonly<Record<string, string>>;
  readonly runsDir: string;
  readonly now?: () => Date;
  readonly gitSha?: string;
  readonly validCorpusIds?: ReadonlySet<string>;
}

export interface RunEvaluationOptions {
  readonly profile: RunProfile;
  readonly split: EvalSplitName;
  readonly splitManifest: SplitManifest;
  readonly questions: readonly DevQuestion[];
  readonly budgetUsd?: number;
  readonly maxBucketUsd?: Readonly<Partial<Record<SpendBucket, number>>>;
  readonly estimatedRunCostUsd?: number;
  readonly unlockHoldout?: boolean;
  readonly allowRepeatHoldout?: boolean;
  readonly calibrationReport?: CalibrationReport;
  readonly onlyIds?: readonly string[];
  readonly stratumFilter?: QuestionStratum;
  readonly concurrency?: number;
  readonly requestHeaders?: Readonly<Record<string, string>>;
  readonly runId?: string;
}

export interface CalibrateJudgeDeps {
  readonly spendLedger: SpendLedgerStore;
  readonly budgetUsd?: number;
  readonly minAgreement?: number;
  readonly split?: EvalSplitName;
  readonly outputPath?: string;
  readonly runId?: string;
  readonly now?: () => Date;
}

export interface PromotionGateOptions {
  readonly noiseBand?: number;
  readonly maxLatencyP95Ms?: number;
}

function buildRunId(profileName: string, split: EvalSplitName, now: Date): string {
  const safeProfile = profileName.replace(/[^a-zA-Z0-9_-]+/g, "-");
  const stamp = now.toISOString().replace(/[:.]/g, "-");
  return `${stamp}_${split}_${safeProfile}`;
}

function assertJudgeDistinctFromResponder(judgeModelId: string, responderModelId: string): void {
  if (normalizeModelIdentifier(judgeModelId) === normalizeModelIdentifier(responderModelId)) {
    throw new JudgePolicyError(
      `Judge model "${judgeModelId}" must differ from Responder model "${responderModelId}".`
    );
  }
}

function assertHoldoutPolicy(
  split: EvalSplitName,
  spendLedger: SpendLedgerStore,
  unlockHoldout: boolean,
  allowRepeatHoldout: boolean
): void {
  if (split !== "holdout") {
    return;
  }
  if (!unlockHoldout) {
    throw new HoldoutLockedError(
      "Holdout split is locked. Pass --unlock-holdout only for final evaluation."
    );
  }
  if (!allowRepeatHoldout) {
    const priorHoldoutEntry = spendLedger
      .readEntries()
      .some((entry) => entry.split === "holdout");
    if (priorHoldoutEntry) {
      throw new HoldoutLockedError(
        "Holdout split already ran in this spend ledger. Repeat holdout runs are blocked."
      );
    }
  }
}

function assertJudgeAuthorizedForSplit(
  split: EvalSplitName,
  judge: JudgeAdapter,
  calibrationReport: CalibrationReport | undefined
): void {
  if (split === "dev") {
    return;
  }
  if (judge.tier === "frontier") {
    return;
  }
  const hasPassingCalibration =
    calibrationReport !== undefined &&
    normalizeModelIdentifier(calibrationReport.judgeModel) ===
      normalizeModelIdentifier(judge.modelId) &&
    calibrationReport.passed &&
    calibrationReport.contradictionAgreement >= DEFAULT_MIN_CALIBRATION_AGREEMENT;

  if (!hasPassingCalibration) {
    throw new JudgePolicyError(
      `Split "${split}" requires a Frontier Tier Judge or a passing calibration report (>= 90% contradiction agreement) for "${judge.modelId}".`
    );
  }
}

function selectQuestions(
  split: EvalSplitName,
  splitManifest: SplitManifest,
  questions: readonly DevQuestion[],
  onlyIds: readonly string[] | undefined,
  stratumFilter: QuestionStratum | undefined
): DevQuestion[] {
  const byId = new Map<string, DevQuestion>();
  for (const question of questions) {
    byId.set(question.question_id, question);
  }

  const splitIds = splitManifest[split];
  const selected: DevQuestion[] = [];
  const onlySet = onlyIds !== undefined && onlyIds.length > 0 ? new Set(onlyIds) : undefined;

  for (const questionId of splitIds) {
    const question = byId.get(questionId);
    if (question === undefined) {
      throw new ValidationError(`Question "${questionId}" from split "${split}" is missing.`);
    }
    if (onlySet !== undefined && !onlySet.has(questionId)) {
      continue;
    }
    if (stratumFilter !== undefined && classifyQuestionStratum(question) !== stratumFilter) {
      continue;
    }
    selected.push(question);
  }

  if (selected.length === 0) {
    throw new ValidationError(
      `No questions matched split "${split}" with the requested filters.`
    );
  }
  return selected;
}

export async function runEvaluation(
  deps: EvalRunnerDeps,
  options: RunEvaluationOptions
): Promise<EvalRunManifest> {
  assertJudgeDistinctFromResponder(deps.judge.modelId, deps.responder.modelId);
  assertHoldoutPolicy(
    options.split,
    deps.spendLedger,
    options.unlockHoldout ?? false,
    options.allowRepeatHoldout ?? false
  );
  assertJudgeAuthorizedForSplit(options.split, deps.judge, options.calibrationReport);

  const budgetCheck: BudgetCheckOptions = {
    maxTotalUsd: options.budgetUsd ?? DEFAULT_EVAL_BUDGET_USD,
    ...(options.estimatedRunCostUsd !== undefined
      ? { estimatedAdditionalUsd: options.estimatedRunCostUsd }
      : {}),
    ...(options.maxBucketUsd !== undefined ? { maxBucketUsd: options.maxBucketUsd } : {})
  };
  deps.spendLedger.assertWithinBudget(budgetCheck);

  const selectedQuestions = selectQuestions(
    options.split,
    options.splitManifest,
    options.questions,
    options.onlyIds,
    options.stratumFilter
  );

  const now = deps.now ?? (() => new Date());
  const startedAt = now();
  const runId = options.runId ?? buildRunId(options.profile.name, options.split, startedAt);
  const indexConfigHash = computeIndexConfigHash(options.profile);
  const gitSha = deps.gitSha ?? "unknown";

  let responderSpendUsd = 0;
  let judgeSpendUsd = 0;

  const mergeCallOptions = (
    question: DevQuestion,
    itemHeaders?: Readonly<Record<string, string>>
  ): ResponderCallOptions | undefined => {
    const providerHeaders = deps.getItemRequestHeaders?.(question);
    const merged: Record<string, string> = {
      ...options.requestHeaders,
      ...providerHeaders,
      ...itemHeaders
    };
    return Object.keys(merged).length > 0 ? { headers: merged } : undefined;
  };

  const executeQuestion = async (
    question: DevQuestion,
    itemContext?: ExperimentItemExecutionContext
  ): Promise<EvalItemResult> => {
    const stratum = classifyQuestionStratum(question);
    const callOptions = mergeCallOptions(question, itemContext?.headers);
    const response = await deps.responder.answer(question.question, callOptions);
    const judgeResult = await deps.judge.evaluate({
      question: question.question,
      goldAnswer: question.gold_answer,
      answerFacts: question.answer_facts,
      expectedDocIds: question.expected_doc_ids,
      responderAnswer: response.answer,
      responderDocIds: response.documentIds
    });

    const responderUsage = response.usage ?? { inputTokens: 0, outputTokens: 0 };
    const responderCostUsd = estimateTokenCostUsd(
      deps.responder.modelId,
      responderUsage.inputTokens,
      responderUsage.outputTokens
    );
    const judgeCostUsd = estimateTokenCostUsd(
      deps.judge.modelId,
      judgeResult.usage.inputTokens,
      judgeResult.usage.outputTokens
    );

    const entryTimestamp = now().toISOString();
    deps.spendLedger.record({
      timestamp: entryTimestamp,
      runId,
      split: options.split,
      bucket: "responder",
      model: deps.responder.modelId,
      inputTokens: responderUsage.inputTokens,
      outputTokens: responderUsage.outputTokens,
      usd: responderCostUsd
    });
    deps.spendLedger.record({
      timestamp: entryTimestamp,
      runId,
      split: options.split,
      bucket: "judge",
      model: deps.judge.modelId,
      inputTokens: judgeResult.usage.inputTokens,
      outputTokens: judgeResult.usage.outputTokens,
      usd: judgeCostUsd
    });

    responderSpendUsd += responderCostUsd;
    judgeSpendUsd += judgeCostUsd;

    const citationScore = scoreCitations(
      response.documentIds,
      question.expected_doc_ids,
      deps.validCorpusIds
    );
    const effectiveAnswerScore = computeEffectiveAnswerScore(
      judgeResult.verdict.answerScore,
      judgeResult.verdict.hasFabricatedOrContradictingClaim
    );
    const compositeScore = scoreComposite(
      judgeResult.verdict.answerScore,
      citationScore.f1,
      judgeResult.verdict.hasFabricatedOrContradictingClaim
    );
    const abstained =
      stratum === "unanswerable"
        ? judgeResult.verdict.abstained && response.documentIds.length === 0
        : judgeResult.verdict.abstained;

    return {
      questionId: question.question_id,
      stratum,
      question: question.question,
      answer: response.answer,
      returnedDocumentIds: [...response.documentIds],
      expectedDocumentIds: [...question.expected_doc_ids],
      verdict: judgeResult.verdict,
      effectiveAnswerScore,
      citationScore,
      compositeScore,
      abstained,
      latencyMs: response.latencyMs,
      responseBytes: response.responseBytes,
      responderCostUsd,
      judgeCostUsd
    };
  };

  const finalizeRun = async (items: readonly EvalItemResult[]): Promise<EvalRunManifest> => {
    const metrics = aggregateSuiteMetrics(
      items.map((item) => ({
        stratum: item.stratum,
        effectiveAnswerScore: item.effectiveAnswerScore,
        citationScore: item.citationScore.f1,
        compositeScore: item.compositeScore,
        abstained: item.abstained,
        invalidCitations: item.citationScore.invalidCount,
        latencyMs: item.latencyMs
      }))
    );

    const manifest = EvalRunManifestSchema.parse({
      runId,
      timestamp: startedAt.toISOString(),
      gitSha,
      split: options.split,
      splitSeed: options.splitManifest.seed,
      profile: options.profile,
      indexConfigHash,
      responderModel: deps.responder.modelId,
      judgeModel: deps.judge.modelId,
      judgeTier: deps.judge.tier,
      metrics,
      spendByBucketUsd: {
        embedding: 0,
        responder: responderSpendUsd,
        judge: judgeSpendUsd
      },
      totalSpendUsd: responderSpendUsd + judgeSpendUsd,
      items: [...items]
    });

    mkdirSync(deps.runsDir, { recursive: true });
    const manifestPath = join(deps.runsDir, `${runId}.json`);
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx"
    });

    for (const exporter of deps.exporters) {
      await exporter.onRunComplete(manifest);
    }

    return manifest;
  };

  const concurrency = options.concurrency ?? 1;
  if (!Number.isInteger(concurrency) || concurrency <= 0) {
    throw new ValidationError(`concurrency must be a positive integer, got ${concurrency}.`);
  }

  if (deps.experimentWrapper !== undefined) {
    const context: ExperimentRunContext = {
      runId,
      timestamp: startedAt.toISOString(),
      gitSha,
      split: options.split,
      splitSeed: options.splitManifest.seed,
      profile: options.profile,
      indexConfigHash,
      responderModel: deps.responder.modelId,
      judgeModel: deps.judge.modelId,
      judgeTier: deps.judge.tier,
      concurrency,
      questions: selectedQuestions
    };
    return deps.experimentWrapper.runExperiment(context, executeQuestion, finalizeRun);
  }

  const orderedResults = Array.from<EvalItemResult | undefined>({
    length: selectedQuestions.length
  });
  let nextIndex = 0;
  let firstError: unknown = null;

  const runWorker = async (): Promise<void> => {
    for (;;) {
      if (firstError !== null) {
        return;
      }
      const currentIndex = nextIndex;
      if (currentIndex >= selectedQuestions.length) {
        return;
      }
      nextIndex += 1;
      const question = selectedQuestions[currentIndex];
      if (question === undefined) {
        return;
      }
      try {
        orderedResults[currentIndex] = await executeQuestion(question);
      } catch (error) {
        if (firstError === null) {
          firstError = error;
        }
        return;
      }
    }
  };

  const workerCount = Math.min(concurrency, selectedQuestions.length);
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  if (firstError !== null) {
    throw firstError;
  }

  const items: EvalItemResult[] = [];
  for (const item of orderedResults) {
    if (item !== undefined) {
      items.push(item);
    }
  }

  return finalizeRun(items);
}

export async function calibrateJudge(
  candidateJudge: JudgeAdapter,
  frontierJudge: JudgeAdapter,
  items: readonly JudgeInput[],
  deps: CalibrateJudgeDeps
): Promise<CalibrationReport> {
  const split = deps.split ?? "dev";
  if (split === "holdout") {
    throw new HoldoutLockedError(
      "Judge calibration cannot run on the locked holdout split. Use dev or val."
    );
  }
  if (frontierJudge.tier !== "frontier") {
    throw new JudgePolicyError(
      `Calibration reference judge "${frontierJudge.modelId}" must use Frontier Tier.`
    );
  }
  if (
    normalizeModelIdentifier(candidateJudge.modelId) ===
    normalizeModelIdentifier(frontierJudge.modelId)
  ) {
    throw new JudgePolicyError(
      `Candidate judge "${candidateJudge.modelId}" must differ from frontier reference judge.`
    );
  }
  if (items.length === 0) {
    throw new ValidationError("Judge calibration requires at least one item.");
  }

  deps.spendLedger.assertWithinBudget({
    maxTotalUsd: deps.budgetUsd ?? DEFAULT_EVAL_BUDGET_USD
  });

  const now = deps.now ?? (() => new Date());
  const startedAt = now();
  const runId = deps.runId ?? `calibrate_${startedAt.toISOString().replace(/[:.]/g, "-")}`;
  const minAgreement = deps.minAgreement ?? DEFAULT_MIN_CALIBRATION_AGREEMENT;

  let agreementCount = 0;

  for (const item of items) {
    const candidateResult = await candidateJudge.evaluate(item);
    const frontierResult = await frontierJudge.evaluate(item);
    const timestamp = now().toISOString();

    deps.spendLedger.record({
      timestamp,
      runId,
      split,
      bucket: "judge",
      model: candidateJudge.modelId,
      inputTokens: candidateResult.usage.inputTokens,
      outputTokens: candidateResult.usage.outputTokens,
      usd: estimateTokenCostUsd(
        candidateJudge.modelId,
        candidateResult.usage.inputTokens,
        candidateResult.usage.outputTokens
      )
    });
    deps.spendLedger.record({
      timestamp,
      runId,
      split,
      bucket: "judge",
      model: frontierJudge.modelId,
      inputTokens: frontierResult.usage.inputTokens,
      outputTokens: frontierResult.usage.outputTokens,
      usd: estimateTokenCostUsd(
        frontierJudge.modelId,
        frontierResult.usage.inputTokens,
        frontierResult.usage.outputTokens
      )
    });

    if (
      candidateResult.verdict.hasFabricatedOrContradictingClaim ===
      frontierResult.verdict.hasFabricatedOrContradictingClaim
    ) {
      agreementCount += 1;
    }
  }

  const contradictionAgreement = agreementCount / items.length;
  const report = CalibrationReportSchema.parse({
    judgeModel: candidateJudge.modelId,
    frontierJudgeModel: frontierJudge.modelId,
    sampleSize: items.length,
    contradictionAgreement,
    passed: contradictionAgreement >= minAgreement,
    timestamp: startedAt.toISOString()
  });

  if (deps.outputPath !== undefined) {
    mkdirSync(dirname(deps.outputPath), { recursive: true });
    writeFileSync(deps.outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }

  return report;
}

function assertManifestSplit(
  manifest: EvalRunManifest,
  expectedSplit: "dev" | "val",
  label: string
): void {
  if (manifest.split !== expectedSplit) {
    throw new ValidationError(
      `Promotion gate requires ${label} manifest split "${expectedSplit}", got "${manifest.split}".`
    );
  }
}

export function evaluatePromotionGate(
  candidateDev: EvalRunManifest,
  championDev: EvalRunManifest,
  candidateVal: EvalRunManifest,
  championVal: EvalRunManifest,
  options?: PromotionGateOptions
): PromotionDecision {
  assertManifestSplit(candidateDev, "dev", "candidateDev");
  assertManifestSplit(championDev, "dev", "championDev");
  assertManifestSplit(candidateVal, "val", "candidateVal");
  assertManifestSplit(championVal, "val", "championVal");

  const noiseBand = options?.noiseBand ?? DEFAULT_PROMOTION_NOISE_BAND;
  const maxLatencyP95Ms = options?.maxLatencyP95Ms ?? DEFAULT_MAX_LATENCY_P95_MS;

  const devDelta = candidateDev.metrics.compositeScore - championDev.metrics.compositeScore;
  const valDelta = candidateVal.metrics.compositeScore - championVal.metrics.compositeScore;
  const reasons: string[] = [];

  if (devDelta < noiseBand) {
    reasons.push(
      `dev composite delta ${devDelta.toFixed(4)} is below noise band ${noiseBand.toFixed(4)}`
    );
  }
  if (valDelta < 0) {
    reasons.push(`val composite delta ${valDelta.toFixed(4)} regressed below 0`);
  }
  if (
    candidateDev.metrics.invalidCitationsTotal > 0 ||
    candidateVal.metrics.invalidCitationsTotal > 0
  ) {
    reasons.push(
      `invalid citations must be 0 (dev=${candidateDev.metrics.invalidCitationsTotal}, val=${candidateVal.metrics.invalidCitationsTotal})`
    );
  }
  if (candidateDev.metrics.abstentionRecall < championDev.metrics.abstentionRecall) {
    reasons.push(
      `dev abstention recall regressed (${candidateDev.metrics.abstentionRecall.toFixed(4)} < ${championDev.metrics.abstentionRecall.toFixed(4)})`
    );
  }
  if (candidateVal.metrics.abstentionRecall < championVal.metrics.abstentionRecall) {
    reasons.push(
      `val abstention recall regressed (${candidateVal.metrics.abstentionRecall.toFixed(4)} < ${championVal.metrics.abstentionRecall.toFixed(4)})`
    );
  }
  if (
    candidateDev.metrics.latencyP95Ms > maxLatencyP95Ms ||
    candidateVal.metrics.latencyP95Ms > maxLatencyP95Ms
  ) {
    reasons.push(
      `latency p95 exceeded ${maxLatencyP95Ms} ms (dev=${Math.round(candidateDev.metrics.latencyP95Ms)}, val=${Math.round(candidateVal.metrics.latencyP95Ms)})`
    );
  }

  return PromotionDecisionSchema.parse({
    promoted: reasons.length === 0,
    reasons,
    devDelta,
    valDelta
  });
}
