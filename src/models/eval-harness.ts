import { z } from "zod";
import { AppError } from "./errors.js";
import { QuestionStratumSchema } from "./eval.js";
import { RunProfileSchema } from "./profile.js";

export class BudgetExceededError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("BUDGET_EXCEEDED", message, options);
  }
}

export class HoldoutLockedError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("HOLDOUT_LOCKED", message, options);
  }
}

export class JudgePolicyError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("JUDGE_POLICY_ERROR", message, options);
  }
}

export const EvalSplitNameSchema = z.enum(["dev", "val", "holdout"]);
export type EvalSplitName = z.infer<typeof EvalSplitNameSchema>;

export const SpendBucketSchema = z.enum(["embedding", "responder", "judge"]);
export type SpendBucket = z.infer<typeof SpendBucketSchema>;

export const JudgeTierSchema = z.enum(["fast", "frontier", "decision"]);
export type JudgeTier = z.infer<typeof JudgeTierSchema>;

export const AskResponseSchema = z.object({
  answer: z.string().min(1),
  document_ids: z.array(z.string())
});
export type AskResponse = z.infer<typeof AskResponseSchema>;

export const JudgeVerdictSchema = z.object({
  answerScore: z.number().min(0).max(1),
  hasFabricatedOrContradictingClaim: z.boolean(),
  abstained: z.boolean(),
  reasoning: z.string().min(1)
});
export type JudgeVerdict = z.infer<typeof JudgeVerdictSchema>;

export const CitationScoreSchema = z.object({
  precision: z.number().min(0).max(1),
  recall: z.number().min(0).max(1),
  f1: z.number().min(0).max(1),
  invalidCount: z.number().int().nonnegative()
});
export type CitationScore = z.infer<typeof CitationScoreSchema>;

export const EvalItemResultSchema = z.object({
  questionId: z.string().min(1),
  stratum: QuestionStratumSchema,
  question: z.string().min(1),
  answer: z.string(),
  returnedDocumentIds: z.array(z.string()),
  expectedDocumentIds: z.array(z.string()),
  verdict: JudgeVerdictSchema,
  effectiveAnswerScore: z.number().min(0).max(1),
  citationScore: CitationScoreSchema,
  compositeScore: z.number().min(0).max(1),
  abstained: z.boolean(),
  latencyMs: z.number().nonnegative(),
  responseBytes: z.number().int().nonnegative(),
  responderCostUsd: z.number().nonnegative(),
  judgeCostUsd: z.number().nonnegative()
});
export type EvalItemResult = z.infer<typeof EvalItemResultSchema>;

export const StratumMetricsSchema = z.object({
  count: z.number().int().nonnegative(),
  compositeScore: z.number().min(0).max(1),
  answerScore: z.number().min(0).max(1),
  citationScore: z.number().min(0).max(1)
});
export type StratumMetrics = z.infer<typeof StratumMetricsSchema>;

export const SuiteMetricsSchema = z.object({
  totalItems: z.number().int().nonnegative(),
  compositeScore: z.number().min(0).max(1),
  answerScore: z.number().min(0).max(1),
  citationScore: z.number().min(0).max(1),
  abstentionRecall: z.number().min(0).max(1),
  falseAbstentionRate: z.number().min(0).max(1),
  invalidCitationsTotal: z.number().int().nonnegative(),
  latencyP50Ms: z.number().nonnegative(),
  latencyP95Ms: z.number().nonnegative(),
  byStratum: z.object({
    unanswerable: StratumMetricsSchema,
    single_doc: StratumMetricsSchema,
    two_doc: StratumMetricsSchema,
    multi_doc: StratumMetricsSchema
  })
});
export type SuiteMetrics = z.infer<typeof SuiteMetricsSchema>;

export const SpendByBucketSchema = z.object({
  embedding: z.number().nonnegative(),
  responder: z.number().nonnegative(),
  judge: z.number().nonnegative()
});
export type SpendByBucket = z.infer<typeof SpendByBucketSchema>;

export const EvalRunManifestSchema = z.object({
  runId: z.string().min(1),
  timestamp: z.string().min(1),
  gitSha: z.string().min(1),
  split: EvalSplitNameSchema,
  splitSeed: z.number().int(),
  profile: RunProfileSchema,
  indexConfigHash: z.string().min(1),
  responderModel: z.string().min(1),
  judgeModel: z.string().min(1),
  judgeTier: JudgeTierSchema,
  metrics: SuiteMetricsSchema,
  spendByBucketUsd: SpendByBucketSchema,
  totalSpendUsd: z.number().nonnegative(),
  items: z.array(EvalItemResultSchema)
});
export type EvalRunManifest = z.infer<typeof EvalRunManifestSchema>;

export const SpendLedgerEntrySchema = z.object({
  timestamp: z.string().min(1),
  runId: z.string().min(1),
  split: EvalSplitNameSchema,
  bucket: SpendBucketSchema,
  model: z.string().min(1),
  inputTokens: z.number().int().nonnegative(),
  outputTokens: z.number().int().nonnegative(),
  usd: z.number().nonnegative()
});
export type SpendLedgerEntry = z.infer<typeof SpendLedgerEntrySchema>;

export const CalibrationReportSchema = z.object({
  judgeModel: z.string().min(1),
  frontierJudgeModel: z.string().min(1),
  sampleSize: z.number().int().positive(),
  contradictionAgreement: z.number().min(0).max(1),
  passed: z.boolean(),
  timestamp: z.string().min(1)
});
export type CalibrationReport = z.infer<typeof CalibrationReportSchema>;

export const PromotionDecisionSchema = z.object({
  promoted: z.boolean(),
  reasons: z.array(z.string()),
  devDelta: z.number(),
  valDelta: z.number()
});
export type PromotionDecision = z.infer<typeof PromotionDecisionSchema>;
