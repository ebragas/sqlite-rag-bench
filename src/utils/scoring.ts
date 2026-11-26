export type ScoringStratum = "unanswerable" | "single_doc" | "two_doc" | "multi_doc";

export interface CitationScoreResult {
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly invalidCount: number;
}

export interface ScorableEvalItem {
  readonly stratum: ScoringStratum;
  readonly effectiveAnswerScore: number;
  readonly citationScore: number;
  readonly compositeScore: number;
  readonly abstained: boolean;
  readonly invalidCitations: number;
  readonly latencyMs: number;
}

export interface StratumAggregate {
  readonly count: number;
  readonly compositeScore: number;
  readonly answerScore: number;
  readonly citationScore: number;
}

export interface SuiteAggregate {
  readonly totalItems: number;
  readonly compositeScore: number;
  readonly answerScore: number;
  readonly citationScore: number;
  readonly abstentionRecall: number;
  readonly falseAbstentionRate: number;
  readonly invalidCitationsTotal: number;
  readonly latencyP50Ms: number;
  readonly latencyP95Ms: number;
  readonly byStratum: {
    readonly unanswerable: StratumAggregate;
    readonly single_doc: StratumAggregate;
    readonly two_doc: StratumAggregate;
    readonly multi_doc: StratumAggregate;
  };
}

interface ModelPricingPerMillionUsd {
  readonly input: number;
  readonly output: number;
}

const MODEL_PRICING_PER_MILLION_USD: Readonly<Record<string, ModelPricingPerMillionUsd>> = {
  "gemini-4-neon": { input: 3.0, output: 15.0 },
  "gemini-4-carbon": { input: 2.0, output: 10.0 },
  "gemini-4-argon": { input: 2.0, output: 10.0 },
  "gemini-4-hydrogen": { input: 1.5, output: 7.5 },
  "gemini-3.8-flash": { input: 0.75, output: 3.75 },
  "gemini-3.7-flash": { input: 0.75, output: 3.75 },
  "gemini-3.6-flash": { input: 0.5, output: 3.0 },
  "gemini-3.5-flash": { input: 0.5, output: 3.0 },
  "gemini-3.5-flash-lite": { input: 0.3, output: 2.5 },
  "gemini-3.1-pro-preview": { input: 2.0, output: 12.0 },
  "gemini-3.1-pro-preview-customtools": { input: 2.0, output: 12.0 },
  "gemini-3.1-flash-lite": { input: 0.25, output: 1.5 },
  "gemini-3-flash-preview": { input: 0.5, output: 3.0 },
  "gemini-2.5-flash-lite": { input: 0.1, output: 0.4 },
  "gemini-2.5-flash": { input: 0.3, output: 2.5 },
  "gemini-2.5-pro": { input: 1.25, output: 10.0 },
  "gemini-embedding-2": { input: 0.2, output: 0.0 },
  "gemini-embedding-2-preview": { input: 0.2, output: 0.0 },
  "gemini-embedding-001": { input: 0.15, output: 0.0 },
  "gpt-4.1": { input: 2.0, output: 8.0 },
  "gpt-4.1-mini": { input: 0.4, output: 1.6 },
  "gpt-4.1-nano": { input: 0.1, output: 0.4 },
  "gpt-4o": { input: 2.5, output: 10.0 },
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  o3: { input: 2.0, output: 8.0 },
  "o4-mini": { input: 1.1, output: 4.4 },
  "clef-flash": { input: 0.0, output: 0.0 },
  "clef:27b-mxfp8": { input: 0.0, output: 0.0 }
};

const ZERO_COST_PREFIXES = ["ollama:", "ollama/", "local:", "local/", "mock:", "fake:"] as const;
const DEFAULT_FALLBACK_PRICING: ModelPricingPerMillionUsd = { input: 1.0, output: 4.0 };
const TOKENS_PER_MILLION = 1_000_000;

export function clampUnit(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

export function normalizeModelIdentifier(model: string): string {
  const trimmed = model.trim().toLowerCase();
  if (trimmed.startsWith("google/")) {
    return trimmed.slice("google/".length);
  }
  if (trimmed.startsWith("openai/")) {
    return trimmed.slice("openai/".length);
  }
  return trimmed;
}

export function scoreCitations(
  returnedIds: readonly string[],
  expectedIds: readonly string[],
  validCorpusIds?: ReadonlySet<string>
): CitationScoreResult {
  const uniqueReturned: string[] = [];
  const seenReturned = new Set<string>();
  for (const rawId of returnedIds) {
    if (!seenReturned.has(rawId)) {
      seenReturned.add(rawId);
      uniqueReturned.push(rawId);
    }
  }

  let invalidCount = 0;
  const validReturned: string[] = [];
  for (const id of uniqueReturned) {
    const isMalformed = id.trim().length === 0;
    const isMissingFromCorpus = validCorpusIds !== undefined && !validCorpusIds.has(id);
    if (isMalformed || isMissingFromCorpus) {
      invalidCount += 1;
    } else {
      validReturned.push(id);
    }
  }

  const expectedSet = new Set(expectedIds);
  if (expectedSet.size === 0) {
    const exactEmpty = uniqueReturned.length === 0;
    return {
      precision: exactEmpty ? 1 : 0,
      recall: exactEmpty ? 1 : 0,
      f1: exactEmpty ? 1 : 0,
      invalidCount
    };
  }

  let matched = 0;
  for (const id of validReturned) {
    if (expectedSet.has(id)) {
      matched += 1;
    }
  }

  const precision = uniqueReturned.length === 0 ? 0 : matched / uniqueReturned.length;
  const recall = matched / expectedSet.size;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return { precision, recall, f1, invalidCount };
}

export function computeEffectiveAnswerScore(
  answerScore: number,
  hasFabricatedOrContradictingClaim: boolean
): number {
  if (hasFabricatedOrContradictingClaim) {
    return 0;
  }
  return clampUnit(answerScore);
}

export function scoreComposite(
  answerScore: number,
  citationScore: number,
  hasFabricatedOrContradictingClaim: boolean
): number {
  const effectiveAnswer = computeEffectiveAnswerScore(
    answerScore,
    hasFabricatedOrContradictingClaim
  );
  const clampedCitation = clampUnit(citationScore);
  return 0.7 * effectiveAnswer + 0.3 * clampedCitation;
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total / values.length;
}

function nearestRankPercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil((percentile / 100) * sorted.length) - 1;
  const index = Math.min(sorted.length - 1, Math.max(0, rank));
  return sorted[index] ?? 0;
}

function aggregateStratum(items: readonly ScorableEvalItem[]): StratumAggregate {
  if (items.length === 0) {
    return { count: 0, compositeScore: 0, answerScore: 0, citationScore: 0 };
  }
  return {
    count: items.length,
    compositeScore: mean(items.map((item) => item.compositeScore)),
    answerScore: mean(items.map((item) => item.effectiveAnswerScore)),
    citationScore: mean(items.map((item) => item.citationScore))
  };
}

export function aggregateSuiteMetrics(items: readonly ScorableEvalItem[]): SuiteAggregate {
  const unanswerableItems: ScorableEvalItem[] = [];
  const singleDocItems: ScorableEvalItem[] = [];
  const twoDocItems: ScorableEvalItem[] = [];
  const multiDocItems: ScorableEvalItem[] = [];
  const answerableItems: ScorableEvalItem[] = [];

  let invalidCitationsTotal = 0;
  const latencies: number[] = [];

  for (const item of items) {
    invalidCitationsTotal += item.invalidCitations;
    latencies.push(item.latencyMs);

    switch (item.stratum) {
      case "unanswerable":
        unanswerableItems.push(item);
        break;
      case "single_doc":
        singleDocItems.push(item);
        answerableItems.push(item);
        break;
      case "two_doc":
        twoDocItems.push(item);
        answerableItems.push(item);
        break;
      case "multi_doc":
        multiDocItems.push(item);
        answerableItems.push(item);
        break;
    }
  }

  const abstentionRecall =
    unanswerableItems.length === 0
      ? 0
      : unanswerableItems.filter((item) => item.abstained).length / unanswerableItems.length;

  const falseAbstentionRate =
    answerableItems.length === 0
      ? 0
      : answerableItems.filter((item) => item.abstained).length / answerableItems.length;

  return {
    totalItems: items.length,
    compositeScore: mean(items.map((item) => item.compositeScore)),
    answerScore: mean(items.map((item) => item.effectiveAnswerScore)),
    citationScore: mean(items.map((item) => item.citationScore)),
    abstentionRecall,
    falseAbstentionRate,
    invalidCitationsTotal,
    latencyP50Ms: nearestRankPercentile(latencies, 50),
    latencyP95Ms: nearestRankPercentile(latencies, 95),
    byStratum: {
      unanswerable: aggregateStratum(unanswerableItems),
      single_doc: aggregateStratum(singleDocItems),
      two_doc: aggregateStratum(twoDocItems),
      multi_doc: aggregateStratum(multiDocItems)
    }
  };
}

export function estimateTokenCostUsd(
  model: string,
  inputTokens: number,
  outputTokens: number
): number {
  const safeInput = Math.max(0, Number.isFinite(inputTokens) ? inputTokens : 0);
  const safeOutput = Math.max(0, Number.isFinite(outputTokens) ? outputTokens : 0);
  const normalized = normalizeModelIdentifier(model);

  for (const prefix of ZERO_COST_PREFIXES) {
    if (normalized.startsWith(prefix)) {
      return 0;
    }
  }

  const pricing = MODEL_PRICING_PER_MILLION_USD[normalized] ?? DEFAULT_FALLBACK_PRICING;
  return (
    (safeInput / TOKENS_PER_MILLION) * pricing.input +
    (safeOutput / TOKENS_PER_MILLION) * pricing.output
  );
}
