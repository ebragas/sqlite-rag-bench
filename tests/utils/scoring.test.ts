import { describe, expect, it } from "vitest";
import {
  aggregateSuiteMetrics,
  computeEffectiveAnswerScore,
  estimateTokenCostUsd,
  normalizeModelIdentifier,
  scoreCitations,
  scoreComposite,
  type ScorableEvalItem
} from "../../src/utils/scoring.js";

describe("scoreCitations", () => {
  it("computes exact precision, recall, and F1 with deduplication", () => {
    const validCorpus = new Set(["dsid_1", "dsid_2", "dsid_3"]);
    const result = scoreCitations(
      ["dsid_1", "dsid_1", "dsid_3"],
      ["dsid_1", "dsid_2"],
      validCorpus
    );

    expect(result.precision).toBeCloseTo(0.5);
    expect(result.recall).toBeCloseTo(0.5);
    expect(result.f1).toBeCloseTo(0.5);
    expect(result.invalidCount).toBe(0);
  });

  it("scores unanswerable questions as 1 only when returned document_ids is empty", () => {
    const validCorpus = new Set(["dsid_1"]);

    const empty = scoreCitations([], [], validCorpus);
    expect(empty).toEqual({ precision: 1, recall: 1, f1: 1, invalidCount: 0 });

    const nonEmpty = scoreCitations(["dsid_1"], [], validCorpus);
    expect(nonEmpty).toEqual({ precision: 0, recall: 0, f1: 0, invalidCount: 0 });

    const invalidOnUnanswerable = scoreCitations(["dsid_bad"], [], validCorpus);
    expect(invalidOnUnanswerable).toEqual({ precision: 0, recall: 0, f1: 0, invalidCount: 1 });
  });

  it("counts invalid citations and prevents them from matching expected IDs", () => {
    const validCorpus = new Set(["dsid_valid"]);
    const result = scoreCitations(
      ["dsid_valid", "dsid_missing", "   "],
      ["dsid_valid", "dsid_missing"],
      validCorpus
    );

    expect(result.invalidCount).toBe(2);
    expect(result.precision).toBeCloseTo(1 / 3);
    expect(result.recall).toBeCloseTo(0.5);
    expect(result.f1).toBeCloseTo(0.4);
  });

  it("returns zero scores when answerable question cites nothing", () => {
    const result = scoreCitations([], ["dsid_1"]);
    expect(result).toEqual({ precision: 0, recall: 0, f1: 0, invalidCount: 0 });
  });
});

describe("scoreComposite and Onyx zeroing", () => {
  it("combines 70% answer score and 30% citation score when truthful", () => {
    expect(scoreComposite(0.8, 0.5, false)).toBeCloseTo(0.7 * 0.8 + 0.3 * 0.5);
    expect(computeEffectiveAnswerScore(1.5, false)).toBe(1);
    expect(computeEffectiveAnswerScore(-0.2, false)).toBe(0);
    expect(computeEffectiveAnswerScore(Number.NaN, false)).toBe(0);
  });

  it("zeroes answer score when any fabricated or contradicting claim is present", () => {
    expect(computeEffectiveAnswerScore(0.95, true)).toBe(0);
    expect(scoreComposite(0.95, 1.0, true)).toBeCloseTo(0.3);
    expect(scoreComposite(0.95, 0.0, true)).toBe(0);
  });
});

describe("aggregateSuiteMetrics", () => {
  it("aggregates composite, per-stratum averages, abstention metrics, and latency percentiles", () => {
    const items: ScorableEvalItem[] = [
      {
        stratum: "unanswerable",
        effectiveAnswerScore: 1,
        citationScore: 1,
        compositeScore: 1,
        abstained: true,
        invalidCitations: 0,
        latencyMs: 100
      },
      {
        stratum: "unanswerable",
        effectiveAnswerScore: 0,
        citationScore: 0,
        compositeScore: 0,
        abstained: false,
        invalidCitations: 1,
        latencyMs: 200
      },
      {
        stratum: "single_doc",
        effectiveAnswerScore: 1,
        citationScore: 1,
        compositeScore: 1,
        abstained: false,
        invalidCitations: 0,
        latencyMs: 300
      },
      {
        stratum: "two_doc",
        effectiveAnswerScore: 0.5,
        citationScore: 0.5,
        compositeScore: 0.5,
        abstained: true,
        invalidCitations: 2,
        latencyMs: 400
      },
      {
        stratum: "multi_doc",
        effectiveAnswerScore: 0.8,
        citationScore: 0.6,
        compositeScore: 0.74,
        abstained: false,
        invalidCitations: 0,
        latencyMs: 1000
      }
    ];

    const summary = aggregateSuiteMetrics(items);

    expect(summary.totalItems).toBe(5);
    expect(summary.compositeScore).toBeCloseTo((1 + 0 + 1 + 0.5 + 0.74) / 5);
    expect(summary.answerScore).toBeCloseTo((1 + 0 + 1 + 0.5 + 0.8) / 5);
    expect(summary.citationScore).toBeCloseTo((1 + 0 + 1 + 0.5 + 0.6) / 5);
    expect(summary.abstentionRecall).toBeCloseTo(0.5);
    expect(summary.falseAbstentionRate).toBeCloseTo(1 / 3);
    expect(summary.invalidCitationsTotal).toBe(3);
    expect(summary.latencyP50Ms).toBe(300);
    expect(summary.latencyP95Ms).toBe(1000);

    expect(summary.byStratum.unanswerable.count).toBe(2);
    expect(summary.byStratum.unanswerable.compositeScore).toBeCloseTo(0.5);
    expect(summary.byStratum.single_doc.count).toBe(1);
    expect(summary.byStratum.single_doc.compositeScore).toBeCloseTo(1);
    expect(summary.byStratum.two_doc.count).toBe(1);
    expect(summary.byStratum.two_doc.compositeScore).toBeCloseTo(0.5);
    expect(summary.byStratum.multi_doc.count).toBe(1);
    expect(summary.byStratum.multi_doc.compositeScore).toBeCloseTo(0.74);
  });

  it("handles empty item arrays cleanly", () => {
    const summary = aggregateSuiteMetrics([]);
    expect(summary.totalItems).toBe(0);
    expect(summary.compositeScore).toBe(0);
    expect(summary.abstentionRecall).toBe(0);
    expect(summary.falseAbstentionRate).toBe(0);
    expect(summary.latencyP50Ms).toBe(0);
    expect(summary.latencyP95Ms).toBe(0);
    expect(summary.byStratum.single_doc.count).toBe(0);
  });
});

describe("estimateTokenCostUsd", () => {
  it("computes explicit USD estimates for Gemini, OpenAI, and zero-cost local models", () => {
    expect(estimateTokenCostUsd("gemini-3.5-flash-lite", 1_000_000, 500_000)).toBeCloseTo(1.55);
    expect(estimateTokenCostUsd("google/gemini-3.8-flash", 1_000_000, 1_000_000)).toBeCloseTo(4.5);
    expect(estimateTokenCostUsd("gemini-4-hydrogen", 1_000_000, 1_000_000)).toBeCloseTo(9.0);
    expect(estimateTokenCostUsd("gemini-4-neon", 1_000_000, 1_000_000)).toBeCloseTo(18.0);
    expect(estimateTokenCostUsd("gemini-4-carbon", 1_000_000, 1_000_000)).toBeCloseTo(12.0);
    expect(estimateTokenCostUsd("gemini-2.5-pro", 1_000_000, 1_000_000)).toBeCloseTo(11.25);
    expect(estimateTokenCostUsd("gemini-embedding-001", 2_000_000, 0)).toBeCloseTo(0.3);
    expect(estimateTokenCostUsd("gemini-embedding-2", 2_000_000, 0)).toBeCloseTo(0.4);
    expect(estimateTokenCostUsd("openai/gpt-4.1", 1_000_000, 1_000_000)).toBeCloseTo(10.0);
    expect(estimateTokenCostUsd("ollama:clef-flash", 500_000, 200_000)).toBe(0);
    expect(estimateTokenCostUsd("clef:27b-mxfp8", 500_000, 200_000)).toBe(0);
    expect(estimateTokenCostUsd("unknown-cloud-model", 1_000_000, 0)).toBeCloseTo(1.0);
    expect(estimateTokenCostUsd("gemini-2.5-flash", -100, Number.NaN)).toBe(0);
    expect(normalizeModelIdentifier("  Google/Gemini-2.5-Pro ")).toBe("gemini-2.5-pro");
  });
});
