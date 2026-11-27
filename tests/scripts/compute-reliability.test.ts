import { describe, expect, it } from "vitest";
import {
  aggregateReliabilityTrials,
  computePassAtK,
  computePassPowK,
  type ReliabilityItemRecord
} from "../../scripts/compute-reliability.js";

describe("compute-reliability unbiased pass@k and pass^k estimators", () => {
  it("computes exact combinatorial pass@k and pass^k values", () => {
    expect(computePassAtK(10, 0, 1)).toBe(0);
    expect(computePassAtK(10, 0, 10)).toBe(0);
    expect(computePassAtK(10, 10, 1)).toBe(1);
    expect(computePassAtK(10, 10, 10)).toBe(1);
    expect(computePassPowK(10, 0, 1)).toBe(0);
    expect(computePassPowK(10, 10, 10)).toBe(1);

    // n=10, c=8, k=3:
    // pass@3 = 1 - C(2,3)/C(10,3) = 1 (impossible to draw 3 failures when only 2 exist)
    expect(computePassAtK(10, 8, 3)).toBe(1);
    // pass^3 = C(8,3)/C(10,3) = 56/120 = 0.466666...
    expect(computePassPowK(10, 8, 3)).toBeCloseTo(56 / 120, 6);

    // n=10, c=5, k=2:
    // pass@2 = 1 - (5*4)/(10*9) = 1 - 20/90 = 7/9
    expect(computePassAtK(10, 5, 2)).toBeCloseTo(7 / 9, 6);
    // pass^2 = (5*4)/(10*9) = 2/9
    expect(computePassPowK(10, 5, 2)).toBeCloseTo(2 / 9, 6);
  });

  it("aggregates strict, perfect, and safety pass bars across trials", () => {
    const makeItem = (
      id: string,
      split: "val" | "holdout",
      score: number,
      fab = false
    ): ReliabilityItemRecord => ({
      questionId: id,
      split,
      stratum: "multi_doc",
      question: `Question ${id}`,
      compositeScore: score,
      effectiveAnswerScore: fab ? 0 : score,
      citationF1: score,
      invalidCitations: 0,
      fabricatedOrContradicted: fab,
      abstained: false,
      latencyMs: 1000
    });

    const trials = [
      [makeItem("q1", "val", 1.0), makeItem("q2", "holdout", 0.85)],
      [makeItem("q1", "val", 1.0), makeItem("q2", "holdout", 0.5, true)]
    ];

    const report = aggregateReliabilityTrials(trials, {
      gitSha: "ca628894",
      profileName: "prod",
      responderModel: "gemini-3.8-flash",
      judgeModel: "gemini-4-neon",
      spendUsd: { responder: 1.5, judge: 0.5, total: 2.0 }
    });

    expect(report.trialsPerQuestion).toBe(2);
    expect(report.totalEvaluations).toBe(4);
    expect(report.totalFabrications).toBe(1);
    // q1 passes strict 2/2, q2 passes strict 1/2 -> mean pass@1 = 0.75, pass@2 = 1.0, pass^2 = 0.5
    expect(report.overall.strict.passAtK).toEqual([0.75, 1]);
    expect(report.overall.strict.passPowK).toEqual([0.75, 0.5]);
    // q1 passes perfect 2/2, q2 passes perfect 0/2 -> mean pass@1 = 0.5, pass@2 = 0.5, pass^2 = 0.5
    expect(report.overall.perfect.passAtK).toEqual([0.5, 0.5]);
    expect(report.overall.perfect.passPowK).toEqual([0.5, 0.5]);
  });
});
