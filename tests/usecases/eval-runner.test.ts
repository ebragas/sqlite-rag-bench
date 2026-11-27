import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { JudgeAdapter, JudgeEvaluation, JudgeInput } from "../../src/adapters/judge.js";
import type {
  ResponderAdapter,
  ResponderCallOptions,
  ResponderResult
} from "../../src/adapters/responder.js";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import {
  formatTerminalScorecard,
  TerminalScorecardExporter
} from "../../src/adapters/terminal-scorecard.js";
import { ValidationError } from "../../src/models/errors.js";
import type { DevQuestion, SplitManifest } from "../../src/models/eval.js";
import {
  BudgetExceededError,
  CalibrationReportSchema,
  EvalRunManifestSchema,
  HoldoutLockedError,
  JudgePolicyError,
  type CalibrationReport,
  type EvalRunManifest,
  type JudgeTier,
  type JudgeVerdict
} from "../../src/models/eval-harness.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import {
  calibrateJudge,
  evaluatePromotionGate,
  runEvaluation
} from "../../src/usecases/eval-runner.js";

const tempDirs: string[] = [];

function makeTempWorkspace(): { root: string; ledger: SpendLedgerStore; runsDir: string } {
  const root = mkdtempSync(join(tmpdir(), "srb-eval-runner-"));
  tempDirs.push(root);
  return {
    root,
    ledger: new SpendLedgerStore(join(root, "spend-ledger.jsonl")),
    runsDir: join(root, "runs")
  };
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

class DeterministicResponder implements ResponderAdapter {
  public readonly modelId: string;
  private readonly responses: ReadonlyMap<string, ResponderResult>;

  public constructor(modelId: string, responses: ReadonlyMap<string, ResponderResult>) {
    this.modelId = modelId;
    this.responses = responses;
  }

  public answer(question: string, _options?: ResponderCallOptions): Promise<ResponderResult> {
    const hit = this.responses.get(question);
    if (hit !== undefined) {
      return Promise.resolve(hit);
    }
    return Promise.resolve({
      answer: "Default answer",
      documentIds: [],
      latencyMs: 50,
      responseBytes: 40,
      usage: { inputTokens: 1000, outputTokens: 100 }
    });
  }
}

class DeterministicJudge implements JudgeAdapter {
  public readonly modelId: string;
  public readonly tier: JudgeTier;
  public callCount = 0;
  private readonly verdicts: ReadonlyMap<string, JudgeVerdict>;

  public constructor(
    modelId: string,
    tier: JudgeTier,
    verdicts: ReadonlyMap<string, JudgeVerdict>
  ) {
    this.modelId = modelId;
    this.tier = tier;
    this.verdicts = verdicts;
  }

  public evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    this.callCount += 1;
    const verdict = this.verdicts.get(input.question) ?? {
      answerScore: 1,
      hasFabricatedOrContradictingClaim: false,
      abstained: input.expectedDocIds.length === 0,
      reasoning: "Matched all expected facts."
    };
    return Promise.resolve({
      verdict,
      usage: { inputTokens: 2000, outputTokens: 200 }
    });
  }
}

const TEST_PROFILE = RunProfileSchema.parse({ name: "l0-fts" });

const TEST_QUESTIONS: readonly DevQuestion[] = [
  {
    question_id: "q_dev_1",
    question: "Which vendor did Magnetar choose?",
    gold_answer: "Magnetar chose AetherAI.",
    answer_facts: ["Magnetar chose AetherAI"],
    expected_doc_ids: ["dsid_a"],
    source_types: ["fireflies"]
  },
  {
    question_id: "q_dev_2",
    question: "What is the secret Mars launch date?",
    gold_answer: "The corpus does not contain this information.",
    answer_facts: [],
    expected_doc_ids: [],
    source_types: []
  },
  {
    question_id: "q_val_1",
    question: "Which two services handle metering?",
    gold_answer: "Metering uses Pulse and Ledger.",
    answer_facts: ["Pulse", "Ledger"],
    expected_doc_ids: ["dsid_b", "dsid_c"],
    source_types: ["github"]
  },
  {
    question_id: "q_holdout_1",
    question: "Who approved the Q4 rollout?",
    gold_answer: "Naomi approved the Q4 rollout.",
    answer_facts: ["Naomi approved"],
    expected_doc_ids: ["dsid_d"],
    source_types: ["gmail"]
  }
];

const TEST_SPLIT_MANIFEST: SplitManifest = {
  version: 1,
  seed: 20261125,
  holdout_seed: 20261125,
  total_questions: 4,
  counts: {
    dev: { unanswerable: 1, single_doc: 1, two_doc: 0, multi_doc: 0 },
    val: { unanswerable: 0, single_doc: 0, two_doc: 1, multi_doc: 0 },
    holdout: { unanswerable: 0, single_doc: 1, two_doc: 0, multi_doc: 0 }
  },
  dev: ["q_dev_1", "q_dev_2"],
  val: ["q_val_1"],
  holdout: ["q_holdout_1"]
};

const VALID_CORPUS_IDS = new Set(["dsid_a", "dsid_b", "dsid_c", "dsid_d"]);

function makeManifest(overrides: {
  compositeScore: number;
  abstentionRecall?: number;
  invalidCitationsTotal?: number;
  latencyP95Ms?: number;
  split?: "dev" | "val" | "holdout";
}): EvalRunManifest {
  return EvalRunManifestSchema.parse({
    runId: `run-${overrides.split ?? "dev"}`,
    timestamp: "2026-11-26T00:00:00.000Z",
    gitSha: "abc1234",
    split: overrides.split ?? "dev",
    splitSeed: 20261125,
    profile: TEST_PROFILE,
    indexConfigHash: "hash1234567890",
    responderModel: "gemini-3.8-flash",
    judgeModel: "gemini-2.5-pro",
    judgeTier: "frontier",
    metrics: {
      totalItems: 10,
      compositeScore: overrides.compositeScore,
      answerScore: overrides.compositeScore,
      citationScore: overrides.compositeScore,
      abstentionRecall: overrides.abstentionRecall ?? 1,
      falseAbstentionRate: 0,
      invalidCitationsTotal: overrides.invalidCitationsTotal ?? 0,
      latencyP50Ms: 500,
      latencyP95Ms: overrides.latencyP95Ms ?? 1200,
      byStratum: {
        unanswerable: { count: 2, compositeScore: 1, answerScore: 1, citationScore: 1 },
        single_doc: { count: 4, compositeScore: 0.8, answerScore: 0.8, citationScore: 0.8 },
        two_doc: { count: 2, compositeScore: 0.7, answerScore: 0.7, citationScore: 0.7 },
        multi_doc: { count: 2, compositeScore: 0.6, answerScore: 0.6, citationScore: 0.6 }
      }
    },
    spendByBucketUsd: { embedding: 0, responder: 0.1, judge: 0.2 },
    totalSpendUsd: 0.3,
    items: []
  });
}

describe("runEvaluation policy invariants and manifest output", () => {
  it("blocks runs when Judge model matches Responder model", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    const responder = new DeterministicResponder("gemini-3.8-flash", new Map());
    const judge = new DeterministicJudge("google/gemini-3.8-flash", "frontier", new Map());

    await expect(
      runEvaluation(
        { responder, judge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "dev",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS
        }
      )
    ).rejects.toThrow(JudgePolicyError);
  });

  it("blocks uncalibrated non-frontier Judges on val and holdout, and allows calibrated >= 90%", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    const responder = new DeterministicResponder(
      "gemini-3.8-flash",
      new Map([
        [
          "Which two services handle metering?",
          {
            answer: "Pulse and Ledger.",
            documentIds: ["dsid_b", "dsid_c"],
            latencyMs: 120,
            responseBytes: 64,
            usage: { inputTokens: 1000, outputTokens: 100 }
          }
        ]
      ])
    );
    const fastJudge = new DeterministicJudge("gemini-3.5-flash-lite", "fast", new Map());

    await expect(
      runEvaluation(
        { responder, judge: fastJudge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "val",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS
        }
      )
    ).rejects.toThrow(JudgePolicyError);

    const failedCalibration: CalibrationReport = {
      judgeModel: "gemini-3.5-flash-lite",
      frontierJudgeModel: "gemini-2.5-pro",
      sampleSize: 20,
      contradictionAgreement: 0.85,
      passed: false,
      timestamp: "2026-11-26T00:00:00.000Z"
    };

    await expect(
      runEvaluation(
        { responder, judge: fastJudge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "val",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS,
          calibrationReport: failedCalibration
        }
      )
    ).rejects.toThrow(JudgePolicyError);

    const passingCalibration: CalibrationReport = {
      judgeModel: "gemini-3.5-flash-lite",
      frontierJudgeModel: "gemini-2.5-pro",
      sampleSize: 20,
      contradictionAgreement: 0.95,
      passed: true,
      timestamp: "2026-11-26T00:00:00.000Z"
    };

    const manifest = await runEvaluation(
      {
        responder,
        judge: fastJudge,
        spendLedger: ledger,
        exporters: [],
        runsDir,
        validCorpusIds: VALID_CORPUS_IDS
      },
      {
        profile: TEST_PROFILE,
        split: "val",
        splitManifest: TEST_SPLIT_MANIFEST,
        questions: TEST_QUESTIONS,
        calibrationReport: passingCalibration,
        runId: "val-calibrated-1"
      }
    );

    expect(manifest.split).toBe("val");
    expect(manifest.metrics.compositeScore).toBeCloseTo(1);
  });

  it("blocks locked holdout without --unlock-holdout and enforces single holdout run", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    const responder = new DeterministicResponder(
      "gemini-3.8-flash",
      new Map([
        [
          "Who approved the Q4 rollout?",
          {
            answer: "Naomi approved the Q4 rollout.",
            documentIds: ["dsid_d"],
            latencyMs: 95,
            responseBytes: 60,
            usage: { inputTokens: 1000, outputTokens: 100 }
          }
        ]
      ])
    );
    const frontierJudge = new DeterministicJudge("gemini-2.5-pro", "frontier", new Map());

    await expect(
      runEvaluation(
        { responder, judge: frontierJudge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "holdout",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS
        }
      )
    ).rejects.toThrow(HoldoutLockedError);

    const firstHoldout = await runEvaluation(
      {
        responder,
        judge: frontierJudge,
        spendLedger: ledger,
        exporters: [],
        runsDir,
        validCorpusIds: VALID_CORPUS_IDS
      },
      {
        profile: TEST_PROFILE,
        split: "holdout",
        splitManifest: TEST_SPLIT_MANIFEST,
        questions: TEST_QUESTIONS,
        unlockHoldout: true,
        runId: "holdout-final-1"
      }
    );

    expect(firstHoldout.split).toBe("holdout");
    expect(firstHoldout.metrics.compositeScore).toBeCloseTo(1);

    await expect(
      runEvaluation(
        { responder, judge: frontierJudge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "holdout",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS,
          unlockHoldout: true,
          runId: "holdout-final-2"
        }
      )
    ).rejects.toThrow(HoldoutLockedError);
  });

  it("aborts pre-flight when spend ledger exceeds budget cap", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    ledger.record({
      timestamp: "2026-11-26T00:00:00.000Z",
      runId: "prior-run",
      split: "dev",
      bucket: "judge",
      model: "gemini-2.5-pro",
      inputTokens: 1_000_000,
      outputTokens: 1_000_000,
      usd: 40.0
    });

    const responder = new DeterministicResponder("gemini-3.8-flash", new Map());
    const judge = new DeterministicJudge("gemini-2.5-pro", "frontier", new Map());

    await expect(
      runEvaluation(
        { responder, judge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "dev",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS,
          budgetUsd: 40
        }
      )
    ).rejects.toThrow(BudgetExceededError);
  });

  it("writes immutable run manifest, logs spend buckets, applies Onyx zeroing, and renders terminal scorecard", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    const responder = new DeterministicResponder(
      "gemini-3.8-flash",
      new Map([
        [
          "Which vendor did Magnetar choose?",
          {
            answer: "Magnetar chose CompetitorX.",
            documentIds: ["dsid_a"],
            latencyMs: 180,
            responseBytes: 58,
            usage: { inputTokens: 10_000, outputTokens: 1_000 }
          }
        ],
        [
          "What is the secret Mars launch date?",
          {
            answer: "The corpus does not contain the Mars launch date.",
            documentIds: [],
            latencyMs: 140,
            responseBytes: 72,
            usage: { inputTokens: 8_000, outputTokens: 800 }
          }
        ]
      ])
    );
    const judge = new DeterministicJudge(
      "gemini-2.5-pro",
      "frontier",
      new Map([
        [
          "Which vendor did Magnetar choose?",
          {
            answerScore: 0.9,
            hasFabricatedOrContradictingClaim: true,
            abstained: false,
            reasoning: "Contradicts gold answer AetherAI."
          }
        ],
        [
          "What is the secret Mars launch date?",
          {
            answerScore: 1.0,
            hasFabricatedOrContradictingClaim: false,
            abstained: true,
            reasoning: "Proper abstention with empty citations."
          }
        ]
      ])
    );

    const renderedLines: string[] = [];
    const exporter = new TerminalScorecardExporter((text) => {
      renderedLines.push(text);
    });

    const manifest = await runEvaluation(
      {
        responder,
        judge,
        spendLedger: ledger,
        exporters: [exporter],
        runsDir,
        gitSha: "deadbee",
        validCorpusIds: VALID_CORPUS_IDS,
        now: () => new Date("2026-11-26T01:00:00.000Z")
      },
      {
        profile: TEST_PROFILE,
        split: "dev",
        splitManifest: TEST_SPLIT_MANIFEST,
        questions: TEST_QUESTIONS,
        runId: "dev-onyx-test"
      }
    );

    const manifestPath = join(runsDir, "dev-onyx-test.json");
    expect(existsSync(manifestPath)).toBe(true);
    const saved = EvalRunManifestSchema.parse(JSON.parse(readFileSync(manifestPath, "utf8")));
    expect(saved.runId).toBe("dev-onyx-test");

    const firstItem = manifest.items[0];
    const secondItem = manifest.items[1];
    expect(firstItem?.effectiveAnswerScore).toBe(0);
    expect(firstItem?.compositeScore).toBeCloseTo(0.3);
    expect(secondItem?.compositeScore).toBeCloseTo(1.0);
    expect(manifest.metrics.abstentionRecall).toBe(1);
    expect(manifest.metrics.falseAbstentionRate).toBe(0);
    expect(manifest.spendByBucketUsd.responder).toBeGreaterThan(0);
    expect(manifest.spendByBucketUsd.judge).toBeGreaterThan(0);
    expect(ledger.readEntries().length).toBe(4);

    expect(renderedLines.length).toBe(1);
    expect(renderedLines[0]).toBe(formatTerminalScorecard(manifest));
    expect(renderedLines[0]).toContain("Composite Score:");
    expect(renderedLines[0]).toContain("unanswerable");
  });

  it("supports --only and --stratum filtering and rejects empty filter selections", async () => {
    const { ledger, runsDir } = makeTempWorkspace();
    const responder = new DeterministicResponder("gemini-3.8-flash", new Map());
    const judge = new DeterministicJudge("gemini-2.5-pro", "frontier", new Map());

    const filtered = await runEvaluation(
      { responder, judge, spendLedger: ledger, exporters: [], runsDir },
      {
        profile: TEST_PROFILE,
        split: "dev",
        splitManifest: TEST_SPLIT_MANIFEST,
        questions: TEST_QUESTIONS,
        onlyIds: ["q_dev_2"],
        stratumFilter: "unanswerable",
        runId: "dev-filtered"
      }
    );

    expect(filtered.items.length).toBe(1);
    expect(filtered.items[0]?.questionId).toBe("q_dev_2");

    await expect(
      runEvaluation(
        { responder, judge, spendLedger: ledger, exporters: [], runsDir },
        {
          profile: TEST_PROFILE,
          split: "dev",
          splitManifest: TEST_SPLIT_MANIFEST,
          questions: TEST_QUESTIONS,
          stratumFilter: "multi_doc"
        }
      )
    ).rejects.toThrow(ValidationError);
  });
});

describe("calibrateJudge", () => {
  it("blocks split holdout before making any judge calls or ledger writes", async () => {
    const { ledger } = makeTempWorkspace();
    const candidate = new DeterministicJudge("gemini-3.5-flash-lite", "fast", new Map());
    const frontier = new DeterministicJudge("gemini-2.5-pro", "frontier", new Map());
    const sampleItem: JudgeInput = {
      question: "Q",
      goldAnswer: "Gold",
      answerFacts: ["Fact"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "Answer",
      responderDocIds: ["dsid_1"]
    };

    await expect(
      calibrateJudge(candidate, frontier, [sampleItem], {
        spendLedger: ledger,
        split: "holdout"
      })
    ).rejects.toThrow(HoldoutLockedError);

    expect(candidate.callCount).toBe(0);
    expect(frontier.callCount).toBe(0);
    expect(ledger.readEntries()).toEqual([]);
  });

  it("rejects non-frontier reference judge, identical models, or empty calibration items", async () => {
    const { ledger } = makeTempWorkspace();
    const fastA = new DeterministicJudge("gemini-3.5-flash-lite", "fast", new Map());
    const fastB = new DeterministicJudge("gemini-2.5-flash-lite", "fast", new Map());
    const frontier = new DeterministicJudge("gemini-2.5-pro", "frontier", new Map());
    const sampleItem: JudgeInput = {
      question: "Q",
      goldAnswer: "Gold",
      answerFacts: ["Fact"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "Answer",
      responderDocIds: ["dsid_1"]
    };

    await expect(
      calibrateJudge(fastA, fastB, [sampleItem], { spendLedger: ledger })
    ).rejects.toThrow(JudgePolicyError);

    await expect(
      calibrateJudge(frontier, frontier, [sampleItem], { spendLedger: ledger })
    ).rejects.toThrow(JudgePolicyError);

    await expect(
      calibrateJudge(fastA, frontier, [], { spendLedger: ledger })
    ).rejects.toThrow(ValidationError);
  });
  it("computes contradiction agreement and writes passing or failing report JSON", async () => {
    const { root, ledger } = makeTempWorkspace();
    const items: JudgeInput[] = Array.from({ length: 10 }, (_, index) => ({
      question: `Q${index}`,
      goldAnswer: "Gold",
      answerFacts: ["Fact"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "Answer",
      responderDocIds: ["dsid_1"]
    }));

    const frontierVerdicts = new Map<string, JudgeVerdict>(
      items.map((item, index) => [
        item.question,
        {
          answerScore: index % 2 === 0 ? 1 : 0,
          hasFabricatedOrContradictingClaim: index % 2 === 1,
          abstained: false,
          reasoning: "Frontier verdict"
        }
      ])
    );

    const passingCandidateVerdicts = new Map(frontierVerdicts);
    passingCandidateVerdicts.set("Q0", {
      answerScore: 0,
      hasFabricatedOrContradictingClaim: true,
      abstained: false,
      reasoning: "One disagreement out of 10"
    });

    const failingCandidateVerdicts = new Map(passingCandidateVerdicts);
    failingCandidateVerdicts.set("Q1", {
      answerScore: 1,
      hasFabricatedOrContradictingClaim: false,
      abstained: false,
      reasoning: "Second disagreement out of 10"
    });

    const frontierJudge = new DeterministicJudge("gemini-2.5-pro", "frontier", frontierVerdicts);
    const passingCandidate = new DeterministicJudge(
      "gemini-3.5-flash-lite",
      "fast",
      passingCandidateVerdicts
    );
    const failingCandidate = new DeterministicJudge(
      "gemini-2.5-flash-lite",
      "fast",
      failingCandidateVerdicts
    );

    const outputPath = join(root, "calibrations", "flash-lite.json");
    const passReport = await calibrateJudge(passingCandidate, frontierJudge, items, {
      spendLedger: ledger,
      outputPath
    });
    expect(passReport.passed).toBe(true);
    expect(passReport.contradictionAgreement).toBeCloseTo(0.9);
    const saved = CalibrationReportSchema.parse(JSON.parse(readFileSync(outputPath, "utf8")));
    expect(saved.passed).toBe(true);

    const failReport = await calibrateJudge(failingCandidate, frontierJudge, items, {
      spendLedger: ledger
    });
    expect(failReport.passed).toBe(false);
    expect(failReport.contradictionAgreement).toBeCloseTo(0.8);
  });
});

describe("evaluatePromotionGate", () => {
  it("promotes when dev clears noise band and val/guardrails hold", () => {
    const decision = evaluatePromotionGate(
      makeManifest({ compositeScore: 0.78, split: "dev" }),
      makeManifest({ compositeScore: 0.74, split: "dev" }),
      makeManifest({ compositeScore: 0.76, split: "val" }),
      makeManifest({ compositeScore: 0.75, split: "val" })
    );

    expect(decision.promoted).toBe(true);
    expect(decision.reasons).toEqual([]);
    expect(decision.devDelta).toBeCloseTo(0.04);
    expect(decision.valDelta).toBeCloseTo(0.01);
  });

  it("rejects with explicit reasons when any gate fails", () => {
    const decision = evaluatePromotionGate(
      makeManifest({
        compositeScore: 0.75,
        abstentionRecall: 0.8,
        invalidCitationsTotal: 1,
        latencyP95Ms: 70_000,
        split: "dev"
      }),
      makeManifest({ compositeScore: 0.74, abstentionRecall: 1.0, split: "dev" }),
      makeManifest({ compositeScore: 0.72, abstentionRecall: 0.9, split: "val" }),
      makeManifest({ compositeScore: 0.75, abstentionRecall: 1.0, split: "val" })
    );

    expect(decision.promoted).toBe(false);
    expect(decision.reasons.length).toBe(6);
  });

  it("throws ValidationError when manifests are passed in mismatched split slots", () => {
    const devManifest = makeManifest({ compositeScore: 0.8, split: "dev" });
    const valManifest = makeManifest({ compositeScore: 0.8, split: "val" });
    const holdoutManifest = makeManifest({ compositeScore: 0.8, split: "holdout" });

    expect(() =>
      evaluatePromotionGate(valManifest, devManifest, valManifest, valManifest)
    ).toThrow(ValidationError);

    expect(() =>
      evaluatePromotionGate(devManifest, valManifest, valManifest, valManifest)
    ).toThrow(ValidationError);

    expect(() =>
      evaluatePromotionGate(devManifest, devManifest, devManifest, valManifest)
    ).toThrow(ValidationError);

    expect(() =>
      evaluatePromotionGate(devManifest, devManifest, valManifest, devManifest)
    ).toThrow(ValidationError);

    expect(() =>
      evaluatePromotionGate(devManifest, devManifest, holdoutManifest, valManifest)
    ).toThrow(ValidationError);
  });
});
