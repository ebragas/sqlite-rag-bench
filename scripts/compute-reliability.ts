import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type QuestionStratum = "unanswerable" | "single_doc" | "two_doc" | "multi_doc";

export interface ReliabilityItemRecord {
  readonly questionId: string;
  readonly split: "val" | "holdout";
  readonly stratum: QuestionStratum;
  readonly question: string;
  readonly compositeScore: number;
  readonly effectiveAnswerScore: number;
  readonly citationF1: number;
  readonly invalidCitations: number;
  readonly fabricatedOrContradicted: boolean;
  readonly abstained: boolean;
  readonly latencyMs: number;
}

export interface PassBarCurve {
  readonly label: string;
  readonly definition: string;
  readonly passAtK: readonly number[];
  readonly passPowK: readonly number[];
}

export interface QuestionReliabilitySummary {
  readonly questionId: string;
  readonly split: "val" | "holdout";
  readonly stratum: QuestionStratum;
  readonly question: string;
  readonly trials: number;
  readonly meanComposite: number;
  readonly minComposite: number;
  readonly maxComposite: number;
  readonly strictWins: number;
  readonly perfectWins: number;
  readonly safetyWins: number;
  readonly fabrications: number;
  readonly falseAbstentions: number;
  readonly invalidCitations: number;
  readonly meanLatencyMs: number;
}

export interface GroupReliabilitySummary {
  readonly questionCount: number;
  readonly trialsPerQuestion: number;
  readonly meanComposite: number;
  readonly strict: PassBarCurve;
  readonly perfect: PassBarCurve;
  readonly safety: PassBarCurve;
}

export interface ReliabilityReport {
  readonly generatedAt: string;
  readonly gitSha: string;
  readonly profileName: string;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly trialsPerQuestion: number;
  readonly totalEvaluations: number;
  readonly trialCompositeScores: readonly number[];
  readonly trialMeanComposite: number;
  readonly trialStdComposite: number;
  readonly totalFabrications: number;
  readonly totalInvalidCitations: number;
  readonly totalFalseAbstentions: number;
  readonly spendUsd: {
    readonly responder: number;
    readonly judge: number;
    readonly total: number;
  };
  readonly overall: GroupReliabilitySummary;
  readonly bySplit: Readonly<Record<"val" | "holdout", GroupReliabilitySummary>>;
  readonly byStratum: Readonly<Record<QuestionStratum, GroupReliabilitySummary>>;
  readonly questions: readonly QuestionReliabilitySummary[];
}

interface RawManifestItem {
  readonly questionId: string;
  readonly stratum: QuestionStratum;
  readonly question: string;
  readonly compositeScore: number;
  readonly effectiveAnswerScore: number;
  readonly abstained: boolean;
  readonly latencyMs: number;
  readonly verdict: {
    readonly hasFabricatedOrContradictingClaim: boolean;
  };
  readonly citationScore: {
    readonly f1: number;
    readonly invalidCount: number;
  };
}

interface RawManifest {
  readonly split: "val" | "holdout";
  readonly profileName: string;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly gitSha: string;
  readonly items: readonly RawManifestItem[];
}

interface SpendLedgerRow {
  readonly bucket: string;
  readonly usd: number;
}

const ORDERED_STRATA: readonly QuestionStratum[] = [
  "unanswerable",
  "single_doc",
  "two_doc",
  "multi_doc"
];

/**
 * Unbiased estimator for pass@k (Chen et al., 2021):
 * probability that at least 1 of k distinct draws passes out of n trials with c passes.
 */
export function computePassAtK(n: number, c: number, k: number): number {
  if (n <= 0 || k <= 0 || k > n) {
    throw new RangeError(`Invalid pass@k parameters: n=${String(n)}, c=${String(c)}, k=${String(k)}`);
  }
  if (c <= 0) {
    return 0;
  }
  if (n - c < k) {
    return 1;
  }
  let failProb = 1;
  for (let i = 0; i < k; i += 1) {
    failProb *= (n - c - i) / (n - i);
  }
  return Math.max(0, Math.min(1, 1 - failProb));
}

/**
 * Unbiased estimator for pass^k (tau-bench, Yao et al., 2024):
 * probability that all k distinct draws pass out of n trials with c passes.
 */
export function computePassPowK(n: number, c: number, k: number): number {
  if (n <= 0 || k <= 0 || k > n) {
    throw new RangeError(`Invalid pass^k parameters: n=${String(n)}, c=${String(c)}, k=${String(k)}`);
  }
  if (c < k) {
    return 0;
  }
  let allPassProb = 1;
  for (let i = 0; i < k; i += 1) {
    allPassProb *= (c - i) / (n - i);
  }
  return Math.max(0, Math.min(1, allPassProb));
}

export function isStrictPass(item: ReliabilityItemRecord): boolean {
  return (
    item.compositeScore >= 0.8 &&
    !item.fabricatedOrContradicted &&
    item.invalidCitations === 0
  );
}

export function isPerfectPass(item: ReliabilityItemRecord): boolean {
  return (
    item.compositeScore >= 0.9999 &&
    !item.fabricatedOrContradicted &&
    item.invalidCitations === 0
  );
}

export function isSafetyPass(item: ReliabilityItemRecord): boolean {
  const falseAbstention = item.stratum !== "unanswerable" && item.abstained;
  return (
    !item.fabricatedOrContradicted &&
    item.invalidCitations === 0 &&
    !falseAbstention
  );
}

function round4(value: number): number {
  return Number(value.toFixed(4));
}

function summarizePassBar(
  questions: readonly QuestionReliabilitySummary[],
  trials: number,
  label: string,
  definition: string,
  getWins: (q: QuestionReliabilitySummary) => number
): PassBarCurve {
  const passAtK: number[] = [];
  const passPowK: number[] = [];
  for (let k = 1; k <= trials; k += 1) {
    if (questions.length === 0) {
      passAtK.push(0);
      passPowK.push(0);
      continue;
    }
    let sumAt = 0;
    let sumPow = 0;
    for (const q of questions) {
      const wins = getWins(q);
      sumAt += computePassAtK(q.trials, wins, k);
      sumPow += computePassPowK(q.trials, wins, k);
    }
    passAtK.push(round4(sumAt / questions.length));
    passPowK.push(round4(sumPow / questions.length));
  }
  return { label, definition, passAtK, passPowK };
}

function summarizeGroup(
  questions: readonly QuestionReliabilitySummary[],
  trials: number
): GroupReliabilitySummary {
  const meanComposite =
    questions.length === 0
      ? 0
      : round4(questions.reduce((acc, q) => acc + q.meanComposite, 0) / questions.length);

  return {
    questionCount: questions.length,
    trialsPerQuestion: trials,
    meanComposite,
    strict: summarizePassBar(
      questions,
      trials,
      "Strict Pass",
      "Composite Score >= 80%, 0 fabrications, 0 Invalid Citations",
      (q) => q.strictWins
    ),
    perfect: summarizePassBar(
      questions,
      trials,
      "Perfect Pass",
      "Composite Score = 100%, 0 fabrications, 0 Invalid Citations",
      (q) => q.perfectWins
    ),
    safety: summarizePassBar(
      questions,
      trials,
      "Safety Pass",
      "0 fabrications, 0 Invalid Citations, 0 False Abstentions",
      (q) => q.safetyWins
    )
  };
}

export function aggregateReliabilityTrials(
  trials: readonly (readonly ReliabilityItemRecord[])[],
  meta: {
    readonly gitSha: string;
    readonly profileName: string;
    readonly responderModel: string;
    readonly judgeModel: string;
    readonly spendUsd: { readonly responder: number; readonly judge: number; readonly total: number };
  }
): ReliabilityReport {
  const k = trials.length;
  if (k === 0) {
    throw new RangeError("At least one trial is required");
  }

  const byQuestion = new Map<string, ReliabilityItemRecord[]>();
  const trialCompositeScores: number[] = [];
  let totalFabrications = 0;
  let totalInvalidCitations = 0;
  let totalFalseAbstentions = 0;

  for (const trialItems of trials) {
    const trialMean =
      trialItems.length === 0
        ? 0
        : trialItems.reduce((acc, item) => acc + item.compositeScore, 0) / trialItems.length;
    trialCompositeScores.push(round4(trialMean));

    for (const item of trialItems) {
      const bucket = byQuestion.get(item.questionId) ?? [];
      bucket.push(item);
      byQuestion.set(item.questionId, bucket);
      if (item.fabricatedOrContradicted) {
        totalFabrications += 1;
      }
      totalInvalidCitations += item.invalidCitations;
      if (item.stratum !== "unanswerable" && item.abstained) {
        totalFalseAbstentions += 1;
      }
    }
  }

  const questions: QuestionReliabilitySummary[] = [];
  for (const [, records] of [...byQuestion.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const first = records[0];
    if (first === undefined) {
      continue;
    }
    const scores = records.map((r) => r.compositeScore);
    const meanComposite = round4(scores.reduce((acc, v) => acc + v, 0) / scores.length);
    const minComposite = round4(Math.min(...scores));
    const maxComposite = round4(Math.max(...scores));
    const meanLatencyMs = Math.round(
      records.reduce((acc, r) => acc + r.latencyMs, 0) / records.length
    );

    questions.push({
      questionId: first.questionId,
      split: first.split,
      stratum: first.stratum,
      question: first.question,
      trials: records.length,
      meanComposite,
      minComposite,
      maxComposite,
      strictWins: records.filter(isStrictPass).length,
      perfectWins: records.filter(isPerfectPass).length,
      safetyWins: records.filter(isSafetyPass).length,
      fabrications: records.filter((r) => r.fabricatedOrContradicted).length,
      falseAbstentions: records.filter((r) => r.stratum !== "unanswerable" && r.abstained).length,
      invalidCitations: records.reduce((acc, r) => acc + r.invalidCitations, 0),
      meanLatencyMs
    });
  }

  const trialMeanComposite = round4(
    trialCompositeScores.reduce((acc, v) => acc + v, 0) / trialCompositeScores.length
  );
  const variance =
    trialCompositeScores.length <= 1
      ? 0
      : trialCompositeScores.reduce((acc, v) => acc + (v - trialMeanComposite) ** 2, 0) /
        (trialCompositeScores.length - 1);
  const trialStdComposite = round4(Math.sqrt(variance));

  const bySplit: Record<"val" | "holdout", GroupReliabilitySummary> = {
    val: summarizeGroup(
      questions.filter((q) => q.split === "val"),
      k
    ),
    holdout: summarizeGroup(
      questions.filter((q) => q.split === "holdout"),
      k
    )
  };

  const byStratum = Object.fromEntries(
    ORDERED_STRATA.map((st) => [
      st,
      summarizeGroup(
        questions.filter((q) => q.stratum === st),
        k
      )
    ])
  ) as Record<QuestionStratum, GroupReliabilitySummary>;

  return {
    generatedAt: new Date().toISOString(),
    gitSha: meta.gitSha,
    profileName: meta.profileName,
    responderModel: meta.responderModel,
    judgeModel: meta.judgeModel,
    trialsPerQuestion: k,
    totalEvaluations: trials.reduce((acc, t) => acc + t.length, 0),
    trialCompositeScores,
    trialMeanComposite,
    trialStdComposite,
    totalFabrications,
    totalInvalidCitations,
    totalFalseAbstentions,
    spendUsd: {
      responder: Number(meta.spendUsd.responder.toFixed(2)),
      judge: Number(meta.spendUsd.judge.toFixed(2)),
      total: Number(meta.spendUsd.total.toFixed(2))
    },
    overall: summarizeGroup(questions, k),
    bySplit,
    byStratum,
    questions
  };
}

function readReliabilitySpend(repoRoot: string, trialCount: number): {
  readonly responder: number;
  readonly judge: number;
  readonly total: number;
} {
  let responder = 0;
  let judge = 0;
  for (let t = 1; t <= trialCount; t += 1) {
    const ledgerPath = resolve(repoRoot, `eval/spend-ledger-reliability-t${String(t)}.jsonl`);
    if (!existsSync(ledgerPath)) {
      continue;
    }
    const lines = readFileSync(ledgerPath, "utf8")
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    for (const line of lines) {
      const row = JSON.parse(line) as SpendLedgerRow;
      if (row.bucket === "responder") {
        responder += row.usd;
      } else if (row.bucket === "judge") {
        judge += row.usd;
      }
    }
  }
  return { responder, judge, total: responder + judge };
}

function loadTrialsFromDisk(runsRoot: string): {
  readonly trials: readonly (readonly ReliabilityItemRecord[])[];
  readonly gitSha: string;
  readonly profileName: string;
  readonly responderModel: string;
  readonly judgeModel: string;
} {
  const entries = readdirSync(runsRoot)
    .filter((name) => /^t\d+$/.test(name))
    .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));

  let gitSha = "unknown";
  let profileName = "prod";
  let responderModel = "gemini-3.8-flash";
  let judgeModel = "gemini-4-neon";
  const trials: ReliabilityItemRecord[][] = [];

  for (const trialDirName of entries) {
    const trialDir = join(runsRoot, trialDirName);
    const manifestFiles = readdirSync(trialDir)
      .filter((f) => f.endsWith(".json"))
      .sort();

    const items: ReliabilityItemRecord[] = [];
    for (const file of manifestFiles) {
      const manifest = JSON.parse(readFileSync(join(trialDir, file), "utf8")) as RawManifest;
      gitSha = manifest.gitSha;
      profileName = manifest.profileName;
      responderModel = manifest.responderModel;
      judgeModel = manifest.judgeModel;
      for (const item of manifest.items) {
        items.push({
          questionId: item.questionId,
          split: manifest.split,
          stratum: item.stratum,
          question: item.question,
          compositeScore: item.compositeScore,
          effectiveAnswerScore: item.effectiveAnswerScore,
          citationF1: item.citationScore.f1,
          invalidCitations: item.citationScore.invalidCount,
          fabricatedOrContradicted: item.verdict.hasFabricatedOrContradictingClaim,
          abstained: item.abstained,
          latencyMs: item.latencyMs
        });
      }
    }
    if (items.length > 0) {
      trials.push(items);
    }
  }

  return { trials, gitSha, profileName, responderModel, judgeModel };
}

function main(): void {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = resolve(scriptDir, "..");
  const runsRoot = resolve(repoRoot, "eval/runs/reliability-k10");
  const outputPath = resolve(repoRoot, "eval/reliability/prod-unseen-k10.json");

  const loaded = loadTrialsFromDisk(runsRoot);
  const spendUsd = readReliabilitySpend(repoRoot, loaded.trials.length);
  const report = aggregateReliabilityTrials(loaded.trials, {
    gitSha: loaded.gitSha,
    profileName: loaded.profileName,
    responderModel: loaded.responderModel,
    judgeModel: loaded.judgeModel,
    spendUsd
  });

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;
  const lastIdx = report.trialsPerQuestion - 1;
  console.log(`[compute-reliability] Wrote ${outputPath}`);
  console.log(
    `Trials: ${String(report.trialsPerQuestion)} x ${String(report.overall.questionCount)} unseen questions (${String(report.totalEvaluations)} evals), Spend: $${report.spendUsd.total.toFixed(2)}`
  );
  console.log(
    `Mean Composite: ${pct(report.trialMeanComposite)} (std ${(report.trialStdComposite * 100).toFixed(2)}%)`
  );
  console.log(
    `Strict (>=80%, 0 fab, 0 inv):  pass@1=${pct(report.overall.strict.passAtK[0] ?? 0)}  pass@${String(report.trialsPerQuestion)}=${pct(report.overall.strict.passAtK[lastIdx] ?? 0)}  pass^${String(report.trialsPerQuestion)}=${pct(report.overall.strict.passPowK[lastIdx] ?? 0)}`
  );
  console.log(
    `Perfect (100%):                pass@1=${pct(report.overall.perfect.passAtK[0] ?? 0)}  pass@${String(report.trialsPerQuestion)}=${pct(report.overall.perfect.passAtK[lastIdx] ?? 0)}  pass^${String(report.trialsPerQuestion)}=${pct(report.overall.perfect.passPowK[lastIdx] ?? 0)}`
  );
  console.log(
    `Safety (0 fab/inv/false-abs):  pass@1=${pct(report.overall.safety.passAtK[0] ?? 0)}  pass@${String(report.trialsPerQuestion)}=${pct(report.overall.safety.passAtK[lastIdx] ?? 0)}  pass^${String(report.trialsPerQuestion)}=${pct(report.overall.safety.passPowK[lastIdx] ?? 0)}`
  );
}

const currentFilePath = fileURLToPath(import.meta.url);
if (process.argv[1] !== undefined && resolve(process.argv[1]) === currentFilePath) {
  main();
}
