import type { DevQuestion } from "../models/eval.js";
import type {
  EvalItemResult,
  EvalRunManifest,
  EvalSplitName,
  JudgeTier
} from "../models/eval-harness.js";
import type { RunProfile } from "../models/profile.js";

export interface ObservabilityExporter {
  onRunComplete(manifest: EvalRunManifest): Promise<void>;
}

export interface ExperimentRunContext {
  readonly runId: string;
  readonly timestamp: string;
  readonly gitSha: string;
  readonly split: EvalSplitName;
  readonly splitSeed: number;
  readonly profile: RunProfile;
  readonly indexConfigHash: string;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly judgeTier: JudgeTier;
  readonly concurrency?: number;
  readonly questions: readonly DevQuestion[];
}

export interface ExperimentItemExecutionContext {
  readonly headers: Readonly<Record<string, string>>;
}

export interface ExperimentRunnerWrapper {
  runExperiment(
    context: ExperimentRunContext,
    executeItem: (
      question: DevQuestion,
      itemContext: ExperimentItemExecutionContext
    ) => Promise<EvalItemResult>,
    finalizeManifest: (items: readonly EvalItemResult[]) => Promise<EvalRunManifest>
  ): Promise<EvalRunManifest>;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatUsd(value: number): string {
  return `$${value.toFixed(4)}`;
}

export function formatTerminalScorecard(manifest: EvalRunManifest): string {
  const { metrics, spendByBucketUsd } = manifest;
  const strata = [
    ["unanswerable", metrics.byStratum.unanswerable],
    ["single_doc", metrics.byStratum.single_doc],
    ["two_doc", metrics.byStratum.two_doc],
    ["multi_doc", metrics.byStratum.multi_doc]
  ] as const;

  const lines: string[] = [
    "=== Evaluation Scorecard ===",
    `Run ID:      ${manifest.runId}`,
    `Split:       ${manifest.split} (seed ${manifest.splitSeed}, items ${metrics.totalItems})`,
    `Profile:     ${manifest.profile.name} (index ${manifest.indexConfigHash.slice(0, 12)})`,
    `Responder:   ${manifest.responderModel}`,
    `Judge:       ${manifest.judgeModel} [${manifest.judgeTier}]`,
    `Git SHA:     ${manifest.gitSha}`,
    "",
    "Headline Metrics:",
    `  Composite Score:       ${formatPercent(metrics.compositeScore)}`,
    `  Answer Score:          ${formatPercent(metrics.answerScore)}`,
    `  Citation Score:        ${formatPercent(metrics.citationScore)}`,
    `  Abstention Recall:     ${formatPercent(metrics.abstentionRecall)}`,
    `  False Abstention Rate: ${formatPercent(metrics.falseAbstentionRate)}`,
    `  Invalid Citations:     ${metrics.invalidCitationsTotal}`,
    `  Latency p50 / p95:     ${Math.round(metrics.latencyP50Ms)} ms / ${Math.round(metrics.latencyP95Ms)} ms`,
    "",
    "Per-Stratum Breakdown:",
    "  Stratum       Count  Composite  Answer  Citation"
  ];

  for (const [name, row] of strata) {
    lines.push(
      `  ${name.padEnd(13)} ${String(row.count).padStart(5)}  ${formatPercent(row.compositeScore).padStart(9)}  ${formatPercent(row.answerScore).padStart(6)}  ${formatPercent(row.citationScore).padStart(8)}`
    );
  }

  lines.push(
    "",
    "Spend Breakdown (USD):",
    `  Total:       ${formatUsd(manifest.totalSpendUsd)}`,
    `  embedding:   ${formatUsd(spendByBucketUsd.embedding)}`,
    `  responder:   ${formatUsd(spendByBucketUsd.responder)}`,
    `  judge:       ${formatUsd(spendByBucketUsd.judge)}`
  );

  return lines.join("\n");
}

export class TerminalScorecardExporter implements ObservabilityExporter {
  private readonly writeLine: (text: string) => void;

  public constructor(writeLine?: (text: string) => void) {
    this.writeLine =
      writeLine ??
      ((text: string) => {
        process.stdout.write(`${text}\n`);
      });
  }

  public onRunComplete(manifest: EvalRunManifest): Promise<void> {
    this.writeLine(formatTerminalScorecard(manifest));
    return Promise.resolve();
  }
}
