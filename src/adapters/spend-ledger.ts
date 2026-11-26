import { appendFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { ValidationError } from "../models/errors.js";
import {
  BudgetExceededError,
  SpendLedgerEntrySchema,
  type EvalSplitName,
  type SpendBucket,
  type SpendLedgerEntry
} from "../models/eval-harness.js";
import { readJsonlFile } from "./jsonl.js";

export interface SpendSummary {
  readonly totalUsd: number;
  readonly byBucket: {
    readonly embedding: number;
    readonly responder: number;
    readonly judge: number;
  };
  readonly bySplit: {
    readonly dev: number;
    readonly val: number;
    readonly holdout: number;
  };
  readonly entryCount: number;
}

export interface BudgetCheckOptions {
  readonly maxTotalUsd: number;
  readonly estimatedAdditionalUsd?: number;
  readonly maxBucketUsd?: Readonly<Partial<Record<SpendBucket, number>>>;
  readonly estimatedAdditionalByBucketUsd?: Readonly<Partial<Record<SpendBucket, number>>>;
}

const SPEND_BUCKETS: readonly SpendBucket[] = ["embedding", "responder", "judge"];

export class SpendLedgerStore {
  public readonly ledgerPath: string;

  public constructor(ledgerPath: string) {
    this.ledgerPath = ledgerPath;
  }

  public record(entry: SpendLedgerEntry): void {
    const validated = SpendLedgerEntrySchema.safeParse(entry);
    if (!validated.success) {
      throw new ValidationError(`Invalid spend ledger entry: ${validated.error.message}`);
    }
    mkdirSync(dirname(this.ledgerPath), { recursive: true });
    appendFileSync(this.ledgerPath, `${JSON.stringify(validated.data)}\n`, "utf8");
  }

  public readEntries(): SpendLedgerEntry[] {
    if (!existsSync(this.ledgerPath)) {
      return [];
    }
    return readJsonlFile(this.ledgerPath, SpendLedgerEntrySchema);
  }

  public summarize(): SpendSummary {
    const entries = this.readEntries();
    const byBucket: Record<SpendBucket, number> = {
      embedding: 0,
      responder: 0,
      judge: 0
    };
    const bySplit: Record<EvalSplitName, number> = {
      dev: 0,
      val: 0,
      holdout: 0
    };
    let totalUsd = 0;

    for (const entry of entries) {
      totalUsd += entry.usd;
      byBucket[entry.bucket] += entry.usd;
      bySplit[entry.split] += entry.usd;
    }

    return {
      totalUsd,
      byBucket: {
        embedding: byBucket.embedding,
        responder: byBucket.responder,
        judge: byBucket.judge
      },
      bySplit: {
        dev: bySplit.dev,
        val: bySplit.val,
        holdout: bySplit.holdout
      },
      entryCount: entries.length
    };
  }

  public assertWithinBudget(options: BudgetCheckOptions): SpendSummary {
    const summary = this.summarize();
    const estimatedTotal = options.estimatedAdditionalUsd ?? 0;

    if (!Number.isFinite(options.maxTotalUsd) || options.maxTotalUsd <= 0) {
      throw new BudgetExceededError(
        `Invalid total budget cap: ${options.maxTotalUsd}. Budget cap must be positive.`
      );
    }

    if (summary.totalUsd >= options.maxTotalUsd) {
      throw new BudgetExceededError(
        `Total spend $${summary.totalUsd.toFixed(4)} already reached cap $${options.maxTotalUsd.toFixed(2)}.`
      );
    }

    if (summary.totalUsd + estimatedTotal > options.maxTotalUsd) {
      throw new BudgetExceededError(
        `Projected total spend $${(summary.totalUsd + estimatedTotal).toFixed(4)} exceeds cap $${options.maxTotalUsd.toFixed(2)}.`
      );
    }

    if (options.maxBucketUsd !== undefined) {
      for (const bucket of SPEND_BUCKETS) {
        const cap = options.maxBucketUsd[bucket];
        if (cap === undefined) {
          continue;
        }
        const current = summary.byBucket[bucket];
        const estimated = options.estimatedAdditionalByBucketUsd?.[bucket] ?? 0;

        if (!Number.isFinite(cap) || cap <= 0) {
          throw new BudgetExceededError(
            `Invalid ${bucket} budget cap: ${cap}. Budget cap must be positive.`
          );
        }
        if (current >= cap) {
          throw new BudgetExceededError(
            `Bucket "${bucket}" spend $${current.toFixed(4)} already reached cap $${cap.toFixed(2)}.`
          );
        }
        if (current + estimated > cap) {
          throw new BudgetExceededError(
            `Projected "${bucket}" spend $${(current + estimated).toFixed(4)} exceeds cap $${cap.toFixed(2)}.`
          );
        }
      }
    }

    return summary;
  }
}
