import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import { BudgetExceededError } from "../../src/models/eval-harness.js";

const tempDirs: string[] = [];

function makeTempLedger(): SpendLedgerStore {
  const dir = mkdtempSync(join(tmpdir(), "srb-spend-ledger-"));
  tempDirs.push(dir);
  return new SpendLedgerStore(join(dir, "eval", "spend-ledger.jsonl"));
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("SpendLedgerStore", () => {
  it("returns empty summary when ledger file does not exist yet", () => {
    const ledger = makeTempLedger();
    expect(ledger.readEntries()).toEqual([]);
    expect(ledger.summarize()).toEqual({
      totalUsd: 0,
      byBucket: { embedding: 0, responder: 0, judge: 0 },
      bySplit: { dev: 0, val: 0, holdout: 0 },
      entryCount: 0
    });
  });

  it("appends JSONL entries and reloads bucket and split totals", () => {
    const ledger = makeTempLedger();

    ledger.record({
      timestamp: "2026-11-26T00:00:00.000Z",
      runId: "run-1",
      split: "dev",
      bucket: "embedding",
      model: "gemini-embedding-001",
      inputTokens: 100_000,
      outputTokens: 0,
      usd: 0.015
    });
    ledger.record({
      timestamp: "2026-11-26T00:00:01.000Z",
      runId: "run-1",
      split: "dev",
      bucket: "responder",
      model: "gemini-3.8-flash",
      inputTokens: 200_000,
      outputTokens: 20_000,
      usd: 0.11
    });
    ledger.record({
      timestamp: "2026-11-26T00:00:02.000Z",
      runId: "run-2",
      split: "val",
      bucket: "judge",
      model: "gemini-2.5-pro",
      inputTokens: 100_000,
      outputTokens: 10_000,
      usd: 0.225
    });

    const reloaded = new SpendLedgerStore(ledger.ledgerPath);
    const summary = reloaded.summarize();

    expect(summary.entryCount).toBe(3);
    expect(summary.totalUsd).toBeCloseTo(0.35);
    expect(summary.byBucket.embedding).toBeCloseTo(0.015);
    expect(summary.byBucket.responder).toBeCloseTo(0.11);
    expect(summary.byBucket.judge).toBeCloseTo(0.225);
    expect(summary.bySplit.dev).toBeCloseTo(0.125);
    expect(summary.bySplit.val).toBeCloseTo(0.225);
    expect(summary.bySplit.holdout).toBe(0);
  });

  it("throws BudgetExceededError on total cap, projected cap, and per-bucket cap", () => {
    const ledger = makeTempLedger();

    ledger.record({
      timestamp: "2026-11-26T00:00:00.000Z",
      runId: "run-1",
      split: "dev",
      bucket: "judge",
      model: "gemini-2.5-pro",
      inputTokens: 100_000,
      outputTokens: 50_000,
      usd: 15.0
    });
    ledger.record({
      timestamp: "2026-11-26T00:00:01.000Z",
      runId: "run-1",
      split: "dev",
      bucket: "responder",
      model: "gemini-3.8-flash",
      inputTokens: 100_000,
      outputTokens: 50_000,
      usd: 25.0
    });

    expect(() => ledger.assertWithinBudget({ maxTotalUsd: 40 })).toThrow(BudgetExceededError);
    expect(() =>
      ledger.assertWithinBudget({ maxTotalUsd: 45, estimatedAdditionalUsd: 6 })
    ).toThrow(BudgetExceededError);
    expect(() =>
      ledger.assertWithinBudget({ maxTotalUsd: 100, maxBucketUsd: { judge: 15 } })
    ).toThrow(BudgetExceededError);
    expect(() =>
      ledger.assertWithinBudget({
        maxTotalUsd: 100,
        maxBucketUsd: { judge: 20 },
        estimatedAdditionalByBucketUsd: { judge: 6 }
      })
    ).toThrow(BudgetExceededError);

    const okSummary = ledger.assertWithinBudget({
      maxTotalUsd: 50,
      estimatedAdditionalUsd: 5,
      maxBucketUsd: { judge: 20, responder: 30 }
    });
    expect(okSummary.totalUsd).toBeCloseTo(40);
  });
});
