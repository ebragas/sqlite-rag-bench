import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import { ConfigError } from "../../src/models/errors.js";
import {
  BudgetExceededError,
  EvalRunManifestSchema,
  HoldoutLockedError
} from "../../src/models/eval-harness.js";
import { runCalibrateJudgeCli } from "../../src/services/calibrate-judge-cli.js";
import { inferJudgeTier, runEvalCli } from "../../src/services/eval-cli.js";

const tempDirs: string[] = [];
const openServers: Server[] = [];

afterEach(async () => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }
});

describe("inferJudgeTier", () => {
  it("fails closed to fast for unrecognized model IDs", () => {
    expect(inferJudgeTier("custom-unverified-judge")).toBe("fast");
    expect(inferJudgeTier("gemini-3.8-flash")).toBe("fast");
    expect(inferJudgeTier("gemini-3.5-flash-lite")).toBe("fast");
  });

  it("recognizes explicit frontier and decision judge models", () => {
    expect(inferJudgeTier("gemini-4-neon")).toBe("frontier");
    expect(inferJudgeTier("google/gemini-4-carbon")).toBe("frontier");
    expect(inferJudgeTier("gemini-4-argon")).toBe("frontier");
    expect(inferJudgeTier("gemini-3.1-pro-preview")).toBe("frontier");
    expect(inferJudgeTier("gemini-2.5-pro")).toBe("frontier");
    expect(inferJudgeTier("google/gemini-2.5-pro")).toBe("frontier");
    expect(inferJudgeTier("openai/gpt-4.1")).toBe("frontier");
    expect(inferJudgeTier("o3")).toBe("frontier");
    expect(inferJudgeTier("gemini-4-hydrogen")).toBe("fast");
    expect(inferJudgeTier("clef:27b-mxfp8")).toBe("decision");
  });
});

describe("runCalibrateJudgeCli pre-flight guards", () => {
  it("rejects --split holdout with HoldoutLockedError before reading files or calling /ask", async () => {
    await expect(runCalibrateJudgeCli(["--split", "holdout"])).rejects.toThrow(HoldoutLockedError);
  });

  it("enforces spend budget pre-flight in live mode before any POST /ask request", async () => {
    const root = mkdtempSync(join(tmpdir(), "srb-calibrate-cli-"));
    tempDirs.push(root);

    const ledgerPath = join(root, "spend-ledger.jsonl");
    const ledger = new SpendLedgerStore(ledgerPath);
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

    const questionsPath = join(root, "questions.jsonl");
    writeFileSync(
      questionsPath,
      `${JSON.stringify({
        question_id: "q_dev_1",
        question: "Which vendor did Magnetar choose?",
        gold_answer: "AetherAI",
        answer_facts: ["AetherAI"],
        expected_doc_ids: ["dsid_1"],
        source_types: ["fireflies"]
      })}\n`,
      "utf8"
    );

    const splitsPath = join(root, "splits.json");
    writeFileSync(
      splitsPath,
      JSON.stringify({
        version: 1,
        seed: 20261125,
        holdout_seed: 20261125,
        total_questions: 1,
        counts: {
          dev: { unanswerable: 0, single_doc: 1, two_doc: 0, multi_doc: 0 },
          val: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 },
          holdout: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 }
        },
        dev: ["q_dev_1"],
        val: [],
        holdout: []
      }),
      "utf8"
    );

    let askRequestsReceived = 0;
    const server = createServer((req, res) => {
      if (req.url === "/ask") {
        askRequestsReceived += 1;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ answer: "Should never be reached", document_ids: [] }));
    });
    openServers.push(server);
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const port = (server.address() as AddressInfo).port;

    await expect(
      runCalibrateJudgeCli([
        "--split",
        "dev",
        "--url",
        `http://127.0.0.1:${port}`,
        "--budget",
        "40",
        "--ledger",
        ledgerPath,
        "--questions",
        questionsPath,
        "--splits-file",
        splitsPath
      ])
    ).rejects.toThrow(BudgetExceededError);

    expect(askRequestsReceived).toBe(0);
  });

  it("validates --judge-provider and --judge-tier combinations", async () => {
    await expect(
      runCalibrateJudgeCli(["--judge-provider", "unknown"])
    ).rejects.toThrow(ConfigError);

    await expect(
      runCalibrateJudgeCli(["--judge-provider", "systemone", "--judge-tier", "fast"])
    ).rejects.toThrow(ConfigError);
  });
});

describe("runEvalCli provider and responder flags", () => {
  it("rejects invalid --judge-provider, invalid --responder, and conflicting flags", async () => {
    await expect(runEvalCli(["--judge-provider", "invalid"])).rejects.toThrow(ConfigError);
    await expect(runEvalCli(["--responder", "invalid"])).rejects.toThrow(ConfigError);
    await expect(
      runEvalCli(["--in-process", "--responder", "frontier-oneshot"])
    ).rejects.toThrow(ConfigError);
    await expect(
      runEvalCli(["--judge-provider", "systemone", "--judge-tier", "fast"])
    ).rejects.toThrow(ConfigError);
    await expect(
      runEvalCli(["--in-process"], { env: {} })
    ).rejects.toThrow(ConfigError);
    await expect(runEvalCli(["--concurrency", "0"])).rejects.toThrow(ConfigError);
    await expect(runEvalCli(["--concurrency", "1.5"])).rejects.toThrow(ConfigError);
  });

  it("runs dev evaluation end-to-end with --judge-provider systemone and writes a decision-tier manifest", async () => {
    const root = mkdtempSync(join(tmpdir(), "sqlite-rag-bench-eval-s1-"));
    tempDirs.push(root);

    const questionsPath = join(root, "questions.jsonl");
    writeFileSync(
      questionsPath,
      `${JSON.stringify({
        question_id: "q_dev_1",
        question: "Which vendor did Magnetar choose?",
        gold_answer: "Magnetar chose AetherAI.",
        answer_facts: ["AetherAI"],
        expected_doc_ids: ["dsid_1"],
        source_types: ["fireflies"]
      })}\n`,
      "utf8"
    );

    const splitsPath = join(root, "splits.json");
    writeFileSync(
      splitsPath,
      JSON.stringify({
        version: 1,
        seed: 20261125,
        holdout_seed: 20261125,
        total_questions: 1,
        counts: {
          dev: { unanswerable: 0, single_doc: 1, two_doc: 0, multi_doc: 0 },
          val: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 },
          holdout: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 }
        },
        dev: ["q_dev_1"],
        val: [],
        holdout: []
      }),
      "utf8"
    );

    const uuidIndexPath = join(root, "uuid_index.json");
    writeFileSync(uuidIndexPath, JSON.stringify({ dsid_1: "doc.json" }), "utf8");

    const ledgerPath = join(root, "spend-ledger.jsonl");
    const runsDir = join(root, "runs");

    const askServer = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          answer: "Magnetar chose AetherAI.",
          document_ids: ["dsid_1"]
        })
      );
    });
    openServers.push(askServer);
    await new Promise<void>((resolve) => {
      askServer.listen(0, "127.0.0.1", () => resolve());
    });
    const askPort = (askServer.address() as AddressInfo).port;

    const capturedAuth: Array<string | undefined> = [];
    const systemOneServer = createServer((req, res) => {
      capturedAuth.push(req.headers.authorization);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          model: "clef:27b-mxfp8",
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", noul: 0.04 },
            abstained: { type: "noul", noul: 0.01 },
            fact_0: { type: "noul", noul: 0.96 }
          },
          usage: { input_tokens: 300, output_tokens: 0 }
        })
      );
    });
    openServers.push(systemOneServer);
    await new Promise<void>((resolve) => {
      systemOneServer.listen(0, "127.0.0.1", () => resolve());
    });
    const systemOnePort = (systemOneServer.address() as AddressInfo).port;

    await runEvalCli([
      "--split",
      "dev",
      "--url",
      `http://127.0.0.1:${askPort}`,
      "--judge-provider",
      "systemone",
      "--judge",
      "clef:27b-mxfp8",
      "--systemone-url",
      `http://127.0.0.1:${systemOnePort}`,
      "--systemone-key",
      "cli-token",
      "--questions",
      questionsPath,
      "--splits-file",
      splitsPath,
      "--uuid-index",
      uuidIndexPath,
      "--ledger",
      ledgerPath,
      "--runs-dir",
      runsDir
    ]);

    expect(capturedAuth).toEqual(["Bearer cli-token"]);

    const runFiles = readdirSync(runsDir);
    expect(runFiles).toHaveLength(1);
    const manifest = EvalRunManifestSchema.parse(
      JSON.parse(readFileSync(join(runsDir, runFiles[0] ?? ""), "utf8"))
    );
    expect(manifest.judgeModel).toBe("clef:27b-mxfp8");
    expect(manifest.judgeTier).toBe("decision");
    expect(manifest.metrics.compositeScore).toBe(1);

    const judgeEntries = new SpendLedgerStore(ledgerPath)
      .readEntries()
      .filter((entry) => entry.bucket === "judge");
    expect(judgeEntries).toHaveLength(1);
    expect(judgeEntries[0]?.outputTokens).toBe(0);
  });
});
