import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { APICallError } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import {
  FrontierOneShotResponder,
  type FrontierGenerateInput,
  type FrontierGenerateOutput
} from "../../src/adapters/frontier-baseline.js";
import type { JudgeAdapter, JudgeEvaluation, JudgeInput } from "../../src/adapters/judge.js";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import { ConfigError, ContractError, ValidationError } from "../../src/models/errors.js";
import type { DevQuestion, SplitManifest } from "../../src/models/eval.js";
import { EvalRunManifestSchema } from "../../src/models/eval-harness.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import { runEvaluation } from "../../src/usecases/eval-runner.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

class StubFrontierJudge implements JudgeAdapter {
  public readonly modelId = "openai/gpt-4.1";
  public readonly tier = "frontier" as const;

  public evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    const unanswerable = input.expectedDocIds.length === 0;
    const abstained = input.responderAnswer.toLowerCase().includes("lacks enough evidence");
    const matched = unanswerable
      ? abstained
      : input.answerFacts.every((fact) =>
          input.responderAnswer.toLowerCase().includes(fact.toLowerCase())
        );

    return Promise.resolve({
      verdict: {
        answerScore: matched ? 1 : 0,
        hasFabricatedOrContradictingClaim: false,
        abstained,
        reasoning: matched ? "Covered required facts." : "Missed required facts."
      },
      usage: { inputTokens: 500, outputTokens: 80 }
    });
  }
}

describe("FrontierOneShotResponder", () => {
  it("returns structured answer, documentIds, responseBytes, usage, and injects optional context", async () => {
    const receivedInputs: FrontierGenerateInput[] = [];
    const responder = new FrontierOneShotResponder({
      modelId: "gemini-2.5-pro",
      corpusSummary: "Acme enterprise corpus covering Slack, Jira, and GitHub.",
      contextPrompt: "Use concise executive prose.",
      generateFn: (input: FrontierGenerateInput): Promise<FrontierGenerateOutput> => {
        receivedInputs.push(input);
        return Promise.resolve({
          output: {
            answer: "Magnetar selected AetherAI.",
            document_ids: ["dsid_a"]
          },
          usage: { inputTokens: 1200, outputTokens: 45 }
        });
      }
    });

    const result = await responder.answer("Which vendor did Magnetar choose?");

    expect(responder.modelId).toBe("gemini-2.5-pro");
    expect(receivedInputs).toHaveLength(1);
    expect(receivedInputs[0]?.system).toContain("Acme enterprise corpus");
    expect(receivedInputs[0]?.system).toContain("Use concise executive prose.");
    expect(receivedInputs[0]?.prompt).toContain("Which vendor did Magnetar choose?");

    expect(result.answer).toBe("Magnetar selected AetherAI.");
    expect(result.documentIds).toEqual(["dsid_a"]);
    expect(result.responseBytes).toBe(
      Buffer.byteLength(
        JSON.stringify({ answer: "Magnetar selected AetherAI.", document_ids: ["dsid_a"] }),
        "utf8"
      )
    );
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    expect(result.usage).toEqual({ inputTokens: 1200, outputTokens: 45 });
  });

  it("retries transient 429 and 5xx errors with bounded exponential backoff", async () => {
    let attempts = 0;
    const sleepDelays: number[] = [];

    const responder = new FrontierOneShotResponder({
      modelId: "openai/gpt-4.1",
      maxAttempts: 3,
      baseDelayMs: 250,
      maxDelayMs: 1000,
      sleepFn: (delayMs: number) => {
        sleepDelays.push(delayMs);
        return Promise.resolve();
      },
      generateFn: (): Promise<FrontierGenerateOutput> => {
        attempts += 1;
        if (attempts === 1) {
          return Promise.reject(
            new APICallError({
              message: "Rate limited",
              url: "https://api.openai.com/v1/responses",
              requestBodyValues: {},
              statusCode: 429,
              isRetryable: true
            })
          );
        }
        if (attempts === 2) {
          return Promise.reject(
            new APICallError({
              message: "Service unavailable",
              url: "https://api.openai.com/v1/responses",
              requestBodyValues: {},
              statusCode: 503,
              isRetryable: true
            })
          );
        }
        return Promise.resolve({
          output: {
            answer: "Pulse and Ledger handle metering.",
            document_ids: []
          },
          usage: { inputTokens: 900, outputTokens: 30 }
        });
      }
    });

    const result = await responder.answer("Which services handle metering?");

    expect(attempts).toBe(3);
    expect(sleepDelays).toEqual([250, 500]);
    expect(result.answer).toBe("Pulse and Ledger handle metering.");
    expect(result.documentIds).toEqual([]);
  });

  it("fails without retry on non-retryable errors and validates structured output", async () => {
    const sleepDelays: number[] = [];
    const nonRetryableResponder = new FrontierOneShotResponder({
      modelId: "gemini-2.5-pro",
      maxAttempts: 3,
      sleepFn: (delayMs: number) => {
        sleepDelays.push(delayMs);
        return Promise.resolve();
      },
      generateFn: (): Promise<FrontierGenerateOutput> =>
        Promise.reject(
          new APICallError({
            message: "Invalid request",
            url: "https://generativelanguage.googleapis.com/v1beta/models",
            requestBodyValues: {},
            statusCode: 400,
            isRetryable: false
          })
        )
    });

    await expect(nonRetryableResponder.answer("Test question")).rejects.toThrow(ContractError);
    expect(sleepDelays).toEqual([]);

    const invalidOutputResponder = new FrontierOneShotResponder({
      modelId: "gemini-2.5-pro",
      generateFn: (): Promise<FrontierGenerateOutput> =>
        Promise.resolve({
          output: {
            answer: "",
            document_ids: []
          },
          usage: { inputTokens: 100, outputTokens: 0 }
        })
    });

    await expect(invalidOutputResponder.answer("Test question")).rejects.toThrow(ValidationError);

    expect(
      () => new FrontierOneShotResponder({ modelId: "   ", generateFn: () => Promise.reject(new Error("unused")) })
    ).toThrow(ConfigError);
  });

  it("integrates cleanly with runEvaluation for spend ledger and manifest generation", async () => {
    const root = mkdtempSync(join(tmpdir(), "srb-frontier-baseline-"));
    tempDirs.push(root);
    const ledger = new SpendLedgerStore(join(root, "spend-ledger.jsonl"));
    const runsDir = join(root, "runs");

    const questions: readonly DevQuestion[] = [
      {
        question_id: "q_1",
        question: "Which vendor did Magnetar choose?",
        gold_answer: "Magnetar chose AetherAI.",
        answer_facts: ["AetherAI"],
        expected_doc_ids: ["dsid_a"],
        source_types: ["fireflies"]
      },
      {
        question_id: "q_2",
        question: "What is the unrecorded Mars launch date?",
        gold_answer: "Not in corpus.",
        answer_facts: [],
        expected_doc_ids: [],
        source_types: []
      }
    ];

    const splitManifest: SplitManifest = {
      version: 1,
      seed: 20261125,
      holdout_seed: 20261125,
      total_questions: 2,
      counts: {
        dev: { unanswerable: 1, single_doc: 1, two_doc: 0, multi_doc: 0 },
        val: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 },
        holdout: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 }
      },
      dev: ["q_1", "q_2"],
      val: [],
      holdout: []
    };

    const responder = new FrontierOneShotResponder({
      modelId: "gemini-2.5-pro",
      generateFn: (input: FrontierGenerateInput): Promise<FrontierGenerateOutput> => {
        if (input.prompt.includes("Mars")) {
          return Promise.resolve({
            output: {
              answer: "The corpus lacks enough evidence to answer this question.",
              document_ids: []
            },
            usage: { inputTokens: 800, outputTokens: 25 }
          });
        }
        return Promise.resolve({
          output: {
            answer: "Magnetar chose AetherAI.",
            document_ids: ["dsid_a"]
          },
          usage: { inputTokens: 1000, outputTokens: 35 }
        });
      }
    });

    const manifest = await runEvaluation(
      {
        responder,
        judge: new StubFrontierJudge(),
        spendLedger: ledger,
        exporters: [],
        runsDir,
        validCorpusIds: new Set(["dsid_a"]),
        gitSha: "deadbee"
      },
      {
        profile: RunProfileSchema.parse({ name: "frontier-oneshot" }),
        split: "dev",
        splitManifest,
        questions,
        runId: "frontier-baseline-test"
      }
    );

    expect(manifest.responderModel).toBe("gemini-2.5-pro");
    expect(manifest.judgeModel).toBe("openai/gpt-4.1");
    expect(manifest.metrics.totalItems).toBe(2);
    expect(manifest.metrics.compositeScore).toBe(1);
    expect(manifest.metrics.abstentionRecall).toBe(1);
    expect(manifest.spendByBucketUsd.responder).toBeGreaterThan(0);
    expect(manifest.spendByBucketUsd.judge).toBeGreaterThan(0);

    const savedManifest = EvalRunManifestSchema.parse(
      JSON.parse(readFileSync(join(runsDir, "frontier-baseline-test.json"), "utf8"))
    );
    expect(savedManifest.responderModel).toBe("gemini-2.5-pro");

    const entries = ledger.readEntries();
    expect(entries.filter((entry) => entry.bucket === "responder")).toHaveLength(2);
    expect(entries.filter((entry) => entry.bucket === "judge")).toHaveLength(2);
  });
});
