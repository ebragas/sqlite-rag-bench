import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { JudgeAdapter, JudgeEvaluation, JudgeInput } from "../../src/adapters/judge.js";
import {
  buildItemScores,
  buildLangfuseExperimentPayload,
  buildRunScores,
  formatW3cTraceparent,
  LangfuseExperimentExporter,
  type LangfuseEvaluation,
  type LangfuseExperimentClient,
  type LangfuseExperimentRunConfig
} from "../../src/adapters/langfuse-exporter.js";
import { LiveHttpResponder } from "../../src/adapters/responder.js";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import {
  createNoopTelemetryController,
  type TelemetryController
} from "../../src/adapters/telemetry.js";
import type { DevQuestion, SplitManifest } from "../../src/models/eval.js";
import {
  EvalItemResultSchema,
  EvalRunManifestSchema,
  type EvalItemResult,
  type EvalRunManifest,
  type JudgeTier,
  type JudgeVerdict
} from "../../src/models/eval-harness.js";
import { computeIndexConfigHash, RunProfileSchema } from "../../src/models/profile.js";
import { runEvaluation } from "../../src/usecases/eval-runner.js";

const tempDirs: string[] = [];
const openServers: Server[] = [];

afterEach(async () => {
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

const SAMPLE_PROFILE = RunProfileSchema.parse({
  name: "l1-agent",
  topK: 20,
  preSearchEnabled: true,
  enabledTools: ["search", "fetch_docs"],
  maxAgentTurns: 4,
  synthesisModel: "gemini-3.8-flash"
});

const SAMPLE_QUESTIONS: readonly DevQuestion[] = [
  {
    question_id: "q_single_1",
    question: "Which vendor did Magnetar choose?",
    gold_answer: "Magnetar chose AetherAI.",
    answer_facts: ["Magnetar chose AetherAI"],
    expected_doc_ids: ["dsid_a"],
    source_types: ["fireflies"]
  },
  {
    question_id: "q_unans_1",
    question: "What is the fictional Mars launch date?",
    gold_answer: "The corpus does not contain this information.",
    answer_facts: [],
    expected_doc_ids: [],
    source_types: []
  }
];

const SAMPLE_SPLIT_MANIFEST: SplitManifest = {
  version: 1,
  seed: 20261125,
  holdout_seed: 20261125,
  total_questions: 2,
  counts: {
    dev: { unanswerable: 1, single_doc: 1, two_doc: 0, multi_doc: 0 },
    val: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 },
    holdout: { unanswerable: 0, single_doc: 0, two_doc: 0, multi_doc: 0 }
  },
  dev: ["q_single_1", "q_unans_1"],
  val: [],
  holdout: []
};

function makeItemResult(overrides: Partial<EvalItemResult> = {}): EvalItemResult {
  return EvalItemResultSchema.parse({
    questionId: "q_single_1",
    stratum: "single_doc",
    question: "Which vendor did Magnetar choose?",
    answer: "Magnetar chose AetherAI.",
    returnedDocumentIds: ["dsid_a"],
    expectedDocumentIds: ["dsid_a"],
    verdict: {
      answerScore: 1,
      hasFabricatedOrContradictingClaim: false,
      abstained: false,
      reasoning: "Covered all required facts."
    },
    effectiveAnswerScore: 1,
    citationScore: {
      precision: 1,
      recall: 1,
      f1: 1,
      invalidCount: 0
    },
    compositeScore: 1,
    abstained: false,
    latencyMs: 240,
    responseBytes: 64,
    responderCostUsd: 0.002,
    judgeCostUsd: 0.005,
    ...overrides
  });
}

function makeManifest(items: readonly EvalItemResult[]): EvalRunManifest {
  return EvalRunManifestSchema.parse({
    runId: "2026-11-26T00-00-00-000Z_dev_l1-agent",
    timestamp: "2026-11-26T00:00:00.000Z",
    gitSha: "abc1234",
    split: "dev",
    splitSeed: 20261125,
    profile: SAMPLE_PROFILE,
    indexConfigHash: computeIndexConfigHash(SAMPLE_PROFILE),
    responderModel: "gemini-3.8-flash",
    judgeModel: "gemini-2.5-pro",
    judgeTier: "frontier",
    metrics: {
      totalItems: items.length,
      compositeScore: 0.85,
      answerScore: 0.8,
      citationScore: 0.95,
      abstentionRecall: 1,
      falseAbstentionRate: 0.1,
      invalidCitationsTotal: 1,
      latencyP50Ms: 240,
      latencyP95Ms: 420,
      byStratum: {
        unanswerable: { count: 1, compositeScore: 1, answerScore: 1, citationScore: 1 },
        single_doc: { count: 1, compositeScore: 0.7, answerScore: 0.6, citationScore: 0.9 },
        two_doc: { count: 0, compositeScore: 0, answerScore: 0, citationScore: 0 },
        multi_doc: { count: 0, compositeScore: 0, answerScore: 0, citationScore: 0 }
      }
    },
    spendByBucketUsd: {
      embedding: 0.01,
      responder: 0.02,
      judge: 0.05
    },
    totalSpendUsd: 0.08,
    items: [...items]
  });
}

class DeterministicJudge implements JudgeAdapter {
  public readonly modelId = "gemini-2.5-pro";
  public readonly tier: JudgeTier = "frontier";
  private readonly verdicts: ReadonlyMap<string, JudgeVerdict>;

  public constructor(verdicts: ReadonlyMap<string, JudgeVerdict>) {
    this.verdicts = verdicts;
  }

  public evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    const verdict = this.verdicts.get(input.question) ?? {
      answerScore: 1,
      hasFabricatedOrContradictingClaim: false,
      abstained: input.expectedDocIds.length === 0,
      reasoning: "Matched gold answer."
    };
    return Promise.resolve({
      verdict,
      usage: { inputTokens: 1000, outputTokens: 100 }
    });
  }
}

interface RecordedExperimentExecution {
  readonly config: LangfuseExperimentRunConfig;
  readonly itemEvaluations: ReadonlyMap<string, readonly LangfuseEvaluation[]>;
  readonly runEvaluations: readonly LangfuseEvaluation[];
}

function createFakeLangfuseClient(): {
  client: LangfuseExperimentClient;
  executions: RecordedExperimentExecution[];
  getFlushCount(): number;
  getShutdownCount(): number;
} {
  const executions: RecordedExperimentExecution[] = [];
  let flushCount = 0;
  let shutdownCount = 0;

  const client: LangfuseExperimentClient = {
    experiment: {
      async run(config: LangfuseExperimentRunConfig): Promise<unknown> {
        const itemEvaluations = new Map<string, LangfuseEvaluation[]>();
        const itemResults: Array<{
          item: (typeof config.data)[number];
          output: unknown;
          evaluations: LangfuseEvaluation[];
        }> = [];

        for (const item of config.data) {
          const output = await config.task(item);
          const evaluations: LangfuseEvaluation[] = [];
          for (const evaluator of config.evaluators ?? []) {
            const produced = await evaluator({
              input: item.input,
              output,
              expectedOutput: item.expectedOutput,
              metadata: item.metadata
            });
            evaluations.push(...(Array.isArray(produced) ? produced : [produced]));
          }
          itemEvaluations.set(item.id, evaluations);
          itemResults.push({ item, output, evaluations });
        }

        const runEvaluations: LangfuseEvaluation[] = [];
        for (const runEvaluator of config.runEvaluators ?? []) {
          const produced = await runEvaluator({ itemResults });
          runEvaluations.push(...(Array.isArray(produced) ? produced : [produced]));
        }

        executions.push({ config, itemEvaluations, runEvaluations });
        return { runName: config.runName, itemResults, runEvaluations };
      }
    },
    flush(): Promise<void> {
      flushCount += 1;
      return Promise.resolve();
    },
    shutdown(): Promise<void> {
      shutdownCount += 1;
      return Promise.resolve();
    }
  };

  return {
    client,
    executions,
    getFlushCount: () => flushCount,
    getShutdownCount: () => shutdownCount
  };
}

describe("Langfuse pure payload and score builders", () => {
  it("builds local split array payload with stable question IDs and full Run Profile metadata", () => {
    const manifest = makeManifest([
      makeItemResult(),
      makeItemResult({
        questionId: "q_unans_1",
        stratum: "unanswerable",
        question: "What is the fictional Mars launch date?",
        answer: "The corpus does not contain this information.",
        returnedDocumentIds: [],
        expectedDocumentIds: [],
        abstained: true
      })
    ]);

    const payload = buildLangfuseExperimentPayload({ kind: "manifest", manifest });
    expect(payload.name).toBe("dev:l1-agent");
    expect(payload.runName).toBe(manifest.runId);
    expect(payload.data.map((item) => item.id)).toEqual(["q_single_1", "q_unans_1"]);
    expect(payload.metadata).toEqual({
      profileName: "l1-agent",
      profile: SAMPLE_PROFILE,
      indexConfigHash: manifest.indexConfigHash,
      gitSha: "abc1234",
      split: "dev",
      splitSeed: 20261125,
      responderModel: "gemini-3.8-flash",
      judgeModel: "gemini-2.5-pro",
      judgeTier: "frontier",
      spendByBucketUsd: {
        embedding: 0.01,
        responder: 0.02,
        judge: 0.05
      }
    });

    const traceparent = formatW3cTraceparent(
      "0af7651916cd43dd8448eb211c80319c",
      "b7ad6b7169203331",
      1
    );
    expect(traceparent).toBe("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01");
  });

  it("builds all 6 item scores accurately across answerable, unanswerable, false-abstention, and invalid-citation cases", () => {
    const falseAbstentionScores = buildItemScores(
      makeItemResult({
        stratum: "single_doc",
        abstained: true,
        effectiveAnswerScore: 0,
        compositeScore: 0,
        citationScore: { precision: 0, recall: 0, f1: 0, invalidCount: 2 },
        verdict: {
          answerScore: 0.5,
          hasFabricatedOrContradictingClaim: true,
          abstained: true,
          reasoning: "Abstained despite evidence and cited invalid IDs."
        }
      })
    );

    const byName = new Map(falseAbstentionScores.map((score) => [score.name, score]));
    expect([...byName.keys()]).toEqual([
      "composite_score",
      "answer_score",
      "citation_score",
      "abstained",
      "false_abstention",
      "invalid_citations"
    ]);
    expect(byName.get("composite_score")?.value).toBe(0);
    expect(byName.get("answer_score")?.value).toBe(0);
    expect(byName.get("citation_score")?.value).toBe(0);
    expect(byName.get("abstained")?.value).toBe(1);
    expect(byName.get("false_abstention")?.value).toBe(1);
    expect(byName.get("invalid_citations")?.value).toBe(2);

    const unanswerableScores = buildItemScores(
      makeItemResult({
        stratum: "unanswerable",
        abstained: true,
        returnedDocumentIds: [],
        expectedDocumentIds: []
      })
    );
    const unansMap = new Map(unanswerableScores.map((score) => [score.name, score]));
    expect(unansMap.get("abstained")?.value).toBe(1);
    expect(unansMap.get("false_abstention")?.value).toBe(0);
  });

  it("builds suite-level and per-stratum aggregate run scores", () => {
    const manifest = makeManifest([makeItemResult()]);
    const runScores = buildRunScores(manifest);
    const byName = new Map(runScores.map((score) => [score.name, score.value]));

    expect(byName.get("composite_score")).toBe(0.85);
    expect(byName.get("answer_score")).toBe(0.8);
    expect(byName.get("citation_score")).toBe(0.95);
    expect(byName.get("abstention_recall")).toBe(1);
    expect(byName.get("false_abstention_rate")).toBe(0.1);
    expect(byName.get("invalid_citations_total")).toBe(1);
    expect(byName.get("latency_p50_ms")).toBe(240);
    expect(byName.get("latency_p95_ms")).toBe(420);
    expect(byName.get("stratum_unanswerable_composite")).toBe(1);
    expect(byName.get("stratum_single_doc_composite")).toBe(0.7);
    expect(byName.get("stratum_single_doc_answer")).toBe(0.6);
    expect(byName.get("stratum_single_doc_citation")).toBe(0.9);
    expect(byName.get("stratum_two_doc_composite")).toBe(0);
    expect(byName.get("stratum_multi_doc_composite")).toBe(0);
    expect(byName.get("spend_embedding_usd")).toBe(0.01);
    expect(byName.get("spend_responder_usd")).toBe(0.02);
    expect(byName.get("spend_judge_usd")).toBe(0.05);
    expect(byName.get("spend_total_usd")).toBe(0.08);
  });
});

describe("LangfuseExperimentExporter", () => {
  it("stays strict no-op without loading Langfuse when keys are absent", async () => {
    let factoryCalls = 0;
    const exporter = new LangfuseExperimentExporter({
      env: {},
      clientFactory: () => {
        factoryCalls += 1;
        return createFakeLangfuseClient().client;
      }
    });

    expect(exporter.enabled).toBe(false);
    const manifest = makeManifest([makeItemResult()]);
    await exporter.onRunComplete(manifest);
    expect(factoryCalls).toBe(0);
  });

  it("wraps runEvaluation in experiment.run, propagates W3C traceparent to LiveHttpResponder, and avoids double export", async () => {
    const root = mkdtempSync(join(tmpdir(), "srb-langfuse-eval-"));
    tempDirs.push(root);

    const receivedTraceparents: Array<string | undefined> = [];
    const server = createServer((req, res) => {
      const header = req.headers.traceparent;
      receivedTraceparents.push(typeof header === "string" ? header : undefined);

      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as { question: string };
        const isUnanswerable = body.question.includes("Mars");
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify(
            isUnanswerable
              ? {
                  answer: "The corpus does not contain this information.",
                  document_ids: []
                }
              : {
                  answer: "Magnetar chose AetherAI.",
                  document_ids: ["dsid_a"]
                }
          )
        );
      });
    });
    openServers.push(server);
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const port = (server.address() as AddressInfo).port;

    let itemIndex = 0;
    const expectedHeaders = [
      formatW3cTraceparent("11111111111111111111111111111111", "1111111111111111", 1),
      formatW3cTraceparent("22222222222222222222222222222222", "2222222222222222", 1)
    ];

    const stubTelemetry: TelemetryController = {
      ...createNoopTelemetryController(),
      enabled: true,
      injectTraceHeaders(headers?: Readonly<Record<string, string>>): Record<string, string> {
        const nextHeader = expectedHeaders[itemIndex] ?? expectedHeaders[0] ?? "";
        itemIndex += 1;
        return { ...headers, traceparent: nextHeader };
      }
    };

    const fake = createFakeLangfuseClient();
    const exporter = new LangfuseExperimentExporter({
      env: {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      telemetry: stubTelemetry,
      clientFactory: () => fake.client
    });

    const ledger = new SpendLedgerStore(join(root, "spend-ledger.jsonl"));
    const responder = new LiveHttpResponder({
      baseUrl: `http://127.0.0.1:${port}`,
      modelId: "gemini-3.8-flash"
    });
    const judge = new DeterministicJudge(new Map());

    const manifest = await runEvaluation(
      {
        responder,
        judge,
        spendLedger: ledger,
        exporters: [exporter],
        experimentWrapper: exporter,
        runsDir: join(root, "runs"),
        gitSha: "feed123",
        validCorpusIds: new Set(["dsid_a"]),
        now: () => new Date("2026-11-26T01:00:00.000Z")
      },
      {
        profile: SAMPLE_PROFILE,
        split: "dev",
        splitManifest: SAMPLE_SPLIT_MANIFEST,
        questions: SAMPLE_QUESTIONS,
        concurrency: 4,
        runId: "langfuse-wrapped-run"
      }
    );

    expect(manifest.items).toHaveLength(2);
    expect(receivedTraceparents).toEqual(expectedHeaders);
    expect(fake.executions).toHaveLength(1);

    const execution = fake.executions[0];
    expect(execution?.config.runName).toBe("langfuse-wrapped-run");
    expect(execution?.config.maxConcurrency).toBe(4);
    expect(execution?.config.data.map((item) => item.id)).toEqual(["q_single_1", "q_unans_1"]);
    expect(execution?.config.metadata.profileName).toBe("l1-agent");
    expect(execution?.config.metadata.profile).toEqual(SAMPLE_PROFILE);
    expect(execution?.config.metadata.gitSha).toBe("feed123");
    expect(execution?.config.metadata.split).toBe("dev");
    expect(execution?.config.metadata.splitSeed).toBe(20261125);
    expect(execution?.config.metadata.spendByBucketUsd).toEqual(manifest.spendByBucketUsd);
    expect(execution?.config.metadata.spendByBucketUsd.judge).toBeGreaterThan(0);

    const firstItemScores = execution?.itemEvaluations.get("q_single_1") ?? [];
    expect(firstItemScores.map((s) => s.name)).toEqual([
      "composite_score",
      "answer_score",
      "citation_score",
      "abstained",
      "false_abstention",
      "invalid_citations"
    ]);
    const runScoresByName = new Map(
      (execution?.runEvaluations ?? []).map((score) => [score.name, score.value])
    );
    expect(runScoresByName.has("composite_score")).toBe(true);
    expect(runScoresByName.has("stratum_unanswerable_composite")).toBe(true);
    expect(runScoresByName.get("spend_embedding_usd")).toBe(manifest.spendByBucketUsd.embedding);
    expect(runScoresByName.get("spend_responder_usd")).toBe(manifest.spendByBucketUsd.responder);
    expect(runScoresByName.get("spend_judge_usd")).toBe(manifest.spendByBucketUsd.judge);
    expect(runScoresByName.get("spend_total_usd")).toBe(manifest.totalSpendUsd);
    expect(runScoresByName.get("spend_judge_usd")).toBeGreaterThan(0);
    expect(runScoresByName.get("spend_total_usd")).toBeGreaterThan(0);
    expect(fake.getFlushCount()).toBeGreaterThanOrEqual(1);

    await exporter.shutdown();
    expect(fake.getShutdownCount()).toBe(1);
  });

  it("exports completed manifests via onRunComplete when used standalone", async () => {
    const fake = createFakeLangfuseClient();
    const exporter = new LangfuseExperimentExporter({
      env: {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      clientFactory: () => fake.client
    });

    const manifest = makeManifest([makeItemResult()]);
    await exporter.onRunComplete(manifest);

    expect(fake.executions).toHaveLength(1);
    expect(fake.executions[0]?.config.runName).toBe(manifest.runId);
    expect(fake.executions[0]?.config.metadata.spendByBucketUsd).toEqual({
      embedding: 0.01,
      responder: 0.02,
      judge: 0.05
    });
    expect(fake.executions[0]?.itemEvaluations.get("q_single_1")).toHaveLength(6);
    expect(fake.getFlushCount()).toBe(1);
  });

  it("warns and degrades cleanly when @langfuse/client devDependency is omitted", async () => {
    const warnings: string[] = [];
    const exporter = new LangfuseExperimentExporter({
      env: {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      loadModule: (specifier) => Promise.reject(new Error(`Cannot find package '${specifier}'`)),
      onWarning: (message) => {
        warnings.push(message);
      }
    });

    await exporter.onRunComplete(makeManifest([makeItemResult()]));
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("@langfuse/client is unavailable");
  });

  it("short-circuits remaining items under Promise.allSettled when item 1 fails", async () => {
    const taskAttemptedIds: string[] = [];
    const runEvaluatorResults: LangfuseEvaluation[] = [];

    const allSettledClient: LangfuseExperimentClient = {
      experiment: {
        async run(config: LangfuseExperimentRunConfig): Promise<unknown> {
          const settledOutputs: Array<PromiseSettledResult<unknown>> = [];
          for (const item of config.data) {
            taskAttemptedIds.push(item.id);
            const [outcome] = await Promise.allSettled([
              Promise.resolve().then(() => config.task(item))
            ]);
            if (outcome !== undefined) {
              settledOutputs.push(outcome);
            }
          }
          for (const runEvaluator of config.runEvaluators ?? []) {
            const produced = await runEvaluator({ itemResults: [] });
            runEvaluatorResults.push(...(Array.isArray(produced) ? produced : [produced]));
          }
          return { runName: config.runName, settledOutputs };
        }
      },
      flush: () => Promise.resolve(),
      shutdown: () => Promise.resolve()
    };

    const exporter = new LangfuseExperimentExporter({
      env: {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      clientFactory: () => allSettledClient
    });

    const threeQuestions: readonly DevQuestion[] = [
      ...SAMPLE_QUESTIONS,
      {
        question_id: "q_single_3",
        question: "Who approved the Q4 budget?",
        gold_answer: "Finance approved the Q4 budget.",
        answer_facts: ["Finance approved the Q4 budget"],
        expected_doc_ids: ["dsid_c"],
        source_types: ["email"]
      }
    ];

    const executedIds: string[] = [];
    let finalizeCalls = 0;

    await expect(
      exporter.runExperiment(
        {
          runId: "short-circuit-run",
          timestamp: "2026-11-26T01:15:00.000Z",
          gitSha: "deadbeef",
          split: "dev",
          splitSeed: 20261125,
          profile: SAMPLE_PROFILE,
          indexConfigHash: computeIndexConfigHash(SAMPLE_PROFILE),
          responderModel: "gemini-3.8-flash",
          judgeModel: "gemini-2.5-pro",
          judgeTier: "frontier",
          questions: threeQuestions
        },
        (question) => {
          executedIds.push(question.question_id);
          if (question.question_id === "q_single_1") {
            return Promise.reject(new Error("Item 1 upstream error"));
          }
          return Promise.resolve(makeItemResult({ questionId: question.question_id }));
        },
        () => {
          finalizeCalls += 1;
          return Promise.resolve(makeManifest([]));
        }
      )
    ).rejects.toThrow("Item 1 upstream error");

    expect(taskAttemptedIds).toEqual(["q_single_1", "q_unans_1", "q_single_3"]);
    expect(executedIds).toEqual(["q_single_1"]);
    expect(finalizeCalls).toBe(0);
    expect(runEvaluatorResults).toEqual([]);
  });

  it("accumulates live per-bucket spend and syncs manifest spend before runEvaluators", async () => {
    const snapshotsAfterItems: Array<{ embedding: number; responder: number; judge: number }> = [];
    let metadataSeenByRunEvaluator: { embedding: number; responder: number; judge: number } | null =
      null;
    const runEvaluations: LangfuseEvaluation[] = [];

    const client: LangfuseExperimentClient = {
      experiment: {
        async run(config: LangfuseExperimentRunConfig): Promise<unknown> {
          for (const item of config.data) {
            await config.task(item);
            snapshotsAfterItems.push({ ...config.metadata.spendByBucketUsd });
          }
          for (const runEvaluator of config.runEvaluators ?? []) {
            const produced = await runEvaluator({ itemResults: [] });
            metadataSeenByRunEvaluator = { ...config.metadata.spendByBucketUsd };
            runEvaluations.push(...(Array.isArray(produced) ? produced : [produced]));
          }
          return { runName: config.runName };
        }
      },
      flush: () => Promise.resolve(),
      shutdown: () => Promise.resolve()
    };

    const exporter = new LangfuseExperimentExporter({
      env: {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      clientFactory: () => client
    });

    const manifest = await exporter.runExperiment(
      {
        runId: "live-spend-sync-run",
        timestamp: "2026-11-26T01:20:00.000Z",
        gitSha: "cafe123",
        split: "dev",
        splitSeed: 20261125,
        profile: SAMPLE_PROFILE,
        indexConfigHash: computeIndexConfigHash(SAMPLE_PROFILE),
        responderModel: "gemini-3.8-flash",
        judgeModel: "gemini-2.5-pro",
        judgeTier: "frontier",
        questions: SAMPLE_QUESTIONS
      },
      (question) =>
        Promise.resolve(
          makeItemResult({
            questionId: question.question_id,
            responderCostUsd: 0.01,
            judgeCostUsd: 0.025
          })
        ),
      (items) => Promise.resolve(makeManifest(items))
    );

    expect(snapshotsAfterItems).toEqual([
      { embedding: 0, responder: 0.01, judge: 0.025 },
      { embedding: 0, responder: 0.02, judge: 0.05 }
    ]);
    expect(metadataSeenByRunEvaluator).toEqual(manifest.spendByBucketUsd);

    const runScoresByName = new Map(runEvaluations.map((score) => [score.name, score.value]));
    expect(runScoresByName.get("spend_embedding_usd")).toBe(0.01);
    expect(runScoresByName.get("spend_responder_usd")).toBe(0.02);
    expect(runScoresByName.get("spend_judge_usd")).toBe(0.05);
    expect(runScoresByName.get("spend_total_usd")).toBe(0.08);
  });
});
