import type { DevQuestion, QuestionStratum } from "../models/eval.js";
import type {
  EvalItemResult,
  EvalRunManifest,
  EvalSplitName,
  JudgeTier,
  SpendByBucket
} from "../models/eval-harness.js";
import type { RunProfile } from "../models/profile.js";
import {
  formatW3cTraceparent,
  parseLangfuseEnv,
  type LangfuseEnvConfig,
  type OptionalModuleLoader,
  type TelemetryController
} from "./telemetry.js";
import type {
  ExperimentItemExecutionContext,
  ExperimentRunContext,
  ExperimentRunnerWrapper,
  ObservabilityExporter
} from "./terminal-scorecard.js";

export { formatW3cTraceparent };

export interface LangfuseEvaluation {
  readonly name: string;
  readonly value: number | string;
  readonly dataType?: "NUMERIC" | "BOOLEAN" | "CATEGORICAL";
  readonly comment?: string;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface LangfuseExperimentItemInput {
  readonly questionId: string;
  readonly question: string;
}

export interface LangfuseExperimentItemExpectedOutput {
  readonly goldAnswer?: string;
  readonly answerFacts?: readonly string[];
  readonly expectedDocIds: readonly string[];
}

export interface LangfuseExperimentItemMetadata {
  readonly questionId: string;
  readonly stratum: QuestionStratum;
  readonly sourceTypes?: readonly string[];
}

export interface LangfuseExperimentItem {
  readonly id: string;
  readonly input: LangfuseExperimentItemInput;
  readonly expectedOutput: LangfuseExperimentItemExpectedOutput;
  readonly metadata: LangfuseExperimentItemMetadata;
}

export interface LangfuseExperimentMetadata {
  readonly profileName: string;
  readonly profile: RunProfile;
  readonly indexConfigHash: string;
  readonly gitSha: string;
  readonly split: EvalSplitName;
  readonly splitSeed: number;
  readonly responderModel: string;
  readonly judgeModel: string;
  readonly judgeTier: JudgeTier;
  readonly spendByBucketUsd: SpendByBucket;
}

export interface LangfuseExperimentPayload {
  readonly name: string;
  readonly runName: string;
  readonly description: string;
  readonly metadata: LangfuseExperimentMetadata;
  readonly data: readonly LangfuseExperimentItem[];
}

export interface LangfuseExperimentTaskOutput {
  readonly questionId: string;
  readonly answer: string;
  readonly documentIds: readonly string[];
  readonly abstained: boolean;
  readonly compositeScore: number;
  readonly latencyMs: number;
}

export interface LangfuseEvaluatorParams {
  readonly input: LangfuseExperimentItemInput;
  readonly output: unknown;
  readonly expectedOutput?: LangfuseExperimentItemExpectedOutput;
  readonly metadata?: LangfuseExperimentItemMetadata;
}

export interface LangfuseRunEvaluatorParams {
  readonly itemResults: readonly {
    readonly item: LangfuseExperimentItem;
    readonly output: unknown;
    readonly evaluations: readonly LangfuseEvaluation[];
  }[];
}

export interface LangfuseExperimentRunConfig {
  readonly name: string;
  readonly runName: string;
  readonly description: string;
  readonly metadata: LangfuseExperimentMetadata;
  readonly data: readonly LangfuseExperimentItem[];
  readonly maxConcurrency?: number;
  readonly task: (item: LangfuseExperimentItem) => Promise<LangfuseExperimentTaskOutput>;
  readonly evaluators?: readonly ((
    params: LangfuseEvaluatorParams
  ) => Promise<LangfuseEvaluation[] | LangfuseEvaluation>)[];
  readonly runEvaluators?: readonly ((
    params: LangfuseRunEvaluatorParams
  ) => Promise<LangfuseEvaluation[] | LangfuseEvaluation>)[];
}

export interface LangfuseExperimentClient {
  readonly experiment: {
    run(config: LangfuseExperimentRunConfig): Promise<unknown>;
  };
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}

export type LangfuseClientFactory = (
  config: LangfuseEnvConfig
) => Promise<LangfuseExperimentClient> | LangfuseExperimentClient;

export interface LangfuseExperimentExporterOptions {
  readonly env?: NodeJS.ProcessEnv;
  readonly telemetry?: TelemetryController;
  readonly clientFactory?: LangfuseClientFactory;
  readonly loadModule?: OptionalModuleLoader;
  readonly onWarning?: (message: string) => void;
}

interface LangfuseClientModule {
  readonly LangfuseClient: new (params: {
    publicKey: string;
    secretKey: string;
    baseUrl: string;
  }) => LangfuseExperimentClient;
}

const ORDERED_STRATA: readonly QuestionStratum[] = [
  "unanswerable",
  "single_doc",
  "two_doc",
  "multi_doc"
];

function inferStratumFromDocCount(expectedDocCount: number): QuestionStratum {
  if (expectedDocCount === 0) {
    return "unanswerable";
  }
  if (expectedDocCount === 1) {
    return "single_doc";
  }
  if (expectedDocCount === 2) {
    return "two_doc";
  }
  return "multi_doc";
}

function defaultLoadModule(specifier: string): Promise<unknown> {
  return import(specifier);
}

function defaultWriteWarning(message: string): void {
  process.stderr.write(`${message}\n`);
}

export function buildItemScores(item: EvalItemResult): LangfuseEvaluation[] {
  const isFalseAbstention = item.stratum !== "unanswerable" && item.abstained;
  return [
    {
      name: "composite_score",
      value: item.compositeScore,
      dataType: "NUMERIC",
      comment: item.verdict.reasoning
    },
    {
      name: "answer_score",
      value: item.effectiveAnswerScore,
      dataType: "NUMERIC",
      metadata: {
        rawAnswerScore: item.verdict.answerScore,
        hasFabricatedOrContradictingClaim: item.verdict.hasFabricatedOrContradictingClaim
      }
    },
    {
      name: "citation_score",
      value: item.citationScore.f1,
      dataType: "NUMERIC",
      metadata: {
        precision: item.citationScore.precision,
        recall: item.citationScore.recall
      }
    },
    {
      name: "abstained",
      value: item.abstained ? 1 : 0,
      dataType: "BOOLEAN"
    },
    {
      name: "false_abstention",
      value: isFalseAbstention ? 1 : 0,
      dataType: "BOOLEAN"
    },
    {
      name: "invalid_citations",
      value: item.citationScore.invalidCount,
      dataType: "NUMERIC"
    }
  ];
}

export function buildRunScores(manifest: EvalRunManifest): LangfuseEvaluation[] {
  const { metrics } = manifest;
  const scores: LangfuseEvaluation[] = [
    {
      name: "composite_score",
      value: metrics.compositeScore,
      dataType: "NUMERIC"
    },
    {
      name: "answer_score",
      value: metrics.answerScore,
      dataType: "NUMERIC"
    },
    {
      name: "citation_score",
      value: metrics.citationScore,
      dataType: "NUMERIC"
    },
    {
      name: "abstention_recall",
      value: metrics.abstentionRecall,
      dataType: "NUMERIC"
    },
    {
      name: "false_abstention_rate",
      value: metrics.falseAbstentionRate,
      dataType: "NUMERIC"
    },
    {
      name: "invalid_citations_total",
      value: metrics.invalidCitationsTotal,
      dataType: "NUMERIC"
    },
    {
      name: "latency_p50_ms",
      value: metrics.latencyP50Ms,
      dataType: "NUMERIC"
    },
    {
      name: "latency_p95_ms",
      value: metrics.latencyP95Ms,
      dataType: "NUMERIC"
    },
    {
      name: "spend_embedding_usd",
      value: manifest.spendByBucketUsd.embedding,
      dataType: "NUMERIC"
    },
    {
      name: "spend_responder_usd",
      value: manifest.spendByBucketUsd.responder,
      dataType: "NUMERIC"
    },
    {
      name: "spend_judge_usd",
      value: manifest.spendByBucketUsd.judge,
      dataType: "NUMERIC"
    },
    {
      name: "spend_total_usd",
      value: manifest.totalSpendUsd,
      dataType: "NUMERIC"
    }
  ];

  for (const stratum of ORDERED_STRATA) {
    const row = metrics.byStratum[stratum];
    scores.push(
      {
        name: `stratum_${stratum}_composite`,
        value: row.compositeScore,
        dataType: "NUMERIC",
        metadata: { count: row.count }
      },
      {
        name: `stratum_${stratum}_answer`,
        value: row.answerScore,
        dataType: "NUMERIC",
        metadata: { count: row.count }
      },
      {
        name: `stratum_${stratum}_citation`,
        value: row.citationScore,
        dataType: "NUMERIC",
        metadata: { count: row.count }
      }
    );
  }

  return scores;
}

export type LangfuseExperimentPayloadInput =
  | { readonly kind: "manifest"; readonly manifest: EvalRunManifest }
  | {
      readonly kind: "context";
      readonly context: ExperimentRunContext;
      readonly spendByBucketUsd?: SpendByBucket;
    };

export function buildLangfuseExperimentPayload(
  input: LangfuseExperimentPayloadInput
): LangfuseExperimentPayload {
  if (input.kind === "manifest") {
    const { manifest } = input;
    return {
      name: `${manifest.split}:${manifest.profile.name}`,
      runName: manifest.runId,
      description: `Eval split=${manifest.split} profile=${manifest.profile.name} responder=${manifest.responderModel} judge=${manifest.judgeModel}`,
      metadata: {
        profileName: manifest.profile.name,
        profile: manifest.profile,
        indexConfigHash: manifest.indexConfigHash,
        gitSha: manifest.gitSha,
        split: manifest.split,
        splitSeed: manifest.splitSeed,
        responderModel: manifest.responderModel,
        judgeModel: manifest.judgeModel,
        judgeTier: manifest.judgeTier,
        spendByBucketUsd: { ...manifest.spendByBucketUsd }
      },
      data: manifest.items.map((item) => ({
        id: item.questionId,
        input: {
          questionId: item.questionId,
          question: item.question
        },
        expectedOutput: {
          expectedDocIds: [...item.expectedDocumentIds]
        },
        metadata: {
          questionId: item.questionId,
          stratum: item.stratum
        }
      }))
    };
  }

  const { context } = input;
  const spendByBucketUsd: SpendByBucket = input.spendByBucketUsd ?? {
    embedding: 0,
    responder: 0,
    judge: 0
  };
  return {
    name: `${context.split}:${context.profile.name}`,
    runName: context.runId,
    description: `Eval split=${context.split} profile=${context.profile.name} responder=${context.responderModel} judge=${context.judgeModel}`,
    metadata: {
      profileName: context.profile.name,
      profile: context.profile,
      indexConfigHash: context.indexConfigHash,
      gitSha: context.gitSha,
      split: context.split,
      splitSeed: context.splitSeed,
      responderModel: context.responderModel,
      judgeModel: context.judgeModel,
      judgeTier: context.judgeTier,
      spendByBucketUsd: { ...spendByBucketUsd }
    },
    data: context.questions.map((question) => ({
      id: question.question_id,
      input: {
        questionId: question.question_id,
        question: question.question
      },
      expectedOutput: {
        goldAnswer: question.gold_answer,
        answerFacts: [...question.answer_facts],
        expectedDocIds: [...question.expected_doc_ids]
      },
      metadata: {
        questionId: question.question_id,
        stratum: inferStratumFromDocCount(question.expected_doc_ids.length),
        sourceTypes: [...question.source_types]
      }
    }))
  };
}

function toTaskOutput(item: EvalItemResult): LangfuseExperimentTaskOutput {
  return {
    questionId: item.questionId,
    answer: item.answer,
    documentIds: [...item.returnedDocumentIds],
    abstained: item.abstained,
    compositeScore: item.compositeScore,
    latencyMs: item.latencyMs
  };
}

export class LangfuseExperimentExporter
  implements ObservabilityExporter, ExperimentRunnerWrapper
{
  public readonly enabled: boolean;
  private readonly config: LangfuseEnvConfig | null;
  private readonly telemetry: TelemetryController | undefined;
  private readonly clientFactory: LangfuseClientFactory | undefined;
  private readonly loadModule: OptionalModuleLoader;
  private readonly onWarning: (message: string) => void;
  private client: LangfuseExperimentClient | null = null;
  private clientLoadAttempted = false;
  private readonly exportedRunIds = new Set<string>();

  public constructor(options?: LangfuseExperimentExporterOptions) {
    this.config = parseLangfuseEnv(options?.env ?? process.env);
    this.enabled = this.config !== null;
    this.telemetry = options?.telemetry;
    this.clientFactory = options?.clientFactory;
    this.loadModule = options?.loadModule ?? defaultLoadModule;
    this.onWarning = options?.onWarning ?? defaultWriteWarning;
  }

  private async resolveClient(): Promise<LangfuseExperimentClient | null> {
    if (this.config === null) {
      return null;
    }
    if (this.client !== null) {
      return this.client;
    }
    if (this.clientLoadAttempted) {
      return null;
    }
    this.clientLoadAttempted = true;

    try {
      if (this.clientFactory !== undefined) {
        this.client = await this.clientFactory(this.config);
        return this.client;
      }
      const loaded = (await this.loadModule("@langfuse/client")) as LangfuseClientModule;
      this.client = new loaded.LangfuseClient({
        publicKey: this.config.publicKey,
        secretKey: this.config.secretKey,
        baseUrl: this.config.baseUrl
      });
      return this.client;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      this.onWarning(
        `[langfuse] Langfuse keys are set, but @langfuse/client is unavailable (${detail}). Skipping Langfuse experiment export.`
      );
      return null;
    }
  }

  public async runExperiment(
    context: ExperimentRunContext,
    executeItem: (
      question: DevQuestion,
      itemContext: ExperimentItemExecutionContext
    ) => Promise<EvalItemResult>,
    finalizeManifest: (items: readonly EvalItemResult[]) => Promise<EvalRunManifest>
  ): Promise<EvalRunManifest> {
    const maxConcurrency =
      context.concurrency !== undefined &&
      Number.isInteger(context.concurrency) &&
      context.concurrency > 0
        ? context.concurrency
        : 1;

    const client = await this.resolveClient();
    if (client === null) {
      const ordered = Array.from<EvalItemResult | undefined>({
        length: context.questions.length
      });
      let nextIndex = 0;
      let firstError: unknown = null;
      const runWorker = async (): Promise<void> => {
        for (;;) {
          if (firstError !== null) {
            return;
          }
          const idx = nextIndex;
          if (idx >= context.questions.length) {
            return;
          }
          nextIndex += 1;
          const question = context.questions[idx];
          if (question === undefined) {
            return;
          }
          try {
            ordered[idx] = await executeItem(question, { headers: {} });
          } catch (error) {
            if (firstError === null) {
              firstError = error;
            }
            return;
          }
        }
      };
      const workerCount = Math.min(maxConcurrency, context.questions.length);
      await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
      if (firstError !== null) {
        throw firstError;
      }
      const items = ordered.filter((item): item is EvalItemResult => item !== undefined);
      return finalizeManifest(items);
    }

    const questionById = new Map<string, DevQuestion>();
    for (const question of context.questions) {
      questionById.set(question.question_id, question);
    }

    const payload = buildLangfuseExperimentPayload({ kind: "context", context });
    const resultsById = new Map<string, EvalItemResult>();
    let finalizedManifest: EvalRunManifest | null = null;
    let taskFailure: unknown = null;

    const ensureFinalized = async (): Promise<EvalRunManifest> => {
      if (finalizedManifest !== null) {
        return finalizedManifest;
      }
      const orderedItems: EvalItemResult[] = [];
      for (const question of context.questions) {
        const result = resultsById.get(question.question_id);
        if (result !== undefined) {
          orderedItems.push(result);
        }
      }
      this.exportedRunIds.add(context.runId);
      finalizedManifest = await finalizeManifest(orderedItems);
      payload.metadata.spendByBucketUsd.embedding =
        finalizedManifest.spendByBucketUsd.embedding;
      payload.metadata.spendByBucketUsd.responder =
        finalizedManifest.spendByBucketUsd.responder;
      payload.metadata.spendByBucketUsd.judge =
        finalizedManifest.spendByBucketUsd.judge;
      return finalizedManifest;
    };

    await client.experiment.run({
      name: payload.name,
      runName: payload.runName,
      description: payload.description,
      metadata: payload.metadata,
      data: payload.data,
      maxConcurrency,
      task: async (item) => {
        if (taskFailure !== null) {
          throw taskFailure;
        }
        const question = questionById.get(item.id);
        if (question === undefined) {
          const error = new Error(`Unknown experiment item id: ${item.id}`);
          taskFailure = error;
          throw error;
        }
        const headers = this.telemetry?.injectTraceHeaders() ?? {};
        try {
          const result = await executeItem(question, { headers });
          resultsById.set(question.question_id, result);
          payload.metadata.spendByBucketUsd.responder += result.responderCostUsd;
          payload.metadata.spendByBucketUsd.judge += result.judgeCostUsd;
          return toTaskOutput(result);
        } catch (error) {
          if (taskFailure === null) {
            taskFailure = error;
          }
          throw error;
        }
      },
      evaluators: [
        (params) => {
          const result = resultsById.get(params.input.questionId);
          if (result === undefined) {
            return Promise.resolve([]);
          }
          return Promise.resolve(buildItemScores(result));
        }
      ],
      runEvaluators: [
        async () => {
          if (taskFailure !== null) {
            return [];
          }
          const manifest = await ensureFinalized();
          return buildRunScores(manifest);
        }
      ]
    });

    if (taskFailure !== null) {
      throw taskFailure;
    }

    const manifest = await ensureFinalized();
    await this.flush();
    return manifest;
  }

  public async onRunComplete(manifest: EvalRunManifest): Promise<void> {
    if (!this.enabled) {
      return;
    }
    if (this.exportedRunIds.has(manifest.runId)) {
      await this.flush();
      return;
    }

    const client = await this.resolveClient();
    if (client === null) {
      return;
    }

    const itemsById = new Map<string, EvalItemResult>();
    for (const item of manifest.items) {
      itemsById.set(item.questionId, item);
    }

    const payload = buildLangfuseExperimentPayload({ kind: "manifest", manifest });
    this.exportedRunIds.add(manifest.runId);

    await client.experiment.run({
      name: payload.name,
      runName: payload.runName,
      description: payload.description,
      metadata: payload.metadata,
      data: payload.data,
      maxConcurrency: 1,
      task: (item) => {
        const result = itemsById.get(item.id);
        if (result === undefined) {
          return Promise.reject(new Error(`Missing manifest item for id: ${item.id}`));
        }
        return Promise.resolve(toTaskOutput(result));
      },
      evaluators: [
        (params) => {
          const result = itemsById.get(params.input.questionId);
          if (result === undefined) {
            return Promise.resolve([]);
          }
          return Promise.resolve(buildItemScores(result));
        }
      ],
      runEvaluators: [() => Promise.resolve(buildRunScores(manifest))]
    });

    await this.flush();
  }

  public async flush(): Promise<void> {
    if (this.client !== null) {
      await this.client.flush();
    }
    if (this.telemetry !== undefined) {
      await this.telemetry.flush();
    }
  }

  public async shutdown(): Promise<void> {
    if (this.client !== null) {
      await this.client.shutdown();
    }
  }
}
