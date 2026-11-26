import { z } from "zod";
import { ConfigError, ContractError, ValidationError } from "../models/errors.js";
import { JudgeVerdictSchema } from "../models/eval-harness.js";
import type { JudgeAdapter, JudgeEvaluation, JudgeInput } from "./judge.js";

export const DEFAULT_SYSTEMONE_BASE_URL = "http://127.0.0.1:11434";
export const DEFAULT_SYSTEMONE_MODEL = "clef:27b-mxfp8";

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_THRESHOLD = 0.5;
const CONTRADICTION_KEY = "has_fabricated_or_contradicting_claim";
const ABSTAINED_KEY = "abstained";
const GOLD_SCORE_KEY = "gold_answer_score";

export interface SystemOneJudgeOptions {
  readonly modelId: string;
  readonly baseUrl?: string;
  readonly apiKey?: string;
  readonly timeoutMs?: number;
  readonly contradictionThreshold?: number;
  readonly abstentionThreshold?: number;
  readonly factCoverageThreshold?: number;
  readonly fetchFn?: typeof fetch;
}

export interface SystemOneNoulQuestion {
  readonly type: "noul";
  readonly instructions: string;
  readonly criteria: {
    readonly true: string;
    readonly false: string;
  };
}

export interface SystemOneScoreQuestion {
  readonly type: "score";
  readonly instructions: string;
}

export type SystemOneQuestionSpec = SystemOneNoulQuestion | SystemOneScoreQuestion;

export interface SystemOneRequestPayload {
  readonly model: string;
  readonly state: {
    readonly question: string;
    readonly question_type: "answerable" | "unanswerable";
    readonly gold_answer: string;
    readonly answer_facts: readonly string[];
    readonly expected_doc_ids: readonly string[];
    readonly responder_answer: string;
    readonly responder_doc_ids: readonly string[];
  };
  readonly questions: Readonly<Record<string, SystemOneQuestionSpec>>;
}

const UnitProbabilitySchema = z.number().finite().min(0).max(1);

const SystemOneAnswerSchema = z
  .object({
    type: z.enum(["noul", "choice", "score"]).optional(),
    noul: UnitProbabilitySchema.optional(),
    probability: UnitProbabilitySchema.optional(),
    score: UnitProbabilitySchema.optional(),
    decision: z.boolean().optional()
  })
  .refine(
    (entry) =>
      entry.noul !== undefined ||
      entry.probability !== undefined ||
      entry.score !== undefined ||
      entry.decision !== undefined,
    {
      message: "System One answer must include noul, probability, score, or decision."
    }
  );

const SystemOneUsageSchema = z
  .object({
    input_tokens: z.number().int().nonnegative().optional(),
    output_tokens: z.number().int().nonnegative().optional(),
    inputTokens: z.number().int().nonnegative().optional(),
    outputTokens: z.number().int().nonnegative().optional()
  })
  .optional();

export const SystemOneResponseSchema = z.object({
  model: z.string().min(1).optional(),
  answers: z.record(z.string(), SystemOneAnswerSchema),
  usage: SystemOneUsageSchema
});

type SystemOneAnswerEntry = z.infer<typeof SystemOneAnswerSchema>;

function assertUnitThreshold(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new ConfigError(`${name} must be between 0 and 1, got ${value}.`);
  }
  return value;
}

function resolveSystemOneEndpoint(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (trimmed.length === 0) {
    throw new ConfigError("System One baseUrl must be a non-empty URL.");
  }
  if (trimmed.endsWith("/v1/systemone")) {
    return trimmed;
  }
  return `${trimmed}/v1/systemone`;
}

function extractNoulProbability(entry: SystemOneAnswerEntry, key: string): number {
  if (entry.noul !== undefined) {
    return entry.noul;
  }
  if (entry.probability !== undefined) {
    return entry.probability;
  }
  if (entry.decision !== undefined) {
    return entry.decision ? 1 : 0;
  }
  throw new ValidationError(
    `System One noul answer "${key}" lacks noul, probability, or decision.`
  );
}

function extractScoreValue(entry: SystemOneAnswerEntry, key: string): number {
  if (entry.score !== undefined) {
    return entry.score;
  }
  if (entry.probability !== undefined) {
    return entry.probability;
  }
  if (entry.noul !== undefined) {
    return entry.noul;
  }
  if (entry.decision !== undefined) {
    return entry.decision ? 1 : 0;
  }
  throw new ValidationError(`System One score answer "${key}" lacks score or probability.`);
}

function requireAnswerEntry(
  answers: Readonly<Record<string, SystemOneAnswerEntry>>,
  key: string
): SystemOneAnswerEntry {
  const entry = answers[key];
  if (entry === undefined) {
    throw new ValidationError(`System One response omitted required answer "${key}".`);
  }
  return entry;
}

export function buildSystemOneRequest(
  modelId: string,
  input: JudgeInput
): SystemOneRequestPayload {
  const unanswerable = input.expectedDocIds.length === 0;
  const questions: Record<string, SystemOneQuestionSpec> = {
    [CONTRADICTION_KEY]: {
      type: "noul",
      instructions: unanswerable
        ? "Does the responder guess an answer or fabricate specifics instead of abstaining cleanly?"
        : "Does the responder state any claim that contradicts the gold answer or answer_facts, or invent unsupported specifics?",
      criteria: {
        true: unanswerable
          ? "Responder guesses an answer, speculates specifics, or fabricates unsupported facts."
          : "Responder contradicts the gold answer or answer_facts, or invents unsupported specifics.",
        false: unanswerable
          ? "Responder abstains cleanly without guessing or fabricating facts."
          : "All claims in the responder answer match or remain consistent with the gold answer and answer_facts."
      }
    },
    [ABSTAINED_KEY]: {
      type: "noul",
      instructions:
        "Does the responder explicitly state that the corpus lacks enough evidence to answer the question?",
      criteria: {
        true: "Responder clearly states the corpus lacks enough evidence or cannot answer the question.",
        false: "Responder attempts to provide an answer rather than abstaining."
      }
    }
  };

  if (!unanswerable && input.answerFacts.length > 0) {
    for (let index = 0; index < input.answerFacts.length; index += 1) {
      const fact = input.answerFacts[index];
      if (fact === undefined) {
        continue;
      }
      questions[`fact_${index}`] = {
        type: "noul",
        instructions: `Does the responder answer accurately cover this required fact: ${fact}`,
        criteria: {
          true: "The responder answer clearly and accurately states this required fact.",
          false: "The responder answer omits, distorts, or contradicts this required fact."
        }
      };
    }
  } else if (!unanswerable) {
    questions[GOLD_SCORE_KEY] = {
      type: "score",
      instructions:
        "Score how completely and accurately the responder answer matches the gold answer from 0 to 1."
    };
  }

  return {
    model: modelId,
    state: {
      question: input.question,
      question_type: unanswerable ? "unanswerable" : "answerable",
      gold_answer: input.goldAnswer,
      answer_facts: input.answerFacts,
      expected_doc_ids: input.expectedDocIds,
      responder_answer: input.responderAnswer,
      responder_doc_ids: input.responderDocIds
    },
    questions
  };
}

export class SystemOneDecisionJudgeAdapter implements JudgeAdapter {
  public readonly modelId: string;
  public readonly tier = "decision" as const;
  private readonly endpointUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly contradictionThreshold: number;
  private readonly abstentionThreshold: number;
  private readonly factCoverageThreshold: number;
  private readonly fetchFn: typeof fetch;

  public constructor(options: SystemOneJudgeOptions) {
    const trimmedModel = options.modelId.trim();
    if (trimmedModel.length === 0) {
      throw new ConfigError("System One modelId must be non-empty.");
    }
    this.modelId = trimmedModel;

    const rawBaseUrl =
      options.baseUrl ?? process.env.SYSTEMONE_BASE_URL ?? DEFAULT_SYSTEMONE_BASE_URL;
    this.endpointUrl = resolveSystemOneEndpoint(rawBaseUrl);

    const rawApiKey = options.apiKey ?? process.env.SYSTEMONE_API_KEY;
    this.apiKey = rawApiKey !== undefined && rawApiKey.trim().length > 0 ? rawApiKey.trim() : undefined;

    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new ConfigError(`System One timeoutMs must be positive, got ${timeoutMs}.`);
    }
    this.timeoutMs = timeoutMs;

    this.contradictionThreshold = assertUnitThreshold(
      options.contradictionThreshold ?? DEFAULT_THRESHOLD,
      "contradictionThreshold"
    );
    this.abstentionThreshold = assertUnitThreshold(
      options.abstentionThreshold ?? DEFAULT_THRESHOLD,
      "abstentionThreshold"
    );
    this.factCoverageThreshold = assertUnitThreshold(
      options.factCoverageThreshold ?? DEFAULT_THRESHOLD,
      "factCoverageThreshold"
    );
    this.fetchFn = options.fetchFn ?? fetch;
  }

  public async evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    const payload = buildSystemOneRequest(this.modelId, input);
    const headers: Record<string, string> = {
      "content-type": "application/json"
    };
    if (this.apiKey !== undefined) {
      headers.authorization = `Bearer ${this.apiKey}`;
    }

    let response: Response;
    try {
      response = await this.fetchFn(this.endpointUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(this.timeoutMs)
      });
    } catch (error) {
      throw new ContractError(`POST ${this.endpointUrl} failed`, { cause: error });
    }

    const rawBody = await response.text();
    if (!response.ok) {
      throw new ContractError(
        `POST ${this.endpointUrl} returned status ${response.status}: ${rawBody.slice(0, 200)}`
      );
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(rawBody);
    } catch (error) {
      throw new ValidationError(`POST ${this.endpointUrl} returned invalid JSON`, { cause: error });
    }

    const validated = SystemOneResponseSchema.safeParse(parsedJson);
    if (!validated.success) {
      throw new ValidationError(
        `POST ${this.endpointUrl} response failed schema check: ${validated.error.message}`
      );
    }

    const answers = validated.data.answers;
    const contradictionProb = extractNoulProbability(
      requireAnswerEntry(answers, CONTRADICTION_KEY),
      CONTRADICTION_KEY
    );
    const abstainedProb = extractNoulProbability(
      requireAnswerEntry(answers, ABSTAINED_KEY),
      ABSTAINED_KEY
    );

    const hasFabricatedOrContradictingClaim = contradictionProb >= this.contradictionThreshold;
    const abstained = abstainedProb >= this.abstentionThreshold;
    const unanswerable = input.expectedDocIds.length === 0;

    let answerScore: number;
    let reasoning: string;

    if (unanswerable) {
      answerScore = !hasFabricatedOrContradictingClaim && abstained ? 1 : 0;
      reasoning = `System One unanswerable verdict: contradiction_p=${contradictionProb.toFixed(3)}, abstained_p=${abstainedProb.toFixed(3)}, answerScore=${answerScore}.`;
    } else if (input.answerFacts.length > 0) {
      let coveredFacts = 0;
      for (let index = 0; index < input.answerFacts.length; index += 1) {
        const factKey = `fact_${index}`;
        const factProb = extractNoulProbability(requireAnswerEntry(answers, factKey), factKey);
        if (factProb >= this.factCoverageThreshold) {
          coveredFacts += 1;
        }
      }
      const rawCoverage = coveredFacts / input.answerFacts.length;
      answerScore = hasFabricatedOrContradictingClaim ? 0 : rawCoverage;
      reasoning = `System One fact coverage ${coveredFacts}/${input.answerFacts.length}: contradiction_p=${contradictionProb.toFixed(3)}, abstained_p=${abstainedProb.toFixed(3)}, answerScore=${answerScore.toFixed(3)}.`;
    } else {
      const goldScore = extractScoreValue(
        requireAnswerEntry(answers, GOLD_SCORE_KEY),
        GOLD_SCORE_KEY
      );
      answerScore = hasFabricatedOrContradictingClaim ? 0 : goldScore;
      reasoning = `System One gold score ${goldScore.toFixed(3)}: contradiction_p=${contradictionProb.toFixed(3)}, abstained_p=${abstainedProb.toFixed(3)}, answerScore=${answerScore.toFixed(3)}.`;
    }

    const verdict = JudgeVerdictSchema.parse({
      answerScore,
      hasFabricatedOrContradictingClaim,
      abstained,
      reasoning
    });

    const inputTokens =
      validated.data.usage?.input_tokens ?? validated.data.usage?.inputTokens ?? 0;

    return {
      verdict,
      usage: {
        inputTokens,
        outputTokens: 0
      }
    };
  }
}
