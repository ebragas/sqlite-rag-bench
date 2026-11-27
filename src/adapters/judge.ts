import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, Output, type LanguageModel } from "ai";
import { ConfigError } from "../models/errors.js";
import {
  JudgeVerdictSchema,
  type JudgeTier,
  type JudgeVerdict
} from "../models/eval-harness.js";
import type { TokenUsage } from "./responder.js";
import type { TelemetryProvider } from "./telemetry.js";

export interface JudgeInput {
  readonly question: string;
  readonly goldAnswer: string;
  readonly answerFacts: readonly string[];
  readonly expectedDocIds: readonly string[];
  readonly responderAnswer: string;
  readonly responderDocIds: readonly string[];
}

export interface JudgeEvaluation {
  readonly verdict: JudgeVerdict;
  readonly usage: TokenUsage;
}

export interface JudgeAdapter {
  readonly modelId: string;
  readonly tier: JudgeTier;
  evaluate(input: JudgeInput): Promise<JudgeEvaluation>;
}

export type JudgeProvider = "google" | "openai";

export interface AiSdkJudgeOptions {
  readonly modelId: string;
  readonly tier: JudgeTier;
  readonly provider?: JudgeProvider;
  readonly apiKey?: string;
  readonly model?: LanguageModel;
  readonly telemetry?: TelemetryProvider;
}

const JUDGE_SYSTEM_PROMPT = [
  "You are a strict evaluation Judge for an enterprise knowledge system.",
  "Score the Responder answer against the gold answer, required answer_facts, and expected_doc_ids.",
  "Rules:",
  "1. Set hasFabricatedOrContradictingClaim to true if the Responder states any claim that contradicts the gold answer or answer_facts, or invents an unsupported answer to the question. Do not flag benign extra technical context that is consistent with the gold answer and plausibly comes from the cited Corpus Document.",
  "2. Set abstained to true if the Responder explicitly states that the corpus does not contain enough evidence to answer the question.",
  "3. For answerable questions (expected_doc_ids is non-empty), set answerScore between 0 and 1 based on the fraction of required answer_facts accurately covered.",
  "4. For unanswerable questions (expected_doc_ids is empty), set answerScore to 1 if the Responder abstains cleanly without fabricating facts, otherwise set answerScore to 0 and hasFabricatedOrContradictingClaim to true if it guesses an answer.",
  "5. Provide concise reasoning explaining covered facts, missing facts, and any contradiction."
].join("\n");

export function detectJudgeProvider(modelId: string): JudgeProvider {
  const normalized = modelId.trim().toLowerCase();
  if (
    normalized.startsWith("openai/") ||
    normalized.startsWith("gpt-") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("o4")
  ) {
    return "openai";
  }
  return "google";
}

function stripProviderPrefix(modelId: string, provider: JudgeProvider): string {
  const prefix = `${provider}/`;
  if (modelId.toLowerCase().startsWith(prefix)) {
    return modelId.slice(prefix.length);
  }
  return modelId;
}

function resolveLanguageModel(options: AiSdkJudgeOptions): LanguageModel {
  if (options.model !== undefined) {
    return options.model;
  }

  const provider = options.provider ?? detectJudgeProvider(options.modelId);
  const rawModelName = stripProviderPrefix(options.modelId.trim(), provider);

  if (provider === "openai") {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new ConfigError(`OPENAI_API_KEY is required for OpenAI Judge model "${options.modelId}".`);
    }
    const openai = createOpenAI({ apiKey });
    return openai(rawModelName);
  }

  const apiKey =
    options.apiKey ?? process.env.GEMINI_API_KEY ?? process.env.GOOGLE_GENERATIVE_AI_API_KEY;
  if (!apiKey) {
    throw new ConfigError(`GEMINI_API_KEY is required for Google Judge model "${options.modelId}".`);
  }
  const google = createGoogleGenerativeAI({ apiKey });
  return google(rawModelName);
}

export function buildJudgePrompt(input: JudgeInput): string {
  const unanswerable = input.expectedDocIds.length === 0;
  const factsText =
    input.answerFacts.length === 0
      ? "(none)"
      : input.answerFacts.map((fact, index) => `${index + 1}. ${fact}`).join("\n");

  return [
    `Question: ${input.question}`,
    `Question type: ${unanswerable ? "unanswerable" : "answerable"}`,
    `Expected document count: ${input.expectedDocIds.length}`,
    `Gold answer:\n${input.goldAnswer}`,
    `Required answer_facts:\n${factsText}`,
    `Responder answer:\n${input.responderAnswer}`,
    `Responder cited document_ids: ${JSON.stringify(input.responderDocIds)}`
  ].join("\n\n");
}

export class AiSdkJudgeAdapter implements JudgeAdapter {
  public readonly modelId: string;
  public readonly tier: JudgeTier;
  private readonly languageModel: LanguageModel;
  private readonly telemetry: TelemetryProvider | undefined;

  public constructor(options: AiSdkJudgeOptions) {
    this.modelId = options.modelId;
    this.tier = options.tier;
    this.languageModel = resolveLanguageModel(options);
    this.telemetry = options.telemetry;
  }

  public async evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    const telemetry = this.telemetry?.getAiSdkTelemetry("eval.judge", {
      model: this.modelId,
      tier: this.tier
    });
    const result = await generateText({
      model: this.languageModel,
      system: JUDGE_SYSTEM_PROMPT,
      prompt: buildJudgePrompt(input),
      temperature: 0,
      output: Output.object({
        schema: JudgeVerdictSchema,
        name: "judge_verdict",
        description: "Strict Onyx rubric verdict for a single Responder output."
      }),
      ...(telemetry !== undefined ? { experimental_telemetry: telemetry } : {})
    });

    return {
      verdict: result.output,
      usage: {
        inputTokens: result.usage.inputTokens ?? 0,
        outputTokens: result.usage.outputTokens ?? 0
      }
    };
  }
}
