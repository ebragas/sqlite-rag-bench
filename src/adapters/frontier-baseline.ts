import { performance } from "node:perf_hooks";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { APICallError, generateText, Output, type LanguageModel } from "ai";
import { ConfigError, ContractError, ValidationError } from "../models/errors.js";
import { AskResponseSchema, type AskResponse } from "../models/eval-harness.js";
import { detectJudgeProvider, type JudgeProvider } from "./judge.js";
import {
  MAX_ASK_RESPONSE_BYTES,
  type ResponderAdapter,
  type ResponderCallOptions,
  type ResponderResult,
  type TokenUsage
} from "./responder.js";

export const DEFAULT_FRONTIER_BASELINE_MODEL = "gemini-2.5-pro";

const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_BASE_DELAY_MS = 500;
const DEFAULT_MAX_DELAY_MS = 4_000;

export interface FrontierGenerateInput {
  readonly modelId: string;
  readonly system: string;
  readonly prompt: string;
  readonly temperature: number;
  readonly signal?: AbortSignal;
}

export interface FrontierGenerateOutput {
  readonly output: AskResponse;
  readonly usage: TokenUsage;
}

export type FrontierGenerateFn = (
  input: FrontierGenerateInput
) => Promise<FrontierGenerateOutput>;

export type RetrySleepFn = (delayMs: number, signal?: AbortSignal) => Promise<void>;

export interface FrontierOneShotResponderOptions {
  readonly modelId: string;
  readonly provider?: JudgeProvider;
  readonly apiKey?: string;
  readonly model?: LanguageModel;
  readonly contextPrompt?: string;
  readonly corpusSummary?: string;
  readonly maxAttempts?: number;
  readonly baseDelayMs?: number;
  readonly maxDelayMs?: number;
  readonly generateFn?: FrontierGenerateFn;
  readonly sleepFn?: RetrySleepFn;
}

const BASELINE_SYSTEM_PROMPT = [
  "You are a Frontier Tier one-shot baseline Responder for an enterprise knowledge evaluation.",
  "Return a structured JSON object with fields answer and document_ids.",
  "Rules:",
  "1. Answer directly and concisely from the provided context only.",
  "2. If no corpus context is provided or the context lacks enough evidence to answer, state clearly that the corpus lacks enough evidence and return document_ids as [].",
  "3. Do not invent unsupported facts or fabricate document_ids."
].join("\n");

function stripProviderPrefix(modelId: string, provider: JudgeProvider): string {
  const prefix = `${provider}/`;
  if (modelId.toLowerCase().startsWith(prefix)) {
    return modelId.slice(prefix.length);
  }
  return modelId;
}

function resolveBaselineLanguageModel(options: FrontierOneShotResponderOptions): LanguageModel {
  if (options.model !== undefined) {
    return options.model;
  }

  const provider = options.provider ?? detectJudgeProvider(options.modelId);
  const rawModelName = stripProviderPrefix(options.modelId.trim(), provider);

  if (provider === "openai") {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new ConfigError(
        `OPENAI_API_KEY is required for OpenAI baseline model "${options.modelId}".`
      );
    }
    const openai = createOpenAI({ apiKey });
    return openai(rawModelName);
  }

  const apiKey =
    options.apiKey ?? process.env.GEMINI_API_KEY ?? process.env.GOOGLE_GENERATIVE_AI_API_KEY;
  if (!apiKey) {
    throw new ConfigError(
      `GEMINI_API_KEY is required for Google baseline model "${options.modelId}".`
    );
  }
  const google = createGoogleGenerativeAI({ apiKey });
  return google(rawModelName);
}

function buildBaselineSystemPrompt(
  corpusSummary: string | undefined,
  contextPrompt: string | undefined
): string {
  const sections = [BASELINE_SYSTEM_PROMPT];
  if (corpusSummary !== undefined && corpusSummary.trim().length > 0) {
    sections.push(`Corpus summary:\n${corpusSummary.trim()}`);
  }
  if (contextPrompt !== undefined && contextPrompt.trim().length > 0) {
    sections.push(`Additional context:\n${contextPrompt.trim()}`);
  }
  return sections.join("\n\n");
}

function buildBaselineUserPrompt(question: string): string {
  return `Question:\n${question}`;
}

function isTransientError(error: unknown): boolean {
  if (APICallError.isInstance(error)) {
    if (error.statusCode === 429) {
      return true;
    }
    if (error.statusCode !== undefined && error.statusCode >= 500) {
      return true;
    }
    return error.isRetryable;
  }

  if (typeof error === "object" && error !== null && "statusCode" in error) {
    const statusCode = (error as { readonly statusCode?: unknown }).statusCode;
    if (typeof statusCode === "number" && (statusCode === 429 || statusCode >= 500)) {
      return true;
    }
  }

  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes("fetch failed") ||
      message.includes("econnreset") ||
      message.includes("etimedout") ||
      message.includes("socket hang up")
    );
  }

  return false;
}

function defaultSleepWithAbort(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.reject(
      new ContractError("Frontier baseline request aborted during retry backoff", {
        cause: signal.reason
      })
    );
  }

  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(
        new ContractError("Frontier baseline request aborted during retry backoff", {
          cause: signal?.reason
        })
      );
    };

    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);

    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function createAiSdkGenerateFn(languageModel: LanguageModel): FrontierGenerateFn {
  return async (input: FrontierGenerateInput): Promise<FrontierGenerateOutput> => {
    const result = await generateText({
      model: languageModel,
      system: input.system,
      prompt: input.prompt,
      temperature: input.temperature,
      maxRetries: 0,
      ...(input.signal !== undefined ? { abortSignal: input.signal } : {}),
      output: Output.object({
        schema: AskResponseSchema,
        name: "frontier_oneshot_answer",
        description: "Structured one-shot baseline response with answer and document_ids."
      })
    });

    return {
      output: result.output,
      usage: {
        inputTokens: result.usage.inputTokens ?? 0,
        outputTokens: result.usage.outputTokens ?? 0
      }
    };
  };
}

export class FrontierOneShotResponder implements ResponderAdapter {
  public readonly modelId: string;
  private readonly systemPrompt: string;
  private readonly maxAttempts: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly generateFn: FrontierGenerateFn;
  private readonly sleepFn: RetrySleepFn;

  public constructor(options: FrontierOneShotResponderOptions) {
    const trimmedModel = options.modelId.trim();
    if (trimmedModel.length === 0) {
      throw new ConfigError("Frontier baseline modelId must be non-empty.");
    }
    this.modelId = trimmedModel;

    const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
    if (!Number.isInteger(maxAttempts) || maxAttempts <= 0) {
      throw new ConfigError(`maxAttempts must be a positive integer, got ${maxAttempts}.`);
    }
    this.maxAttempts = maxAttempts;

    const baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
    const maxDelayMs = options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
    if (!Number.isFinite(baseDelayMs) || baseDelayMs < 0) {
      throw new ConfigError(`baseDelayMs must be non-negative, got ${baseDelayMs}.`);
    }
    if (!Number.isFinite(maxDelayMs) || maxDelayMs < 0) {
      throw new ConfigError(`maxDelayMs must be non-negative, got ${maxDelayMs}.`);
    }
    this.baseDelayMs = baseDelayMs;
    this.maxDelayMs = maxDelayMs;

    this.systemPrompt = buildBaselineSystemPrompt(options.corpusSummary, options.contextPrompt);
    this.sleepFn = options.sleepFn ?? defaultSleepWithAbort;

    if (options.generateFn !== undefined) {
      this.generateFn = options.generateFn;
    } else {
      const languageModel = resolveBaselineLanguageModel(options);
      this.generateFn = createAiSdkGenerateFn(languageModel);
    }
  }

  public async answer(question: string, options?: ResponderCallOptions): Promise<ResponderResult> {
    const startedAt = performance.now();
    const signal = options?.signal;
    const prompt = buildBaselineUserPrompt(question);

    let lastError: unknown;

    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      if (signal?.aborted) {
        throw new ContractError(`Frontier baseline request aborted for model "${this.modelId}".`, {
          cause: signal.reason
        });
      }

      try {
        const generated = await this.generateFn({
          modelId: this.modelId,
          system: this.systemPrompt,
          prompt,
          temperature: 0,
          ...(signal !== undefined ? { signal } : {})
        });

        const validated = AskResponseSchema.safeParse(generated.output);
        if (!validated.success) {
          throw new ValidationError(
            `Frontier baseline response failed schema check: ${validated.error.message}`
          );
        }

        const serialized = JSON.stringify(validated.data);
        const responseBytes = Buffer.byteLength(serialized, "utf8");
        if (responseBytes > MAX_ASK_RESPONSE_BYTES) {
          throw new ContractError(
            `Frontier baseline response size ${responseBytes} bytes exceeds limit ${MAX_ASK_RESPONSE_BYTES}`
          );
        }

        const latencyMs = Math.max(0, performance.now() - startedAt);
        return {
          answer: validated.data.answer,
          documentIds: validated.data.document_ids,
          latencyMs,
          responseBytes,
          usage: {
            inputTokens: generated.usage.inputTokens,
            outputTokens: generated.usage.outputTokens
          }
        };
      } catch (error) {
        if (error instanceof ValidationError) {
          throw error;
        }
        lastError = error;
        if (signal?.aborted) {
          throw new ContractError(`Frontier baseline request aborted for model "${this.modelId}".`, {
            cause: error
          });
        }

        const canRetry = attempt < this.maxAttempts && isTransientError(error);
        if (!canRetry) {
          if (error instanceof ContractError) {
            throw error;
          }
          throw new ContractError(
            `Frontier baseline generation failed on model "${this.modelId}".`,
            { cause: error }
          );
        }

        const backoffMs = Math.min(this.maxDelayMs, this.baseDelayMs * 2 ** (attempt - 1));
        await this.sleepFn(backoffMs, signal);
      }
    }

    throw new ContractError(
      `Frontier baseline exhausted ${this.maxAttempts} attempts on model "${this.modelId}".`,
      { cause: lastError }
    );
  }
}
