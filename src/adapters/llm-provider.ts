import { createGoogleGenerativeAI } from "@ai-sdk/google";
import {
  APICallError,
  generateText,
  Output,
  tool,
  type ModelMessage,
  type ToolSet
} from "ai";
import type { z } from "zod";
import {
  AgentSynthesisSchema,
  ProviderError,
  type AgentSynthesis
} from "../models/corpus.js";
import type { TelemetryProvider } from "./telemetry.js";

export interface LlmTokenUsage {
  inputTokens: number;
  outputTokens: number;
}

export interface LlmSynthesizeInput {
  model: string;
  system: string;
  prompt: string;
}

export interface LlmSynthesizeOutput {
  result: AgentSynthesis;
  usage: LlmTokenUsage;
  model: string;
  reasoningText?: string;
}

export interface LlmToolDefinition {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: z.ZodType;
}

export interface LlmToolCall {
  readonly toolCallId: string;
  readonly toolName: string;
  readonly input: unknown;
}

export interface LlmToolResult {
  readonly toolCallId: string;
  readonly toolName: string;
  readonly output: unknown;
  readonly isError?: boolean;
}

export type LlmAgentMessage =
  | { readonly role: "user"; readonly content: string }
  | {
      readonly role: "assistant";
      readonly text: string;
      readonly toolCalls: readonly LlmToolCall[];
      readonly rawProviderMessage?: unknown;
    }
  | {
      readonly role: "tool";
      readonly results: readonly LlmToolResult[];
    };

export interface LlmAgentStepInput {
  readonly model: string;
  readonly system: string;
  readonly messages: readonly LlmAgentMessage[];
  readonly tools: readonly LlmToolDefinition[];
}

export interface LlmAgentStepOutput {
  readonly text: string;
  readonly toolCalls: readonly LlmToolCall[];
  readonly usage: LlmTokenUsage;
  readonly model: string;
  readonly rawProviderMessage?: unknown;
  readonly reasoningText?: string;
}

export interface LlmProvider {
  synthesize(
    input: LlmSynthesizeInput,
    signal?: AbortSignal
  ): Promise<LlmSynthesizeOutput>;
  runAgentStep?(
    input: LlmAgentStepInput,
    signal?: AbortSignal
  ): Promise<LlmAgentStepOutput>;
}

export interface GeminiLlmProviderOptions {
  apiKey: string;
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  temperature?: number;
  telemetry?: TelemetryProvider;
}

const DEFAULT_MAX_ATTEMPTS = 3;
const DEFAULT_BASE_DELAY_MS = 500;
const DEFAULT_MAX_DELAY_MS = 4000;
const DEFAULT_TEMPERATURE = 0;

function isTransientProviderError(error: unknown): boolean {
  if (APICallError.isInstance(error)) {
    if (error.statusCode === 429) {
      return true;
    }
    if (error.statusCode !== undefined && error.statusCode >= 500) {
      return true;
    }
    return error.isRetryable;
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

function waitWithAbort(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.reject(
      new ProviderError("LLM request aborted during retry backoff", {
        cause: signal.reason
      })
    );
  }

  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(
        new ProviderError("LLM request aborted during retry backoff", {
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

function isAssistantModelMessage(value: unknown): value is ModelMessage {
  return (
    typeof value === "object" &&
    value !== null &&
    "role" in value &&
    (value as { role: unknown }).role === "assistant" &&
    "content" in value
  );
}

function toAiToolSet(tools: readonly LlmToolDefinition[]): ToolSet {
  const toolSet: ToolSet = {};
  for (const definition of tools) {
    toolSet[definition.name] = tool({
      description: definition.description,
      inputSchema: definition.inputSchema
    });
  }
  return toolSet;
}

function formatToolOutputText(output: unknown): string {
  if (typeof output === "string") {
    return output;
  }
  return JSON.stringify(output);
}

function toAiMessages(messages: readonly LlmAgentMessage[]): ModelMessage[] {
  const converted: ModelMessage[] = [];

  for (const message of messages) {
    if (message.role === "user") {
      converted.push({ role: "user", content: message.content });
      continue;
    }

    if (message.role === "assistant") {
      if (isAssistantModelMessage(message.rawProviderMessage)) {
        converted.push(message.rawProviderMessage);
        continue;
      }

      const parts: Array<
        | { type: "text"; text: string }
        | {
            type: "tool-call";
            toolCallId: string;
            toolName: string;
            input: unknown;
          }
      > = [];
      if (message.text.trim().length > 0) {
        parts.push({ type: "text", text: message.text });
      }
      for (const call of message.toolCalls) {
        parts.push({
          type: "tool-call",
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: call.input
        });
      }
      converted.push({
        role: "assistant",
        content: parts.length > 0 ? parts : message.text
      });
      continue;
    }

    converted.push({
      role: "tool",
      content: message.results.map((result) => ({
        type: "tool-result",
        toolCallId: result.toolCallId,
        toolName: result.toolName,
        output: result.isError
          ? { type: "error-text", value: formatToolOutputText(result.output) }
          : { type: "text", value: formatToolOutputText(result.output) }
      }))
    });
  }

  return converted;
}

function normalizeReasoningText(value: string | undefined): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export class GeminiLlmProvider implements LlmProvider {
  private readonly google: ReturnType<typeof createGoogleGenerativeAI>;
  private readonly maxAttempts: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly temperature: number;
  private readonly telemetry: TelemetryProvider | undefined;

  public constructor(options: GeminiLlmProviderOptions) {
    this.google = createGoogleGenerativeAI({ apiKey: options.apiKey });
    this.maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
    this.baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
    this.maxDelayMs = options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
    this.temperature = options.temperature ?? DEFAULT_TEMPERATURE;
    this.telemetry = options.telemetry;
  }

  private async runWithRetry<T>(
    operation: string,
    model: string,
    execute: () => Promise<T>,
    signal?: AbortSignal
  ): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      if (signal?.aborted) {
        throw new ProviderError(`LLM ${operation} aborted`, {
          cause: signal.reason
        });
      }

      try {
        return await execute();
      } catch (error) {
        lastError = error;
        if (signal?.aborted) {
          throw new ProviderError(`LLM ${operation} aborted`, { cause: error });
        }

        const canRetry =
          attempt < this.maxAttempts && isTransientProviderError(error);
        if (!canRetry) {
          throw new ProviderError(
            `Gemini ${operation} failed on model ${model}`,
            { cause: error }
          );
        }

        const backoffMs = Math.min(
          this.maxDelayMs,
          this.baseDelayMs * 2 ** (attempt - 1)
        );
        await waitWithAbort(backoffMs, signal);
      }
    }

    throw new ProviderError(
      `Gemini ${operation} exhausted ${this.maxAttempts} attempts on model ${model}`,
      { cause: lastError }
    );
  }

  public async synthesize(
    input: LlmSynthesizeInput,
    signal?: AbortSignal
  ): Promise<LlmSynthesizeOutput> {
    return this.runWithRetry(
      "synthesis",
      input.model,
      async () => {
        const telemetry = this.telemetry?.getAiSdkTelemetry("synthesize-answer", {
          model: input.model
        });
        const response = await generateText({
          model: this.google(input.model),
          system: input.system,
          prompt: input.prompt,
          temperature: this.temperature,
          output: Output.object({ schema: AgentSynthesisSchema }),
          maxRetries: 0,
          ...(signal !== undefined ? { abortSignal: signal } : {}),
          ...(telemetry !== undefined ? { experimental_telemetry: telemetry } : {})
        });

        const reasoningText = normalizeReasoningText(response.reasoningText);
        return {
          result: response.output,
          usage: {
            inputTokens: response.usage.inputTokens ?? 0,
            outputTokens: response.usage.outputTokens ?? 0
          },
          model: input.model,
          ...(reasoningText !== undefined ? { reasoningText } : {})
        };
      },
      signal
    );
  }

  public async runAgentStep(
    input: LlmAgentStepInput,
    signal?: AbortSignal
  ): Promise<LlmAgentStepOutput> {
    return this.runWithRetry(
      "agent step",
      input.model,
      async () => {
        const telemetry = this.telemetry?.getAiSdkTelemetry("run-agent-step", {
          model: input.model,
          toolCount: input.tools.length
        });
        const response = await generateText({
          model: this.google(input.model),
          system: input.system,
          messages: toAiMessages(input.messages),
          tools: toAiToolSet(input.tools),
          temperature: this.temperature,
          maxRetries: 0,
          ...(signal !== undefined ? { abortSignal: signal } : {}),
          ...(telemetry !== undefined ? { experimental_telemetry: telemetry } : {})
        });

        const toolCalls: LlmToolCall[] = response.toolCalls.map((call) => ({
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: call.input
        }));

        const assistantMessage = response.response.messages.find(
          (msg) => msg.role === "assistant"
        );
        const reasoningText = normalizeReasoningText(response.reasoningText);

        return {
          text: response.text,
          toolCalls,
          usage: {
            inputTokens: response.usage.inputTokens ?? 0,
            outputTokens: response.usage.outputTokens ?? 0
          },
          model: input.model,
          ...(assistantMessage !== undefined
            ? { rawProviderMessage: assistantMessage }
            : {}),
          ...(reasoningText !== undefined ? { reasoningText } : {})
        };
      },
      signal
    );
  }
}
