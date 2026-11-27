import {
  createGoogleGenerativeAI,
  type GoogleEmbeddingModelOptions,
  type GoogleGenerativeAIProviderSettings
} from "@ai-sdk/google";
import { APICallError, embedMany } from "ai";
import { ProviderError } from "../models/corpus.js";
import { ConfigError } from "../models/errors.js";
import { l2Normalize } from "../utils/vector.js";

export interface EmbedBatchResult {
  readonly vectors: readonly Float32Array[];
  readonly inputTokens: number;
}

export interface Embedder {
  readonly model: string;
  readonly dims: number;
  embedBatch(
    texts: readonly string[],
    signal?: AbortSignal
  ): Promise<EmbedBatchResult>;
}

export interface GeminiEmbedderOptions {
  readonly apiKey: string;
  readonly model?: string;
  readonly dims?: number;
  readonly batchSize?: number;
  readonly maxAttempts?: number;
  readonly baseDelayMs?: number;
  readonly maxDelayMs?: number;
  readonly tokensPerMinute?: number;
  readonly taskType?: GoogleEmbeddingModelOptions["taskType"];
  readonly baseURL?: string;
  readonly fetch?: GoogleGenerativeAIProviderSettings["fetch"];
}

const DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001";
const DEFAULT_EMBEDDING_DIMS = 768;
const DEFAULT_BATCH_SIZE = 100;
const DEFAULT_MAX_ATTEMPTS = 8;
const DEFAULT_BASE_DELAY_MS = 1000;
const DEFAULT_MAX_DELAY_MS = 65_000;
const DEFAULT_TOKENS_PER_MINUTE = 900_000;
const TPM_WINDOW_MS = 60_000;
const RETRY_DELAY_SECONDS_PATTERN = /retry(?:\s+in\s+|Delay["':\s]+)(\d+(?:\.\d+)?)s/i;

function isTransientEmbeddingError(error: unknown): boolean {
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
      new ProviderError("Embedding request aborted during retry backoff", {
        cause: signal.reason
      })
    );
  }

  return new Promise<void>((resolve, reject) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(
        new ProviderError("Embedding request aborted during retry backoff", {
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

export function parseRetryDelayMs(error: unknown): number | null {
  const candidates: string[] = [];
  if (APICallError.isInstance(error)) {
    const retryAfter = error.responseHeaders?.["retry-after"];
    if (typeof retryAfter === "string") {
      const parsedSeconds = Number(retryAfter.trim());
      if (Number.isFinite(parsedSeconds) && parsedSeconds >= 0) {
        return Math.ceil(parsedSeconds * 1000);
      }
    }
    if (typeof error.responseBody === "string") {
      candidates.push(error.responseBody);
    }
    candidates.push(error.message);
  } else if (error instanceof Error) {
    candidates.push(error.message);
  }

  for (const text of candidates) {
    const match = RETRY_DELAY_SECONDS_PATTERN.exec(text);
    const rawSeconds = match?.[1];
    if (rawSeconds !== undefined) {
      const seconds = Number(rawSeconds);
      if (Number.isFinite(seconds) && seconds >= 0) {
        return Math.ceil(seconds * 1000) + 1000;
      }
    }
  }
  return null;
}

function estimateBatchTokens(texts: readonly string[]): number {
  let total = 0;
  for (const text of texts) {
    const trimmed = text.trim();
    if (trimmed.length > 0) {
      total += Math.max(1, Math.ceil(trimmed.length / 4));
    }
  }
  return total;
}

export class GeminiEmbedder implements Embedder {
  public readonly model: string;
  public readonly dims: number;

  private readonly google: ReturnType<typeof createGoogleGenerativeAI>;
  private readonly batchSize: number;
  private readonly maxAttempts: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly taskType: GoogleEmbeddingModelOptions["taskType"] | undefined;
  private readonly tpmEntries: Array<{ timestamp: number; tokens: number }> = [];
  private activeTokensPerMinute: number | null;
  private cooldownUntilMs = 0;
  private tpmWaitChain: Promise<void> = Promise.resolve();

  public get effectiveTokensPerMinute(): number | null {
    return this.activeTokensPerMinute;
  }

  public constructor(options: GeminiEmbedderOptions) {
    if (options.apiKey.trim().length === 0) {
      throw new ConfigError("GeminiEmbedder requires a non-empty apiKey");
    }

    const model = options.model ?? DEFAULT_EMBEDDING_MODEL;
    if (model.trim().length === 0) {
      throw new ConfigError("GeminiEmbedder requires a non-empty model");
    }

    const dims = options.dims ?? DEFAULT_EMBEDDING_DIMS;
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new ConfigError(`GeminiEmbedder dims must be a positive integer, got ${dims}`);
    }

    const batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      throw new ConfigError(
        `GeminiEmbedder batchSize must be a positive integer, got ${batchSize}`
      );
    }

    const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
    if (!Number.isInteger(maxAttempts) || maxAttempts <= 0) {
      throw new ConfigError(
        `GeminiEmbedder maxAttempts must be a positive integer, got ${maxAttempts}`
      );
    }

    this.model = model;
    this.dims = dims;
    this.batchSize = batchSize;
    this.maxAttempts = maxAttempts;
    this.baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
    this.maxDelayMs = options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
    this.activeTokensPerMinute =
      options.tokensPerMinute !== undefined
        ? options.tokensPerMinute > 0
          ? options.tokensPerMinute
          : null
        : null;
    this.taskType = options.taskType;
    this.google = createGoogleGenerativeAI({
      apiKey: options.apiKey,
      ...(options.baseURL !== undefined ? { baseURL: options.baseURL } : {}),
      ...(options.fetch !== undefined ? { fetch: options.fetch } : {})
    });
  }

  private async paceTokens(estimatedTokens: number, signal?: AbortSignal): Promise<void> {
    if (estimatedTokens <= 0) {
      return;
    }

    const waitTask = this.tpmWaitChain.then(async () => {
      for (;;) {
        const now = Date.now();
        if (now < this.cooldownUntilMs) {
          await waitWithAbort(this.cooldownUntilMs - now, signal);
          continue;
        }
        while (
          this.tpmEntries.length > 0 &&
          now - (this.tpmEntries[0]?.timestamp ?? 0) >= TPM_WINDOW_MS
        ) {
          this.tpmEntries.shift();
        }
        if (this.activeTokensPerMinute === null) {
          this.tpmEntries.push({ timestamp: Date.now(), tokens: estimatedTokens });
          return;
        }
        let used = 0;
        for (const entry of this.tpmEntries) {
          used += entry.tokens;
        }
        if (used + estimatedTokens <= this.activeTokensPerMinute || this.tpmEntries.length === 0) {
          this.tpmEntries.push({ timestamp: Date.now(), tokens: estimatedTokens });
          return;
        }
        const oldest = this.tpmEntries[0]?.timestamp ?? now;
        const sleepMs = Math.max(250, oldest + TPM_WINDOW_MS - now + 250);
        await waitWithAbort(sleepMs, signal);
      }
    });

    this.tpmWaitChain = waitTask.catch(() => {});
    await waitTask;
  }

  private async runWithRetry<T>(
    execute: () => Promise<T>,
    signal?: AbortSignal
  ): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      if (signal?.aborted) {
        throw new ProviderError("Gemini embedding request aborted", {
          cause: signal.reason
        });
      }

      try {
        return await execute();
      } catch (error) {
        lastError = error;
        if (signal?.aborted) {
          throw new ProviderError("Gemini embedding request aborted", {
            cause: error
          });
        }

        const canRetry =
          attempt < this.maxAttempts && isTransientEmbeddingError(error);
        if (!canRetry) {
          throw new ProviderError(
            `Gemini embedding failed on model ${this.model}`,
            { cause: error }
          );
        }

        const serverDelayMs = parseRetryDelayMs(error);
        const exponentialDelayMs = this.baseDelayMs * 2 ** (attempt - 1);
        const backoffMs = Math.min(
          this.maxDelayMs,
          serverDelayMs !== null
            ? Math.max(serverDelayMs, exponentialDelayMs)
            : exponentialDelayMs
        );
        if (APICallError.isInstance(error) && error.statusCode === 429) {
          if (this.activeTokensPerMinute === null) {
            this.activeTokensPerMinute = DEFAULT_TOKENS_PER_MINUTE;
          }
          this.cooldownUntilMs = Math.max(this.cooldownUntilMs, Date.now() + backoffMs);
        }
        await waitWithAbort(backoffMs, signal);
      }
    }

    throw new ProviderError(
      `Gemini embedding exhausted ${this.maxAttempts} attempts on model ${this.model}`,
      { cause: lastError }
    );
  }

  public async embedBatch(
    texts: readonly string[],
    signal?: AbortSignal
  ): Promise<EmbedBatchResult> {
    if (texts.length === 0) {
      return { vectors: [], inputTokens: 0 };
    }

    const vectors: Float32Array[] = [];
    let inputTokens = 0;

    for (let offset = 0; offset < texts.length; offset += this.batchSize) {
      const batch = texts.slice(offset, offset + this.batchSize);
      const batchEstTokens = estimateBatchTokens(batch);
      await this.paceTokens(batchEstTokens, signal);
      const response = await this.runWithRetry(async () => {
        const googleOptions: GoogleEmbeddingModelOptions = {
          outputDimensionality: this.dims,
          ...(this.taskType !== undefined ? { taskType: this.taskType } : {})
        };

        return embedMany({
          model: this.google.embeddingModel(this.model),
          values: [...batch],
          maxRetries: 0,
          providerOptions: {
            google: googleOptions
          },
          ...(signal !== undefined ? { abortSignal: signal } : {})
        });
      }, signal);

      if (response.embeddings.length !== batch.length) {
        throw new ProviderError(
          `Gemini embedding returned ${response.embeddings.length} vectors for batch of ${batch.length}`
        );
      }

      for (const rawVector of response.embeddings) {
        if (rawVector.length !== this.dims) {
          throw new ProviderError(
            `Gemini embedding vector dimension mismatch: expected ${this.dims}, got ${rawVector.length}`
          );
        }
        vectors.push(l2Normalize(Float32Array.from(rawVector)));
      }

      const reportedTokens = response.usage.tokens;
      if (Number.isFinite(reportedTokens) && reportedTokens > 0) {
        inputTokens += Math.trunc(reportedTokens);
      } else {
        inputTokens += estimateBatchTokens(batch);
      }
    }

    return {
      vectors,
      inputTokens
    };
  }
}
