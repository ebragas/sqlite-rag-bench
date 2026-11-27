import { performance } from "node:perf_hooks";
import { ContractError } from "../models/errors.js";
import { AskResponseSchema } from "../models/eval-harness.js";

export const MAX_ASK_RESPONSE_BYTES = 1024 * 1024;
const DEFAULT_ASK_TIMEOUT_MS = 180_000;

export interface TokenUsage {
  readonly inputTokens: number;
  readonly outputTokens: number;
}

export interface ResponderCallOptions {
  readonly headers?: Readonly<Record<string, string>>;
  readonly signal?: AbortSignal;
}

export interface ResponderResult {
  readonly answer: string;
  readonly documentIds: readonly string[];
  readonly latencyMs: number;
  readonly responseBytes: number;
  readonly usage?: TokenUsage;
}

export interface ResponderAdapter {
  readonly modelId: string;
  answer(question: string, options?: ResponderCallOptions): Promise<ResponderResult>;
}

export interface LiveHttpResponderOptions {
  readonly baseUrl: string;
  readonly modelId: string;
  readonly timeoutMs?: number;
  readonly fetchFn?: typeof fetch;
}

function buildAbortSignal(timeoutMs: number, externalSignal?: AbortSignal): AbortSignal {
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  if (externalSignal === undefined) {
    return timeoutSignal;
  }
  return AbortSignal.any([timeoutSignal, externalSignal]);
}

export class LiveHttpResponder implements ResponderAdapter {
  public readonly modelId: string;
  private readonly askUrl: string;
  private readonly timeoutMs: number;
  private readonly fetchFn: typeof fetch;

  public constructor(options: LiveHttpResponderOptions) {
    const normalizedBase = options.baseUrl.replace(/\/+$/, "");
    this.askUrl = `${normalizedBase}/ask`;
    this.modelId = options.modelId;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_ASK_TIMEOUT_MS;
    this.fetchFn = options.fetchFn ?? fetch;
  }

  public async answer(question: string, options?: ResponderCallOptions): Promise<ResponderResult> {
    const startedAt = performance.now();
    const signal = buildAbortSignal(this.timeoutMs, options?.signal);
    const headers: Record<string, string> = {
      "content-type": "application/json",
      ...options?.headers
    };

    let response: Response;
    try {
      response = await this.fetchFn(this.askUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({ question }),
        signal
      });
    } catch (error) {
      throw new ContractError(`POST ${this.askUrl} failed`, { cause: error });
    }

    const rawBody = await response.text();
    const latencyMs = Math.max(0, performance.now() - startedAt);
    const responseBytes = Buffer.byteLength(rawBody, "utf8");

    if (response.status !== 200) {
      throw new ContractError(
        `POST ${this.askUrl} returned status ${response.status}: ${rawBody.slice(0, 200)}`
      );
    }

    if (responseBytes > MAX_ASK_RESPONSE_BYTES) {
      throw new ContractError(
        `POST ${this.askUrl} response size ${responseBytes} bytes exceeds limit ${MAX_ASK_RESPONSE_BYTES}`
      );
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(rawBody);
    } catch (error) {
      throw new ContractError(`POST ${this.askUrl} returned invalid JSON`, { cause: error });
    }

    const validated = AskResponseSchema.safeParse(parsedJson);
    if (!validated.success) {
      throw new ContractError(
        `POST ${this.askUrl} failed response schema check: ${validated.error.message}`
      );
    }

    return {
      answer: validated.data.answer,
      documentIds: validated.data.document_ids,
      latencyMs,
      responseBytes
    };
  }
}
