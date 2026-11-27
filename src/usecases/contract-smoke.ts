import { spawn, type ChildProcess } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { MAX_ASK_RESPONSE_BYTES } from "../adapters/responder.js";
import { ContractError } from "../models/errors.js";
import { AskResponseSchema } from "../models/eval-harness.js";

const DEFAULT_SMOKE_QUESTION = "What is the current service smoke test status?";
const DEFAULT_REQUEST_TIMEOUT_MS = 180_000;
const DEFAULT_STARTUP_TIMEOUT_MS = 30_000;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 10_000;
const DEFAULT_POLL_INTERVAL_MS = 100;
const CONCURRENCY_LEVEL = 4;
const MAX_CHILD_LOG_TAIL_BYTES = 4096;

interface ChildOutputCapture {
  readonly getTailSummary: () => string;
}

function appendTailBuffer(current: Buffer, chunk: Buffer): Buffer {
  if (chunk.length >= MAX_CHILD_LOG_TAIL_BYTES) {
    return Buffer.from(chunk.subarray(chunk.length - MAX_CHILD_LOG_TAIL_BYTES));
  }
  const combined = Buffer.concat([current, chunk]);
  if (combined.length <= MAX_CHILD_LOG_TAIL_BYTES) {
    return combined;
  }
  return Buffer.from(combined.subarray(combined.length - MAX_CHILD_LOG_TAIL_BYTES));
}

function attachChildOutputDrain(child: ChildProcess): ChildOutputCapture {
  let stdoutTail: Buffer = Buffer.alloc(0);
  let stderrTail: Buffer = Buffer.alloc(0);

  child.stdout?.on("data", (chunk: Buffer | string) => {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
    stdoutTail = appendTailBuffer(stdoutTail, buffer);
  });
  child.stderr?.on("data", (chunk: Buffer | string) => {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
    stderrTail = appendTailBuffer(stderrTail, buffer);
  });

  return {
    getTailSummary: (): string => {
      const parts: string[] = [];
      const stderrText = stderrTail.toString("utf8").trim();
      const stdoutText = stdoutTail.toString("utf8").trim();
      if (stderrText.length > 0) {
        parts.push(`stderr tail: ${stderrText}`);
      }
      if (stdoutText.length > 0) {
        parts.push(`stdout tail: ${stdoutText}`);
      }
      return parts.join("; ");
    }
  };
}

export interface SpawnedServerOptions {
  readonly command: string;
  readonly args?: readonly string[];
  readonly env?: NodeJS.ProcessEnv;
  readonly cwd?: string;
  readonly startupTimeoutMs?: number;
  readonly shutdownTimeoutMs?: number;
  readonly pollIntervalMs?: number;
}

export interface ContractSmokeOptions {
  readonly baseUrl: string;
  readonly sampleQuestion?: string;
  readonly requestTimeoutMs?: number;
  readonly fetchFn?: typeof fetch;
  readonly spawn?: SpawnedServerOptions;
  readonly throwOnFailure?: boolean;
}

export interface ContractSmokeCheckResult {
  readonly name: string;
  readonly passed: boolean;
  readonly detail: string;
}

export interface ContractSmokeReport {
  readonly passed: boolean;
  readonly baseUrl: string;
  readonly observedNotReadyStatus: number | null;
  readonly checks: readonly ContractSmokeCheckResult[];
}

async function fetchValidAsk(
  askUrl: string,
  question: string,
  timeoutMs: number,
  fetchFn: typeof fetch
): Promise<{ ok: true; responseBytes: number } | { ok: false; reason: string }> {
  let response: Response;
  try {
    response = await fetchFn(askUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(timeoutMs)
    });
  } catch (error) {
    return {
      ok: false,
      reason: `request failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }

  const rawBody = await response.text();
  const responseBytes = Buffer.byteLength(rawBody, "utf8");

  if (response.status !== 200) {
    return { ok: false, reason: `expected status 200, got ${response.status}` };
  }
  if (responseBytes > MAX_ASK_RESPONSE_BYTES) {
    return {
      ok: false,
      reason: `response size ${responseBytes} bytes exceeds 1 MiB (${MAX_ASK_RESPONSE_BYTES} bytes)`
    };
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(rawBody);
  } catch (error) {
    return {
      ok: false,
      reason: `invalid JSON body: ${error instanceof Error ? error.message : String(error)}`
    };
  }

  const validated = AskResponseSchema.safeParse(parsedJson);
  if (!validated.success) {
    return { ok: false, reason: `invalid response schema: ${validated.error.message}` };
  }

  return { ok: true, responseBytes };
}

async function checkHealth(
  healthUrl: string,
  timeoutMs: number,
  fetchFn: typeof fetch
): Promise<ContractSmokeCheckResult> {
  try {
    const response = await fetchFn(healthUrl, {
      method: "GET",
      signal: AbortSignal.timeout(timeoutMs)
    });
    if (response.status === 200) {
      return { name: "health_ready", passed: true, detail: "GET /health returned 200" };
    }
    return {
      name: "health_ready",
      passed: false,
      detail: `expected GET /health status 200, got ${response.status}`
    };
  } catch (error) {
    return {
      name: "health_ready",
      passed: false,
      detail: `GET /health failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

async function checkMalformedAsk(
  askUrl: string,
  timeoutMs: number,
  fetchFn: typeof fetch
): Promise<ContractSmokeCheckResult> {
  try {
    const response = await fetchFn(askUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{malformed-json",
      signal: AbortSignal.timeout(timeoutMs)
    });
    if (response.status === 400) {
      return { name: "ask_malformed", passed: true, detail: "malformed POST /ask returned 400" };
    }
    return {
      name: "ask_malformed",
      passed: false,
      detail: `expected status 400 for malformed POST /ask, got ${response.status}`
    };
  } catch (error) {
    return {
      name: "ask_malformed",
      passed: false,
      detail: `malformed POST /ask failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

function formatTailSuffix(outputCapture: ChildOutputCapture): string {
  const tail = outputCapture.getTailSummary();
  return tail.length > 0 ? ` [${tail}]` : "";
}

async function waitForHealthReady(
  healthUrl: string,
  child: ChildProcess,
  getSpawnError: () => Error | null,
  outputCapture: ChildOutputCapture,
  startupTimeoutMs: number,
  pollIntervalMs: number,
  fetchFn: typeof fetch
): Promise<{ ready: boolean; observedNotReadyStatus: number | null; detail: string }> {
  const deadline = Date.now() + startupTimeoutMs;
  let observedNotReadyStatus: number | null = null;
  let exitedDetail: string | null = null;
  let lastErrorDetail: string | null = null;
  const lifecycleAbort = new AbortController();

  const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
    exitedDetail = `process exited before ready (code=${String(code)}, signal=${String(signal)})`;
    lifecycleAbort.abort();
  };
  const onError = (): void => {
    lifecycleAbort.abort();
  };
  child.once("exit", onExit);
  child.once("error", onError);

  try {
    while (Date.now() < deadline) {
      const currentSpawnError = getSpawnError();
      if (currentSpawnError !== null) {
        return {
          ready: false,
          observedNotReadyStatus,
          detail: `failed to spawn process: ${currentSpawnError.message}`
        };
      }
      if (exitedDetail !== null) {
        return {
          ready: false,
          observedNotReadyStatus,
          detail: `${exitedDetail}${formatTailSuffix(outputCapture)}`
        };
      }

      try {
        const response = await fetchFn(healthUrl, {
          method: "GET",
          signal: AbortSignal.timeout(Math.min(1000, Math.max(100, deadline - Date.now())))
        });
        if (response.status === 200) {
          return {
            ready: true,
            observedNotReadyStatus,
            detail:
              observedNotReadyStatus !== null
                ? `observed /health ${observedNotReadyStatus} -> 200`
                : "observed /health 200"
          };
        }
        if (observedNotReadyStatus === null) {
          observedNotReadyStatus = response.status;
        }
      } catch (error) {
        lastErrorDetail = error instanceof Error ? error.message : String(error);
        if (error instanceof Error && error.name === "AbortError") {
          break;
        }
      }

      const postFetchSpawnError = getSpawnError();
      if (postFetchSpawnError !== null) {
        return {
          ready: false,
          observedNotReadyStatus,
          detail: `failed to spawn process: ${postFetchSpawnError.message}`
        };
      }
      if (exitedDetail !== null) {
        return {
          ready: false,
          observedNotReadyStatus,
          detail: `${exitedDetail}${formatTailSuffix(outputCapture)}`
        };
      }

      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        break;
      }
      try {
        await delay(Math.min(pollIntervalMs, remaining), undefined, {
          signal: lifecycleAbort.signal
        });
      } catch (error) {
        if (!(error instanceof Error && error.name === "AbortError")) {
          throw error;
        }
      }
    }

    const finalSpawnError = getSpawnError();
    if (finalSpawnError !== null) {
      return {
        ready: false,
        observedNotReadyStatus,
        detail: `failed to spawn process: ${finalSpawnError.message}`
      };
    }
    if (exitedDetail !== null) {
      return {
        ready: false,
        observedNotReadyStatus,
        detail: `${exitedDetail}${formatTailSuffix(outputCapture)}`
      };
    }

    const suffix = lastErrorDetail !== null ? ` (last error: ${lastErrorDetail})` : "";
    return {
      ready: false,
      observedNotReadyStatus,
      detail: `server did not become ready within ${startupTimeoutMs} ms${suffix}${formatTailSuffix(outputCapture)}`
    };
  } finally {
    child.off("exit", onExit);
    child.off("error", onError);
  }
}

async function stopChildGracefully(
  child: ChildProcess,
  shutdownTimeoutMs: number
): Promise<ContractSmokeCheckResult> {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) {
    return {
      name: "sigterm_shutdown",
      passed: false,
      detail: `process already exited before SIGTERM (code=${String(child.exitCode)}, signal=${String(child.signalCode)})`
    };
  }

  return await new Promise<ContractSmokeCheckResult>((resolve) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve({
        name: "sigterm_shutdown",
        passed: false,
        detail: `process did not exit within ${shutdownTimeoutMs} ms after SIGTERM`
      });
    }, shutdownTimeoutMs);

    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      const clean = code === 0 || signal === "SIGTERM";
      resolve({
        name: "sigterm_shutdown",
        passed: clean,
        detail: `process exited after SIGTERM (code=${String(code)}, signal=${String(signal)})`
      });
    });

    child.kill("SIGTERM");
  });
}

export async function runContractSmokeSuite(
  options: ContractSmokeOptions
): Promise<ContractSmokeReport> {
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const healthUrl = `${baseUrl}/health`;
  const askUrl = `${baseUrl}/ask`;
  const question = options.sampleQuestion ?? DEFAULT_SMOKE_QUESTION;
  const requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
  const fetchFn = options.fetchFn ?? fetch;
  const checks: ContractSmokeCheckResult[] = [];
  let observedNotReadyStatus: number | null = null;

  let child: ChildProcess | undefined;
  let spawnError: Error | null = null;
  let outputCapture: ChildOutputCapture | undefined;
  let startupSucceeded = false;

  if (options.spawn !== undefined) {
    child = spawn(options.spawn.command, [...(options.spawn.args ?? [])], {
      cwd: options.spawn.cwd,
      env: options.spawn.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"]
    });
    child.on("error", (error: Error) => {
      if (spawnError === null) {
        spawnError = error;
      }
    });
    outputCapture = attachChildOutputDrain(child);
  }

  try {
    if (child !== undefined && options.spawn !== undefined && outputCapture !== undefined) {
      const startup = await waitForHealthReady(
        healthUrl,
        child,
        () => spawnError,
        outputCapture,
        options.spawn.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS,
        options.spawn.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS,
        fetchFn
      );
      observedNotReadyStatus = startup.observedNotReadyStatus;
      checks.push({
        name: "startup_readiness",
        passed: startup.ready,
        detail: startup.detail
      });
      if (!startup.ready) {
        const report: ContractSmokeReport = {
          passed: false,
          baseUrl,
          observedNotReadyStatus,
          checks
        };
        if (options.throwOnFailure !== false) {
          throw new ContractError(`Contract smoke failed: ${startup.detail}`);
        }
        return report;
      }
      startupSucceeded = true;
    }

    checks.push(await checkHealth(healthUrl, requestTimeoutMs, fetchFn));

    const validAsk = await fetchValidAsk(askUrl, question, requestTimeoutMs, fetchFn);
    checks.push({
      name: "ask_valid",
      passed: validAsk.ok,
      detail: validAsk.ok
        ? `POST /ask returned valid JSON (${validAsk.responseBytes} bytes)`
        : validAsk.reason
    });

    checks.push(await checkMalformedAsk(askUrl, requestTimeoutMs, fetchFn));

    const concurrentResults = await Promise.all(
      Array.from({ length: CONCURRENCY_LEVEL }, (_, index) =>
        fetchValidAsk(askUrl, `${question} [${index + 1}]`, requestTimeoutMs, fetchFn)
      )
    );
    const failedConcurrent = concurrentResults.filter((result) => !result.ok);
    checks.push({
      name: "ask_concurrency_4",
      passed: failedConcurrent.length === 0,
      detail:
        failedConcurrent.length === 0
          ? `${CONCURRENCY_LEVEL}/${CONCURRENCY_LEVEL} concurrent POST /ask requests returned 200`
          : `${failedConcurrent.length}/${CONCURRENCY_LEVEL} concurrent requests failed: ${failedConcurrent
              .map((item) => (item.ok ? "" : item.reason))
              .join("; ")}`
    });
  } finally {
    if (child !== undefined && spawnError === null && child.pid !== undefined) {
      const shutdownCheck = await stopChildGracefully(
        child,
        options.spawn?.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS
      );
      if (startupSucceeded) {
        checks.push(shutdownCheck);
      }
    }
  }

  const passed = checks.every((check) => check.passed);
  const report: ContractSmokeReport = {
    passed,
    baseUrl,
    observedNotReadyStatus,
    checks
  };

  if (!passed && options.throwOnFailure !== false) {
    const failedSummary = checks
      .filter((check) => !check.passed)
      .map((check) => `${check.name}: ${check.detail}`)
      .join(" | ");
    throw new ContractError(`Contract smoke failed: ${failedSummary}`);
  }

  return report;
}
