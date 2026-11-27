import type { ServerPhase } from "../models/corpus.js";
import { ConfigError, ContractError } from "../models/errors.js";

export interface ServerLifecycleOptions {
  maxConcurrent?: number;
}

interface QueuedSlot {
  resolve: (token: symbol) => void;
  reject: (error: Error) => void;
}

const DEFAULT_MAX_CONCURRENT = 4;
const DEFAULT_DRAIN_GRACE_MS = 8000;

export class ServerLifecycleController {
  private readonly maxConcurrent: number;
  private phase: ServerPhase = { kind: "booting" };
  private ingestDone = false;
  private sigtermReceived = false;

  private readonly activeRequests = new Set<symbol>();
  private readonly waitQueue: QueuedSlot[] = [];
  private readonly startupAbortController = new AbortController();
  private readonly requestAbortController = new AbortController();
  private drainPromise: Promise<void> | null = null;
  private drainResolve: (() => void) | null = null;
  private drainTimer: ReturnType<typeof setTimeout> | null = null;

  public constructor(options: ServerLifecycleOptions = {}) {
    const maxConcurrent = options.maxConcurrent ?? DEFAULT_MAX_CONCURRENT;
    if (!Number.isInteger(maxConcurrent) || maxConcurrent <= 0) {
      throw new ConfigError("maxConcurrent must be a positive integer");
    }
    this.maxConcurrent = maxConcurrent;
  }

  public getPhase(): ServerPhase {
    return this.phase;
  }

  public isIngestDone(): boolean {
    return this.ingestDone;
  }

  public getInFlightCount(): number {
    return this.activeRequests.size;
  }

  public getHealthStatusCode(): 200 | 503 {
    return this.phase.kind === "ready" && this.ingestDone ? 200 : 503;
  }

  public getStartupSignal(): AbortSignal {
    return this.startupAbortController.signal;
  }

  public startIngest(): void {
    if (this.phase.kind !== "booting") {
      throw new ContractError(
        `Cannot start ingest from phase "${this.phase.kind}"`
      );
    }
    this.phase = { kind: "ingesting" };
  }

  public markIngestSucceeded(): void {
    if (this.phase.kind !== "ingesting" || this.sigtermReceived) {
      return;
    }
    this.ingestDone = true;
    this.phase = { kind: "ready" };
  }

  public markIngestFailed(reason: string): void {
    if (this.phase.kind !== "ingesting" || this.sigtermReceived) {
      return;
    }
    this.phase = { kind: "failed", reason };
  }

  private rejectQueuedWaiters(message: string): void {
    while (this.waitQueue.length > 0) {
      const queued = this.waitQueue.shift();
      queued?.reject(new ContractError(message));
    }
  }

  private acquireSlot(): Promise<symbol> {
    if (this.phase.kind !== "ready" || !this.ingestDone) {
      return Promise.reject(
        new ContractError("Server is not ready to accept requests")
      );
    }

    if (this.activeRequests.size < this.maxConcurrent) {
      const token = Symbol("ask");
      this.activeRequests.add(token);
      return Promise.resolve(token);
    }

    return new Promise<symbol>((resolve, reject) => {
      this.waitQueue.push({ resolve, reject });
    });
  }

  private releaseSlot(token: symbol): void {
    if (!this.activeRequests.delete(token)) {
      return;
    }

    if (this.phase.kind === "ready" && this.waitQueue.length > 0) {
      const next = this.waitQueue.shift();
      if (next !== undefined) {
        const nextToken = Symbol("ask");
        this.activeRequests.add(nextToken);
        next.resolve(nextToken);
      }
      return;
    }

    if (this.phase.kind === "draining" && this.activeRequests.size === 0) {
      this.finishDrain();
    }
  }

  private finishDrain(): void {
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
    this.activeRequests.clear();
    this.phase = { kind: "stopped" };
    const resolve = this.drainResolve;
    this.drainResolve = null;
    resolve?.();
  }

  public async runAsk<T>(
    handler: (signal: AbortSignal) => Promise<T>
  ): Promise<T> {
    const token = await this.acquireSlot();
    try {
      return await handler(this.requestAbortController.signal);
    } finally {
      this.releaseSlot(token);
    }
  }

  public requestShutdown(graceMs = DEFAULT_DRAIN_GRACE_MS): Promise<void> {
    this.sigtermReceived = true;
    this.rejectQueuedWaiters("Server is draining and cannot start new requests");

    if (this.phase.kind === "stopped" || this.phase.kind === "failed") {
      return Promise.resolve();
    }

    if (this.phase.kind === "booting" || this.phase.kind === "ingesting") {
      this.startupAbortController.abort(new Error("Shutdown requested"));
      this.phase = { kind: "stopped" };
      return Promise.resolve();
    }

    if (this.phase.kind === "ready") {
      if (this.activeRequests.size === 0) {
        this.phase = { kind: "stopped" };
        return Promise.resolve();
      }

      this.phase = { kind: "draining", startedAtMs: Date.now() };
      this.drainPromise = new Promise<void>((resolve) => {
        this.drainResolve = resolve;
      });

      this.drainTimer = setTimeout(() => {
        this.requestAbortController.abort(new Error("Drain timeout exceeded"));
        this.finishDrain();
      }, Math.max(0, graceMs));
      this.drainTimer.unref();

      return this.drainPromise;
    }

    return this.drainPromise ?? Promise.resolve();
  }
}
