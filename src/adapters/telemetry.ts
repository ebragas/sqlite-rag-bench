export const DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com";

export interface LangfuseEnvConfig {
  readonly publicKey: string;
  readonly secretKey: string;
  readonly baseUrl: string;
}

export interface W3cTraceparent {
  readonly version: "00";
  readonly traceId: string;
  readonly spanId: string;
  readonly traceFlags: number;
}

export type TelemetryMetadataValue =
  | string
  | number
  | boolean
  | string[]
  | number[]
  | boolean[];

export interface AiSdkTelemetryConfig {
  readonly isEnabled: true;
  readonly functionId: string;
  readonly metadata?: Record<string, TelemetryMetadataValue>;
}

export type TelemetryObservationType = "agent" | "retriever" | "tool" | "span";

export interface TelemetryObservationOptions {
  readonly name: string;
  readonly asType: TelemetryObservationType;
  readonly input?: unknown;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface TelemetryObservationContext {
  setOutput(output: unknown): void;
  readonly traceId?: string;
}

export interface TelemetryProvider {
  readonly enabled: boolean;
  getAiSdkTelemetry(
    functionId: string,
    metadata?: Readonly<Record<string, TelemetryMetadataValue>>
  ): AiSdkTelemetryConfig | undefined;
  startObservation<T>(
    options: TelemetryObservationOptions,
    fn: (ctx: TelemetryObservationContext) => Promise<T>
  ): Promise<T>;
  getActiveTraceId(): string | undefined;
}

export interface TelemetryController extends TelemetryProvider {
  readonly enabled: boolean;
  readonly config: LangfuseEnvConfig | null;
  injectTraceHeaders(headers?: Readonly<Record<string, string>>): Record<string, string>;
  runWithIncomingTraceContext<T>(
    headers: Readonly<Record<string, string | string[] | undefined>>,
    fn: () => Promise<T>
  ): Promise<T>;
  runInSpan<T>(
    name: string,
    fn: () => Promise<T>,
    attributes?: Readonly<Record<string, string | number | boolean>>
  ): Promise<T>;
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}

export type OptionalModuleLoader = (specifier: string) => Promise<unknown>;

export interface SpanProcessorLike {
  forceFlush(): Promise<void>;
  shutdown(): Promise<void>;
}

export type TelemetryExportMode = "immediate" | "batched";

export interface TelemetryControllerOptions {
  readonly serviceName?: string;
  readonly exportMode?: TelemetryExportMode;
  readonly loadModule?: OptionalModuleLoader;
  readonly spanProcessor?: SpanProcessorLike;
  readonly onWarning?: (message: string) => void;
}

interface OtelSpanContextLike {
  readonly traceId: string;
  readonly spanId: string;
  readonly traceFlags: number;
}

interface OtelSpanLike {
  spanContext(): OtelSpanContextLike;
  setAttributes(attributes: Record<string, string | number | boolean>): void;
  recordException(exception: unknown): void;
  setStatus(status: { code: number; message?: string }): void;
  end(): void;
}

interface OtelTracerLike {
  startActiveSpan<T>(
    name: string,
    options: { kind?: number; attributes?: Record<string, string | number | boolean> },
    context: unknown,
    fn: (span: OtelSpanLike) => Promise<T>
  ): Promise<T>;
}

interface OtelApiModule {
  readonly ROOT_CONTEXT: unknown;
  readonly SpanKind: { readonly SERVER: number; readonly INTERNAL: number };
  readonly SpanStatusCode: { readonly ERROR: number };
  readonly context: {
    active(): unknown;
    with<T>(ctx: unknown, fn: () => T): T;
    disable?(): void;
  };
  readonly propagation: {
    extract(ctx: unknown, carrier: Record<string, string>): unknown;
    inject(ctx: unknown, carrier: Record<string, string>): void;
    disable?(): void;
  };
  readonly trace: {
    getTracer(name: string): OtelTracerLike;
    getSpan(ctx: unknown): OtelSpanLike | undefined;
    disable?(): void;
  };
}

interface OtelNodeSdkInstance {
  start(): void;
  shutdown(): Promise<void>;
}

interface OtelSdkNodeModule {
  readonly NodeSDK: new (config: {
    serviceName?: string;
    spanProcessors: SpanProcessorLike[];
    instrumentations: unknown[];
  }) => OtelNodeSdkInstance;
}

interface LangfuseOtelModule {
  readonly LangfuseSpanProcessor: new (params: {
    publicKey: string;
    secretKey: string;
    baseUrl: string;
    exportMode?: TelemetryExportMode;
  }) => SpanProcessorLike;
}

interface LangfuseObservationLike {
  readonly traceId?: string;
  update(attributes: {
    input?: unknown;
    output?: unknown;
    metadata?: Record<string, unknown>;
    level?: "DEBUG" | "DEFAULT" | "WARNING" | "ERROR";
    statusMessage?: string;
  }): unknown;
}

interface LangfusePropagateAttributesParams {
  readonly traceName?: string;
  readonly metadata?: Record<string, unknown>;
  readonly tags?: string[];
  readonly version?: string;
}

interface LangfuseTracingModule {
  readonly startActiveObservation: <T>(
    name: string,
    fn: (observation: LangfuseObservationLike) => Promise<T>,
    options?: { asType?: TelemetryObservationType }
  ) => Promise<T>;
  readonly propagateAttributes?: <T>(
    params: LangfusePropagateAttributesParams,
    fn: () => Promise<T>
  ) => Promise<T>;
  readonly getLangfuseTracer?: () => OtelTracerLike;
  readonly getActiveTraceId?: () => string | undefined;
}

const TRACE_ID_PATTERN = /^[0-9a-f]{32}$/;
const SPAN_ID_PATTERN = /^[0-9a-f]{16}$/;
const ALL_ZERO_TRACE_ID = "00000000000000000000000000000000";
const ALL_ZERO_SPAN_ID = "0000000000000000";
const TRACEPARENT_PATTERN = /^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/;

function defaultLoadModule(specifier: string): Promise<unknown> {
  return import(specifier);
}

function defaultWriteWarning(message: string): void {
  process.stderr.write(`${message}\n`);
}

function readTrimmedEnv(env: NodeJS.ProcessEnv, key: string): string | null {
  const value = env[key];
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function parseLangfuseEnv(env: NodeJS.ProcessEnv): LangfuseEnvConfig | null {
  const publicKey = readTrimmedEnv(env, "LANGFUSE_PUBLIC_KEY");
  const secretKey = readTrimmedEnv(env, "LANGFUSE_SECRET_KEY");
  if (publicKey === null || secretKey === null) {
    return null;
  }

  const baseUrl =
    readTrimmedEnv(env, "LANGFUSE_BASE_URL") ??
    readTrimmedEnv(env, "LANGFUSE_HOST") ??
    DEFAULT_LANGFUSE_BASE_URL;

  return { publicKey, secretKey, baseUrl };
}

function isValidW3cSpanContext(traceId: string, spanId: string, traceFlags: number): boolean {
  const normalizedTraceId = traceId.trim().toLowerCase();
  const normalizedSpanId = spanId.trim().toLowerCase();
  return (
    TRACE_ID_PATTERN.test(normalizedTraceId) &&
    normalizedTraceId !== ALL_ZERO_TRACE_ID &&
    SPAN_ID_PATTERN.test(normalizedSpanId) &&
    normalizedSpanId !== ALL_ZERO_SPAN_ID &&
    Number.isInteger(traceFlags) &&
    traceFlags >= 0 &&
    traceFlags <= 255
  );
}

export function formatW3cTraceparent(
  traceId: string,
  spanId: string,
  traceFlags = 1
): string {
  const normalizedTraceId = traceId.trim().toLowerCase();
  const normalizedSpanId = spanId.trim().toLowerCase();
  if (!TRACE_ID_PATTERN.test(normalizedTraceId) || normalizedTraceId === ALL_ZERO_TRACE_ID) {
    throw new RangeError(`Invalid W3C traceId: "${traceId}"`);
  }
  if (!SPAN_ID_PATTERN.test(normalizedSpanId) || normalizedSpanId === ALL_ZERO_SPAN_ID) {
    throw new RangeError(`Invalid W3C spanId: "${spanId}"`);
  }
  if (!Number.isInteger(traceFlags) || traceFlags < 0 || traceFlags > 255) {
    throw new RangeError(`Invalid W3C traceFlags: "${String(traceFlags)}"`);
  }
  const flagsHex = traceFlags.toString(16).padStart(2, "0");
  return `00-${normalizedTraceId}-${normalizedSpanId}-${flagsHex}`;
}

export function parseW3cTraceparent(header: string): W3cTraceparent | null {
  const normalized = header.trim().toLowerCase();
  const match = TRACEPARENT_PATTERN.exec(normalized);
  if (match === null) {
    return null;
  }
  const traceId = match[1];
  const spanId = match[2];
  const flagsHex = match[3];
  if (
    traceId === undefined ||
    spanId === undefined ||
    flagsHex === undefined ||
    traceId === ALL_ZERO_TRACE_ID ||
    spanId === ALL_ZERO_SPAN_ID
  ) {
    return null;
  }
  return {
    version: "00",
    traceId,
    spanId,
    traceFlags: Number.parseInt(flagsHex, 16)
  };
}

function normalizeIncomingHeaders(
  headers: Readonly<Record<string, string | string[] | undefined>>
): Record<string, string> {
  const normalized: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (typeof value === "string") {
      normalized[key.toLowerCase()] = value;
    } else if (Array.isArray(value) && value.length > 0 && typeof value[0] === "string") {
      normalized[key.toLowerCase()] = value[0];
    }
  }
  return normalized;
}

export function createNoopTelemetryController(): TelemetryController {
  return {
    enabled: false,
    config: null,
    getAiSdkTelemetry(): undefined {
      return undefined;
    },
    startObservation<T>(
      _options: TelemetryObservationOptions,
      fn: (ctx: TelemetryObservationContext) => Promise<T>
    ): Promise<T> {
      return fn({ setOutput() {} });
    },
    getActiveTraceId(): string | undefined {
      return undefined;
    },
    injectTraceHeaders(headers?: Readonly<Record<string, string>>): Record<string, string> {
      return headers !== undefined ? { ...headers } : {};
    },
    runWithIncomingTraceContext<T>(
      _headers: Readonly<Record<string, string | string[] | undefined>>,
      fn: () => Promise<T>
    ): Promise<T> {
      return fn();
    },
    runInSpan<T>(_name: string, fn: () => Promise<T>): Promise<T> {
      return fn();
    },
    flush(): Promise<void> {
      return Promise.resolve();
    },
    shutdown(): Promise<void> {
      return Promise.resolve();
    }
  };
}

function serializeAttributeValue(value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "<unserializable>";
  }
}

function isValidTraceId(traceId: string | undefined): traceId is string {
  if (typeof traceId !== "string") {
    return false;
  }
  const normalized = traceId.trim().toLowerCase();
  return TRACE_ID_PATTERN.test(normalized) && normalized !== ALL_ZERO_TRACE_ID;
}

const LANGFUSE_TRACER_SCOPE = "langfuse-sdk";
const MAX_PROPAGATED_ATTR_CHARS = 200;

function serializeJsonMetadataValue(value: unknown): string | undefined {
  if (value === undefined || value === null || typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  try {
    const encoded = JSON.stringify(value);
    return typeof encoded === "string" ? encoded : undefined;
  } catch {
    return undefined;
  }
}

function filterPropagatableMetadata(
  metadata: Readonly<Record<string, unknown>> | undefined
): Record<string, unknown> | undefined {
  if (metadata === undefined) {
    return undefined;
  }
  const filtered: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(metadata)) {
    const encoded = serializeJsonMetadataValue(value);
    if (encoded !== undefined && encoded.length <= MAX_PROPAGATED_ATTR_CHARS) {
      filtered[key] = value;
    }
  }
  return Object.keys(filtered).length > 0 ? filtered : undefined;
}

function cloneMetadataValue(value: TelemetryMetadataValue): TelemetryMetadataValue {
  if (!Array.isArray(value)) {
    return value;
  }
  return value.slice() as TelemetryMetadataValue;
}

export async function createTelemetryController(
  env: NodeJS.ProcessEnv,
  options?: TelemetryControllerOptions
): Promise<TelemetryController> {
  const config = parseLangfuseEnv(env);
  if (config === null) {
    return createNoopTelemetryController();
  }

  const loadModule = options?.loadModule ?? defaultLoadModule;
  const onWarning = options?.onWarning ?? defaultWriteWarning;
  const serviceName = options?.serviceName ?? "sqlite-rag-bench";

  let otelApi: OtelApiModule;
  let sdkNodeMod: OtelSdkNodeModule;
  let spanProcessor: SpanProcessorLike;
  let langfuseTracing: LangfuseTracingModule | null = null;

  try {
    const [loadedApi, loadedSdkNode] = await Promise.all([
      loadModule("@opentelemetry/api"),
      loadModule("@opentelemetry/sdk-node")
    ]);
    otelApi = loadedApi as OtelApiModule;
    sdkNodeMod = loadedSdkNode as OtelSdkNodeModule;

    if (options?.spanProcessor !== undefined) {
      spanProcessor = options.spanProcessor;
    } else {
      const loadedLangfuseOtel = (await loadModule("@langfuse/otel")) as LangfuseOtelModule;
      spanProcessor = new loadedLangfuseOtel.LangfuseSpanProcessor({
        publicKey: config.publicKey,
        secretKey: config.secretKey,
        baseUrl: config.baseUrl,
        ...(options?.exportMode !== undefined ? { exportMode: options.exportMode } : {})
      });
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    onWarning(
      `[telemetry] Langfuse keys are set, but optional OpenTelemetry/Langfuse packages are unavailable (${detail}). Falling back to no-op telemetry.`
    );
    return createNoopTelemetryController();
  }

  try {
    const loadedTracing = (await loadModule("@langfuse/tracing")) as Partial<LangfuseTracingModule>;
    if (typeof loadedTracing.startActiveObservation === "function") {
      langfuseTracing = loadedTracing as LangfuseTracingModule;
    }
  } catch {
    // Optional: fall back to direct OpenTelemetry span attributes when @langfuse/tracing is absent.
    langfuseTracing = null;
  }

  const sdk = new sdkNodeMod.NodeSDK({
    serviceName,
    spanProcessors: [spanProcessor],
    instrumentations: []
  });
  sdk.start();

  const tracer =
    langfuseTracing?.getLangfuseTracer?.() ?? otelApi.trace.getTracer(LANGFUSE_TRACER_SCOPE);
  let shutdownPromise: Promise<void> | null = null;

  const getActiveTraceId = (): string | undefined => {
    const tracingId = langfuseTracing?.getActiveTraceId?.();
    if (isValidTraceId(tracingId)) {
      return tracingId.trim().toLowerCase();
    }
    const activeSpan = otelApi.trace.getSpan(otelApi.context.active());
    if (activeSpan === undefined) {
      return undefined;
    }
    const ctx = activeSpan.spanContext();
    return isValidW3cSpanContext(ctx.traceId, ctx.spanId, ctx.traceFlags)
      ? ctx.traceId.trim().toLowerCase()
      : undefined;
  };

  return {
    enabled: true,
    config,

    getAiSdkTelemetry(
      functionId: string,
      metadata?: Readonly<Record<string, TelemetryMetadataValue>>
    ): AiSdkTelemetryConfig {
      if (metadata === undefined) {
        return { isEnabled: true, functionId };
      }
      const copy: Record<string, TelemetryMetadataValue> = {};
      for (const [key, value] of Object.entries(metadata)) {
        copy[key] = cloneMetadataValue(value);
      }
      return {
        isEnabled: true,
        functionId,
        metadata: copy
      };
    },

    startObservation<T>(
      obsOptions: TelemetryObservationOptions,
      fn: (ctx: TelemetryObservationContext) => Promise<T>
    ): Promise<T> {
      const hasParentSpan = otelApi.trace.getSpan(otelApi.context.active()) !== undefined;

      if (langfuseTracing !== null) {
        const tracing = langfuseTracing;
        const runObservation = (): Promise<T> =>
          tracing.startActiveObservation(
            obsOptions.name,
            async (observation) => {
              const initialAttributes: {
                input?: unknown;
                metadata?: Record<string, unknown>;
              } = {};
              if (obsOptions.input !== undefined) {
                initialAttributes.input = obsOptions.input;
              }
              if (obsOptions.metadata !== undefined) {
                initialAttributes.metadata = { ...obsOptions.metadata };
              }
              if (Object.keys(initialAttributes).length > 0) {
                observation.update(initialAttributes);
              }

              const rawTraceId = observation.traceId ?? getActiveTraceId();
              const traceId = isValidTraceId(rawTraceId)
                ? rawTraceId.trim().toLowerCase()
                : undefined;

              const ctx: TelemetryObservationContext = {
                setOutput(output: unknown): void {
                  observation.update({ output });
                },
                ...(traceId !== undefined ? { traceId } : {})
              };

              try {
                return await fn(ctx);
              } catch (error) {
                observation.update({
                  level: "ERROR",
                  statusMessage: error instanceof Error ? error.message : String(error)
                });
                throw error;
              }
            },
            { asType: obsOptions.asType }
          );

        if (!hasParentSpan && typeof tracing.propagateAttributes === "function") {
          const propagatedMetadata = filterPropagatableMetadata(obsOptions.metadata);
          return tracing.propagateAttributes(
            {
              traceName: obsOptions.name.slice(0, MAX_PROPAGATED_ATTR_CHARS),
              ...(propagatedMetadata !== undefined ? { metadata: propagatedMetadata } : {})
            },
            runObservation
          );
        }
        return runObservation();
      }

      const initialAttrs: Record<string, string | number | boolean> = {
        "langfuse.observation.type": obsOptions.asType
      };
      const serializedInput = serializeAttributeValue(obsOptions.input);
      if (serializedInput !== undefined) {
        initialAttrs["langfuse.observation.input"] = serializedInput;
      }
      if (!hasParentSpan) {
        initialAttrs["langfuse.trace.name"] = obsOptions.name.slice(0, MAX_PROPAGATED_ATTR_CHARS);
      }
      if (obsOptions.metadata !== undefined) {
        for (const [key, value] of Object.entries(obsOptions.metadata)) {
          const encoded = serializeJsonMetadataValue(value);
          if (encoded !== undefined) {
            initialAttrs[`langfuse.observation.metadata.${key}`] = encoded;
            if (!hasParentSpan && encoded.length <= MAX_PROPAGATED_ATTR_CHARS) {
              initialAttrs[`langfuse.trace.metadata.${key}`] = encoded;
            }
          }
        }
      }

      return tracer.startActiveSpan(
        obsOptions.name,
        { kind: otelApi.SpanKind.INTERNAL, attributes: initialAttrs },
        otelApi.context.active(),
        async (span) => {
          const spanCtx = span.spanContext();
          const traceId = isValidW3cSpanContext(
            spanCtx.traceId,
            spanCtx.spanId,
            spanCtx.traceFlags
          )
            ? spanCtx.traceId.trim().toLowerCase()
            : undefined;

          const ctx: TelemetryObservationContext = {
            setOutput(output: unknown): void {
              const serializedOutput = serializeAttributeValue(output);
              if (serializedOutput !== undefined) {
                span.setAttributes({
                  "langfuse.observation.output": serializedOutput
                });
              }
            },
            ...(traceId !== undefined ? { traceId } : {})
          };

          try {
            return await fn(ctx);
          } catch (error) {
            span.recordException(error);
            span.setStatus({
              code: otelApi.SpanStatusCode.ERROR,
              message: error instanceof Error ? error.message : String(error)
            });
            throw error;
          } finally {
            span.end();
          }
        }
      );
    },

    getActiveTraceId,

    injectTraceHeaders(headers?: Readonly<Record<string, string>>): Record<string, string> {
      const carrier: Record<string, string> = headers !== undefined ? { ...headers } : {};
      const activeContext = otelApi.context.active();
      otelApi.propagation.inject(activeContext, carrier);

      if (carrier.traceparent === undefined) {
        const activeSpan = otelApi.trace.getSpan(activeContext);
        if (activeSpan !== undefined) {
          const ctx = activeSpan.spanContext();
          if (isValidW3cSpanContext(ctx.traceId, ctx.spanId, ctx.traceFlags)) {
            carrier.traceparent = formatW3cTraceparent(ctx.traceId, ctx.spanId, ctx.traceFlags);
          }
        }
      }
      return carrier;
    },

    runWithIncomingTraceContext<T>(
      headers: Readonly<Record<string, string | string[] | undefined>>,
      fn: () => Promise<T>
    ): Promise<T> {
      const carrier = normalizeIncomingHeaders(headers);
      const extracted = otelApi.propagation.extract(otelApi.ROOT_CONTEXT, carrier);
      return tracer.startActiveSpan(
        "POST /ask",
        { kind: otelApi.SpanKind.SERVER },
        extracted,
        async (span) => {
          try {
            return await fn();
          } catch (error) {
            span.recordException(error);
            span.setStatus({
              code: otelApi.SpanStatusCode.ERROR,
              message: error instanceof Error ? error.message : String(error)
            });
            throw error;
          } finally {
            span.end();
          }
        }
      );
    },

    runInSpan<T>(
      name: string,
      fn: () => Promise<T>,
      attributes?: Readonly<Record<string, string | number | boolean>>
    ): Promise<T> {
      const spanOptions: {
        kind: number;
        attributes?: Record<string, string | number | boolean>;
      } = {
        kind: otelApi.SpanKind.INTERNAL,
        ...(attributes !== undefined ? { attributes: { ...attributes } } : {})
      };
      return tracer.startActiveSpan(name, spanOptions, otelApi.context.active(), async (span) => {
        try {
          return await fn();
        } catch (error) {
          span.recordException(error);
          span.setStatus({
            code: otelApi.SpanStatusCode.ERROR,
            message: error instanceof Error ? error.message : String(error)
          });
          throw error;
        } finally {
          span.end();
        }
      });
    },

    async flush(): Promise<void> {
      await spanProcessor.forceFlush();
    },

    shutdown(): Promise<void> {
      if (shutdownPromise !== null) {
        return shutdownPromise;
      }
      shutdownPromise = sdk.shutdown().finally(() => {
        otelApi.trace.disable?.();
        otelApi.context.disable?.();
        otelApi.propagation.disable?.();
      });
      return shutdownPromise;
    }
  };
}
