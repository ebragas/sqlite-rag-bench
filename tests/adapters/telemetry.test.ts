import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { afterEach, describe, expect, it } from "vitest";
import { LiveHttpResponder } from "../../src/adapters/responder.js";
import {
  createNoopTelemetryController,
  createTelemetryController,
  DEFAULT_LANGFUSE_BASE_URL,
  formatW3cTraceparent,
  parseLangfuseEnv,
  parseW3cTraceparent,
  type TelemetryController
} from "../../src/adapters/telemetry.js";

const openServers: Server[] = [];
const activeControllers: TelemetryController[] = [];

afterEach(async () => {
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }
  while (activeControllers.length > 0) {
    const controller = activeControllers.pop();
    if (controller !== undefined) {
      await controller.shutdown();
    }
  }
});

describe("parseLangfuseEnv", () => {
  it("returns null when either Langfuse key is absent or blank", () => {
    expect(parseLangfuseEnv({})).toBeNull();
    expect(parseLangfuseEnv({ LANGFUSE_PUBLIC_KEY: "pk-test" })).toBeNull();
    expect(parseLangfuseEnv({ LANGFUSE_SECRET_KEY: "sk-test" })).toBeNull();
    expect(
      parseLangfuseEnv({ LANGFUSE_PUBLIC_KEY: "   ", LANGFUSE_SECRET_KEY: "sk-test" })
    ).toBeNull();
  });

  it("parses trimmed keys with default or custom base URL", () => {
    expect(
      parseLangfuseEnv({
        LANGFUSE_PUBLIC_KEY: " pk-123 ",
        LANGFUSE_SECRET_KEY: " sk-456 "
      })
    ).toEqual({
      publicKey: "pk-123",
      secretKey: "sk-456",
      baseUrl: DEFAULT_LANGFUSE_BASE_URL
    });

    expect(
      parseLangfuseEnv({
        LANGFUSE_PUBLIC_KEY: "pk-123",
        LANGFUSE_SECRET_KEY: "sk-456",
        LANGFUSE_BASE_URL: "https://us.cloud.langfuse.com"
      })
    ).toEqual({
      publicKey: "pk-123",
      secretKey: "sk-456",
      baseUrl: "https://us.cloud.langfuse.com"
    });

    expect(
      parseLangfuseEnv({
        LANGFUSE_PUBLIC_KEY: "pk-123",
        LANGFUSE_SECRET_KEY: "sk-456",
        LANGFUSE_HOST: "https://eu.cloud.langfuse.com"
      })
    ).toEqual({
      publicKey: "pk-123",
      secretKey: "sk-456",
      baseUrl: "https://eu.cloud.langfuse.com"
    });
  });
});

describe("W3C traceparent helpers", () => {
  it("formats and parses valid W3C traceparent headers", () => {
    const traceId = "4bf92f3577b34da6a3ce929d0e0e4736";
    const spanId = "00f067aa0ba902b7";
    const header = formatW3cTraceparent(traceId, spanId, 1);

    expect(header).toBe(`00-${traceId}-${spanId}-01`);
    expect(parseW3cTraceparent(header)).toEqual({
      version: "00",
      traceId,
      spanId,
      traceFlags: 1
    });
    expect(parseW3cTraceparent(header.toUpperCase())).toEqual({
      version: "00",
      traceId,
      spanId,
      traceFlags: 1
    });
  });

  it("rejects malformed or all-zero traceparent identifiers", () => {
    expect(() => formatW3cTraceparent("00000000000000000000000000000000", "00f067aa0ba902b7")).toThrow(
      RangeError
    );
    expect(() => formatW3cTraceparent("4bf92f3577b34da6a3ce929d0e0e4736", "0000000000000000")).toThrow(
      RangeError
    );
    expect(() => formatW3cTraceparent("not-hex", "00f067aa0ba902b7")).toThrow(RangeError);
    expect(() => formatW3cTraceparent("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", 256)).toThrow(
      RangeError
    );

    expect(parseW3cTraceparent("invalid")).toBeNull();
    expect(
      parseW3cTraceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01")
    ).toBeNull();
    expect(
      parseW3cTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01")
    ).toBeNull();
  });
});

describe("createTelemetryController", () => {
  it("returns strict zero-import no-op controller when Langfuse keys are absent", async () => {
    const importedModules: string[] = [];
    const controller = await createTelemetryController(
      {},
      {
        loadModule: (specifier) => {
          importedModules.push(specifier);
          return Promise.reject(new Error(`Unexpected import: ${specifier}`));
        }
      }
    );

    expect(importedModules).toEqual([]);
    expect(controller.enabled).toBe(false);
    expect(controller.config).toBeNull();
    expect(controller.getAiSdkTelemetry("brain.synthesize", { model: "gemini-3.8-flash" })).toBeUndefined();
    expect(controller.injectTraceHeaders()).toEqual({});
    expect(controller.injectTraceHeaders({ "x-custom": "keep" })).toEqual({ "x-custom": "keep" });

    const value = await controller.runWithIncomingTraceContext(
      { traceparent: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01" },
      () => Promise.resolve("ok")
    );
    expect(value).toBe("ok");

    const spanValue = await controller.runInSpan("test-span", () => Promise.resolve(42));
    expect(spanValue).toBe(42);
    expect(controller.getActiveTraceId()).toBeUndefined();

    const obsValue = await controller.startObservation(
      { name: "answer-question", asType: "agent", input: { question: "hi" } },
      (ctx) => {
        expect(ctx.traceId).toBeUndefined();
        ctx.setOutput({ ok: true });
        return Promise.resolve("obs-ok");
      }
    );
    expect(obsValue).toBe("obs-ok");

    await controller.flush();
    await controller.shutdown();

    const directNoop = createNoopTelemetryController();
    expect(directNoop.enabled).toBe(false);
    expect(directNoop.getAiSdkTelemetry("noop")).toBeUndefined();
  });

  it("warns and degrades cleanly to no-op when devDependencies are omitted", async () => {
    const warnings: string[] = [];
    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      {
        loadModule: (specifier) =>
          Promise.reject(new Error(`Cannot find package '${specifier}'`)),
        onWarning: (message) => {
          warnings.push(message);
        }
      }
    );

    expect(controller.enabled).toBe(false);
    expect(controller.getAiSdkTelemetry("brain.synthesize")).toBeUndefined();
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("Falling back to no-op telemetry");
  });

  it("enables AI SDK telemetry and propagates W3C traceparent from LiveHttpResponder to HTTP server", async () => {
    const exporter = new InMemorySpanExporter();
    const spanProcessor = new SimpleSpanProcessor(exporter);

    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-local-test",
        LANGFUSE_SECRET_KEY: "sk-local-test",
        LANGFUSE_BASE_URL: "http://127.0.0.1:3999"
      },
      {
        serviceName: "srb-telemetry-test",
        spanProcessor
      }
    );
    activeControllers.push(controller);

    expect(controller.enabled).toBe(true);
    expect(controller.config).toEqual({
      publicKey: "pk-local-test",
      secretKey: "sk-local-test",
      baseUrl: "http://127.0.0.1:3999"
    });

    expect(
      controller.getAiSdkTelemetry("brain.synthesize", {
        model: "gemini-3.8-flash",
        retries: 0
      })
    ).toEqual({
      isEnabled: true,
      functionId: "brain.synthesize",
      metadata: {
        model: "gemini-3.8-flash",
        retries: 0
      }
    });

    const receivedTraceparents: Array<string | undefined> = [];
    const server = createServer((req, res) => {
      const headerValue = req.headers.traceparent;
      receivedTraceparents.push(typeof headerValue === "string" ? headerValue : undefined);

      void controller
        .runWithIncomingTraceContext(req.headers, async () => {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(
            JSON.stringify({
              answer: "Traced answer.",
              document_ids: ["dsid_1"]
            })
          );
        })
        .catch(() => {
          res.writeHead(500);
          res.end();
        });
    });
    openServers.push(server);
    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const port = (server.address() as AddressInfo).port;
    const responder = new LiveHttpResponder({
      baseUrl: `http://127.0.0.1:${port}`,
      modelId: "gemini-3.8-flash"
    });

    let injectedHeader: string | undefined;
    const askResult = await controller.runInSpan(
      "experiment-item-run",
      async () => {
        const headers = controller.injectTraceHeaders();
        injectedHeader = headers.traceparent;
        return responder.answer("Where is the billing runbook?", { headers });
      },
      { questionId: "q_trace_1" }
    );

    expect(askResult.answer).toBe("Traced answer.");
    expect(askResult.documentIds).toEqual(["dsid_1"]);
    expect(injectedHeader).toBeDefined();
    expect(receivedTraceparents).toEqual([injectedHeader]);

    const parsedHeader = parseW3cTraceparent(injectedHeader ?? "");
    expect(parsedHeader).not.toBeNull();

    await controller.flush();
    const spans = exporter.getFinishedSpans();
    const parentSpan = spans.find((span) => span.name === "experiment-item-run");
    const serverSpan = spans.find((span) => span.name === "POST /ask");

    expect(parentSpan).toBeDefined();
    expect(serverSpan).toBeDefined();
    expect(parsedHeader?.traceId).toBe(parentSpan?.spanContext().traceId);
    expect(parsedHeader?.spanId).toBe(parentSpan?.spanContext().spanId);
    expect(serverSpan?.spanContext().traceId).toBe(parentSpan?.spanContext().traceId);
    expect(serverSpan?.parentSpanContext?.spanId).toBe(parentSpan?.spanContext().spanId);
  });

  it("skips traceparent fallback when active span context has all-zero or malformed IDs", async () => {
    const fakeSpanProcessor = {
      forceFlush: () => Promise.resolve(),
      shutdown: () => Promise.resolve()
    };
    const fakeOtelApi = {
      ROOT_CONTEXT: {},
      SpanKind: { SERVER: 1, INTERNAL: 0 },
      SpanStatusCode: { ERROR: 2 },
      context: {
        active: () => ({}),
        with: <T>(_ctx: unknown, fn: () => T) => fn()
      },
      propagation: {
        extract: () => ({}),
        inject: () => {}
      },
      trace: {
        getTracer: () => ({
          startActiveSpan: <T>(
            _name: string,
            _options: unknown,
            _ctx: unknown,
            fn: (span: unknown) => Promise<T>
          ) => fn({})
        }),
        getSpan: () => ({
          spanContext: () => ({
            traceId: "00000000000000000000000000000000",
            spanId: "0000000000000000",
            traceFlags: 1
          })
        })
      }
    };
    const fakeSdkNode = {
      NodeSDK: class {
        public start(): void {}
        public shutdown(): Promise<void> {
          return Promise.resolve();
        }
      }
    };

    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-test",
        LANGFUSE_SECRET_KEY: "sk-test"
      },
      {
        spanProcessor: fakeSpanProcessor,
        loadModule: (specifier) => {
          if (specifier === "@opentelemetry/api") {
            return Promise.resolve(fakeOtelApi);
          }
          if (specifier === "@opentelemetry/sdk-node") {
            return Promise.resolve(fakeSdkNode);
          }
          return Promise.reject(new Error(`Unexpected module: ${specifier}`));
        }
      }
    );
    activeControllers.push(controller);

    expect(controller.injectTraceHeaders({ "x-test": "1" })).toEqual({ "x-test": "1" });
  });

  it("emits typed nested Langfuse observations with clean input/output and active traceId", async () => {
    const exporter = new InMemorySpanExporter();
    const spanProcessor = new LangfuseSpanProcessor({
      exporter,
      exportMode: "immediate",
      mediaUploadEnabled: false
    });

    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-obs-test",
        LANGFUSE_SECRET_KEY: "sk-obs-test",
        LANGFUSE_BASE_URL: "http://127.0.0.1:3999"
      },
      {
        serviceName: "srb-telemetry-obs-test",
        spanProcessor
      }
    );
    activeControllers.push(controller);

    let rootTraceId: string | undefined;
    let activeTraceIdInside: string | undefined;

    await controller.startObservation(
      {
        name: "answer-question",
        asType: "agent",
        input: { question: "What is SSO?" },
        metadata: { profile: "l3-graph" }
      },
      async (rootCtx) => {
        rootTraceId = rootCtx.traceId;
        activeTraceIdInside = controller.getActiveTraceId();

        await controller.startObservation(
          {
            name: "retrieve-evidence",
            asType: "retriever",
            input: { query: "SSO", topK: 15 }
          },
          (retCtx) => {
            retCtx.setOutput({ hitCount: 2, topDocIds: ["dsid_1", "dsid_2"] });
            return Promise.resolve();
          }
        );

        await controller.startObservation(
          {
            name: "fetch-docs",
            asType: "tool",
            input: { doc_ids: ["dsid_1"] }
          },
          (toolCtx) => {
            toolCtx.setOutput({ fetchedCount: 1 });
            return Promise.resolve();
          }
        );

        rootCtx.setOutput({ answer: "SSO is Single Sign-On.", abstained: false });
      }
    );

    await controller.flush();
    expect(rootTraceId).toMatch(/^[0-9a-f]{32}$/);
    expect(activeTraceIdInside).toBe(rootTraceId);

    const spans = exporter.getFinishedSpans();
    const agentSpan = spans.find((span) => span.name === "answer-question");
    const retrieverSpan = spans.find((span) => span.name === "retrieve-evidence");
    const toolSpan = spans.find((span) => span.name === "fetch-docs");

    expect(agentSpan).toBeDefined();
    expect(retrieverSpan).toBeDefined();
    expect(toolSpan).toBeDefined();

    expect(agentSpan?.attributes["langfuse.observation.type"]).toBe("agent");
    expect(agentSpan?.attributes["langfuse.observation.input"]).toBe(
      JSON.stringify({ question: "What is SSO?" })
    );
    expect(agentSpan?.attributes["langfuse.observation.output"]).toBe(
      JSON.stringify({ answer: "SSO is Single Sign-On.", abstained: false })
    );
    expect(agentSpan?.attributes["langfuse.trace.name"]).toBe("answer-question");
    expect(agentSpan?.attributes["langfuse.trace.metadata.profile"]).toBe('"l3-graph"');

    expect(retrieverSpan?.attributes["langfuse.observation.type"]).toBe("retriever");
    expect(retrieverSpan?.parentSpanContext?.spanId).toBe(agentSpan?.spanContext().spanId);
    expect(retrieverSpan?.spanContext().traceId).toBe(rootTraceId);

    expect(toolSpan?.attributes["langfuse.observation.type"]).toBe("tool");
    expect(toolSpan?.parentSpanContext?.spanId).toBe(agentSpan?.spanContext().spanId);
    expect(toolSpan?.spanContext().traceId).toBe(rootTraceId);
  });

  it("passes exportMode through to LangfuseSpanProcessor when configured", async () => {
    const capturedParams: Array<Record<string, unknown>> = [];
    const fakeOtelApi = {
      ROOT_CONTEXT: {},
      SpanKind: { SERVER: 1, INTERNAL: 0 },
      SpanStatusCode: { ERROR: 2 },
      context: { active: () => ({}), with: <T>(_ctx: unknown, fn: () => T) => fn() },
      propagation: { extract: () => ({}), inject: () => {} },
      trace: {
        getTracer: () => ({
          startActiveSpan: <T>(
            _name: string,
            _options: unknown,
            _ctx: unknown,
            fn: (span: unknown) => Promise<T>
          ) => fn({})
        }),
        getSpan: () => undefined
      }
    };
    const fakeSdkNode = {
      NodeSDK: class {
        public start(): void {}
        public shutdown(): Promise<void> {
          return Promise.resolve();
        }
      }
    };
    const fakeLangfuseOtel = {
      LangfuseSpanProcessor: class {
        public constructor(params: Record<string, unknown>) {
          capturedParams.push(params);
        }
        public forceFlush(): Promise<void> {
          return Promise.resolve();
        }
        public shutdown(): Promise<void> {
          return Promise.resolve();
        }
      }
    };

    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-mode",
        LANGFUSE_SECRET_KEY: "sk-mode",
        LANGFUSE_BASE_URL: "https://us.cloud.langfuse.com"
      },
      {
        exportMode: "immediate",
        loadModule: (specifier) => {
          if (specifier === "@opentelemetry/api") {
            return Promise.resolve(fakeOtelApi);
          }
          if (specifier === "@opentelemetry/sdk-node") {
            return Promise.resolve(fakeSdkNode);
          }
          if (specifier === "@langfuse/otel") {
            return Promise.resolve(fakeLangfuseOtel);
          }
          return Promise.reject(new Error(`Optional module absent: ${specifier}`));
        }
      }
    );
    activeControllers.push(controller);

    expect(capturedParams).toEqual([
      {
        publicKey: "pk-mode",
        secretKey: "sk-mode",
        baseUrl: "https://us.cloud.langfuse.com",
        exportMode: "immediate"
      }
    ]);
  });

  it("exports fallback spans through LangfuseSpanProcessor and enforces 200-char JSON limit on propagated metadata", async () => {
    const exporter = new InMemorySpanExporter();
    const spanProcessor = new LangfuseSpanProcessor({
      exporter,
      exportMode: "immediate",
      mediaUploadEnabled: false
    });

    const controller = await createTelemetryController(
      {
        LANGFUSE_PUBLIC_KEY: "pk-fallback",
        LANGFUSE_SECRET_KEY: "sk-fallback",
        LANGFUSE_BASE_URL: "http://127.0.0.1:3999"
      },
      {
        serviceName: "srb-fallback-test",
        spanProcessor,
        loadModule: (specifier) => {
          if (specifier === "@langfuse/tracing") {
            return Promise.reject(new Error("Simulated missing @langfuse/tracing"));
          }
          return import(specifier);
        }
      }
    );
    activeControllers.push(controller);

    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const borderline199 = "a".repeat(199); // JSON-encoded length is 201 (> 200)
    const fitting198 = "b".repeat(198); // JSON-encoded length is 200 (<= 200)

    await controller.startObservation(
      {
        name: "fallback-agent",
        asType: "agent",
        input: { q: "ping" },
        metadata: {
          fitting: fitting198,
          borderline: borderline199,
          nullVal: null,
          circular
        }
      },
      (ctx) => {
        ctx.setOutput({ ok: true });
        return Promise.resolve();
      }
    );

    await controller.flush();
    const spans = exporter.getFinishedSpans();
    const fallbackSpan = spans.find((span) => span.name === "fallback-agent");
    expect(fallbackSpan).toBeDefined();
    expect(fallbackSpan?.instrumentationScope.name).toBe("langfuse-sdk");
    expect(fallbackSpan?.attributes["langfuse.trace.metadata.fitting"]).toBe(
      JSON.stringify(fitting198)
    );
    expect(fallbackSpan?.attributes["langfuse.trace.metadata.borderline"]).toBeUndefined();
    expect(fallbackSpan?.attributes["langfuse.trace.metadata.nullVal"]).toBeUndefined();
    expect(fallbackSpan?.attributes["langfuse.trace.metadata.circular"]).toBeUndefined();
    expect(fallbackSpan?.attributes["langfuse.observation.metadata.borderline"]).toBe(
      JSON.stringify(borderline199)
    );
    expect(fallbackSpan?.attributes["langfuse.observation.metadata.nullVal"]).toBeUndefined();
    expect(fallbackSpan?.attributes["langfuse.observation.metadata.circular"]).toBeUndefined();
  });
});
