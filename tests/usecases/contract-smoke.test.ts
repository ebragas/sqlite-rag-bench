import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { LiveHttpResponder } from "../../src/adapters/responder.js";
import { ContractError } from "../../src/models/errors.js";
import { runContractSmokeSuite } from "../../src/usecases/contract-smoke.js";

const openServers: Server[] = [];

async function startTestServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void
): Promise<{ server: Server; baseUrl: string; port: number }> {
  const server = createServer(handler);
  openServers.push(server);

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });

  const address = server.address() as AddressInfo;
  return {
    server,
    port: address.port,
    baseUrl: `http://127.0.0.1:${address.port}`
  };
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

afterEach(async () => {
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }
});

describe("runContractSmokeSuite", () => {
  it("passes on a compliant HTTP server and forwards traceparent via LiveHttpResponder", async () => {
    const receivedTraceparents: string[] = [];

    const { baseUrl } = await startTestServer((req, res) => {
      if (req.method === "GET" && req.url === "/health") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ status: "ok" }));
        return;
      }

      if (req.method === "POST" && req.url === "/ask") {
        const traceparent = req.headers.traceparent;
        if (typeof traceparent === "string") {
          receivedTraceparents.push(traceparent);
        }
        void readRequestBody(req).then((raw) => {
          try {
            const body = JSON.parse(raw) as { question?: unknown };
            if (typeof body.question !== "string" || body.question.trim().length === 0) {
              res.writeHead(400, { "content-type": "application/json" });
              res.end(JSON.stringify({ error: "missing question" }));
              return;
            }
            res.writeHead(200, { "content-type": "application/json" });
            res.end(
              JSON.stringify({
                answer: `Echo: ${body.question}`,
                document_ids: ["dsid_smoke_1"]
              })
            );
          } catch {
            res.writeHead(400, { "content-type": "application/json" });
            res.end(JSON.stringify({ error: "invalid json" }));
          }
        });
        return;
      }

      res.writeHead(404);
      res.end();
    });

    const report = await runContractSmokeSuite({ baseUrl });
    expect(report.passed).toBe(true);
    expect(report.checks.map((check) => check.name)).toEqual([
      "health_ready",
      "ask_valid",
      "ask_malformed",
      "ask_concurrency_4"
    ]);
    expect(report.checks.every((check) => check.passed)).toBe(true);

    const responder = new LiveHttpResponder({ baseUrl, modelId: "gemini-3.8-flash" });
    const traceValue = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    const result = await responder.answer("Check trace header", {
      headers: { traceparent: traceValue }
    });
    expect(result.answer).toBe("Echo: Check trace header");
    expect(result.documentIds).toEqual(["dsid_smoke_1"]);
    expect(result.responseBytes).toBeGreaterThan(0);
    expect(receivedTraceparents).toEqual([traceValue]);
  });

  it("fails loudly when /ask returns a malformed schema payload", async () => {
    const { baseUrl } = await startTestServer((req, res) => {
      if (req.method === "GET" && req.url === "/health") {
        res.writeHead(200);
        res.end("ok");
        return;
      }
      if (req.method === "POST" && req.url === "/ask") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ answer: "", document_ids: "not-an-array" }));
        return;
      }
      res.writeHead(404);
      res.end();
    });

    await expect(runContractSmokeSuite({ baseUrl })).rejects.toThrow(
      /invalid response schema/i
    );
  });

  it("fails loudly when /ask response exceeds 1 MiB limit", async () => {
    const oversizedAnswer = "x".repeat(1024 * 1024 + 64);
    const { baseUrl } = await startTestServer((req, res) => {
      if (req.method === "GET" && req.url === "/health") {
        res.writeHead(200);
        res.end("ok");
        return;
      }
      if (req.method === "POST" && req.url === "/ask") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ answer: oversizedAnswer, document_ids: [] }));
        return;
      }
      res.writeHead(404);
      res.end();
    });

    await expect(runContractSmokeSuite({ baseUrl })).rejects.toThrow(/exceeds 1 MiB/i);

    const responder = new LiveHttpResponder({ baseUrl, modelId: "gemini-3.8-flash" });
    await expect(responder.answer("Oversized test")).rejects.toThrow(ContractError);
    await expect(responder.answer("Oversized test")).rejects.toThrow(/exceeds limit/i);
  });

  it("verifies spawned server 503 -> 200 readiness transition and SIGTERM clean exit", async () => {
    const probeServer = await startTestServer((_req, res) => {
      res.writeHead(200);
      res.end();
    });
    const freePort = probeServer.port;
    await new Promise<void>((resolve) => {
      probeServer.server.close(() => resolve());
    });
    openServers.pop();

    const inlineServerScript = `
      const http = require("node:http");
      let healthCalls = 0;
      const server = http.createServer((req, res) => {
        if (req.method === "GET" && req.url === "/health") {
          healthCalls += 1;
          if (healthCalls === 1) {
            res.writeHead(503);
            res.end("loading");
            return;
          }
          res.writeHead(200);
          res.end("ready");
          return;
        }
        if (req.method === "POST" && req.url === "/ask") {
          let body = "";
          req.on("data", (c) => { body += c; });
          req.on("end", () => {
            try {
              const parsed = JSON.parse(body);
              if (!parsed || typeof parsed.question !== "string") {
                res.writeHead(400);
                res.end("bad");
                return;
              }
              res.writeHead(200, { "content-type": "application/json" });
              res.end(JSON.stringify({ answer: "ok", document_ids: [] }));
            } catch {
              res.writeHead(400);
              res.end("bad json");
            }
          });
          return;
        }
        res.writeHead(404);
        res.end();
      });
      server.listen(${freePort}, "127.0.0.1");
      process.on("SIGTERM", () => {
        server.close(() => process.exit(0));
      });
    `;

    const report = await runContractSmokeSuite({
      baseUrl: `http://127.0.0.1:${freePort}`,
      spawn: {
        command: process.execPath,
        args: ["-e", inlineServerScript],
        startupTimeoutMs: 5000,
        shutdownTimeoutMs: 5000,
        pollIntervalMs: 25
      }
    });

    expect(report.passed).toBe(true);
    expect(report.observedNotReadyStatus).toBe(503);
    expect(report.checks.some((check) => check.name === "sigterm_shutdown" && check.passed)).toBe(
      true
    );
  });

  it("handles a nonexistent spawn binary cleanly without unhandled error crashes", async () => {
    const nonexistentBinary = "/nonexistent-srb-smoke-binary-12345";
    const baseUrl = "http://127.0.0.1:1";

    const report = await runContractSmokeSuite({
      baseUrl,
      throwOnFailure: false,
      spawn: {
        command: nonexistentBinary,
        startupTimeoutMs: 2000,
        shutdownTimeoutMs: 2000,
        pollIntervalMs: 25
      }
    });

    expect(report.passed).toBe(false);
    expect(report.checks.length).toBe(1);
    expect(report.checks[0]?.name).toBe("startup_readiness");
    expect(report.checks[0]?.passed).toBe(false);
    expect(report.checks[0]?.detail).toMatch(/failed to spawn process/i);

    await expect(
      runContractSmokeSuite({
        baseUrl,
        spawn: {
          command: nonexistentBinary,
          startupTimeoutMs: 2000,
          shutdownTimeoutMs: 2000,
          pollIntervalMs: 25
        }
      })
    ).rejects.toThrow(ContractError);
  });

  it("drains more than 64 KiB of stdout and stderr from a spawned child without deadlocking", async () => {
    const probeServer = await startTestServer((_req, res) => {
      res.writeHead(200);
      res.end();
    });
    const freePort = probeServer.port;
    await new Promise<void>((resolve) => {
      probeServer.server.close(() => resolve());
    });
    openServers.pop();

    const noisyServerScript = `
      const http = require("node:http");
      const chunk = "L".repeat(1024) + "\\n";
      for (let i = 0; i < 96; i += 1) {
        process.stdout.write(chunk);
        process.stderr.write(chunk);
      }
      const server = http.createServer((req, res) => {
        if (req.method === "GET" && req.url === "/health") {
          res.writeHead(200);
          res.end("ready");
          return;
        }
        if (req.method === "POST" && req.url === "/ask") {
          let body = "";
          req.on("data", (c) => { body += c; });
          req.on("end", () => {
            try {
              const parsed = JSON.parse(body);
              if (!parsed || typeof parsed.question !== "string") {
                res.writeHead(400);
                res.end("bad");
                return;
              }
              process.stdout.write("A".repeat(4096) + "\\n");
              process.stderr.write("E".repeat(4096) + "\\n");
              res.writeHead(200, { "content-type": "application/json" });
              res.end(JSON.stringify({ answer: "ok", document_ids: [] }));
            } catch {
              res.writeHead(400);
              res.end("bad json");
            }
          });
          return;
        }
        res.writeHead(404);
        res.end();
      });
      server.listen(${freePort}, "127.0.0.1");
      process.on("SIGTERM", () => {
        server.close(() => process.exit(0));
      });
    `;

    const report = await runContractSmokeSuite({
      baseUrl: `http://127.0.0.1:${freePort}`,
      spawn: {
        command: process.execPath,
        args: ["-e", noisyServerScript],
        startupTimeoutMs: 5000,
        shutdownTimeoutMs: 5000,
        pollIntervalMs: 25
      }
    });

    expect(report.passed).toBe(true);
    expect(report.checks.every((check) => check.passed)).toBe(true);
  });
});
