import { execFileSync, spawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { ConfigError, ContractError } from "../src/models/errors.js";

export const DOCKER_IMAGE = "node:20-slim";
export const DEFAULT_PORT = 8080;
export const DEFAULT_MAX_RSS_MIB = 4096;
export const DEFAULT_STARTUP_TIMEOUT_MS = 120_000;
export const SHUTDOWN_TIMEOUT_MS = 10_000;
export const MAX_ASK_RESPONSE_BYTES = 1024 * 1024;
export const DUMMY_GEMINI_API_KEY = "dummy-docker-smoke-key";
export const REPORT_START_MARKER = "__VERIFY_DOCKER_REPORT_START__";
export const REPORT_END_MARKER = "__VERIFY_DOCKER_REPORT_END__";

export const DEFAULT_CONTAINER_PATH =
  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
export const DEFAULT_CONTAINER_HOME = "/tmp/serve-home";
export const DEFAULT_CONTAINER_TMPDIR = "/tmp/serve-tmp";
export const DEFAULT_CONTAINER_LANG = "C.UTF-8";

export const CONTRACT_ENV_KEYS = [
  "CORPUS_DIR",
  "PORT",
  "GEMINI_API_KEY",
  "PATH",
  "HOME",
  "TMPDIR",
  "LANG"
] as const;

export type ContractEnvKey = (typeof CONTRACT_ENV_KEYS)[number];

export const STAGING_EXCLUDED_PATHS = [
  "node_modules",
  "dist",
  ".env",
  ".cache",
  ".agents",
  ".claude"
] as const;

export const FORBIDDEN_BUILD_TOOLS = ["gcc", "g++", "make", "python3"] as const;

export interface VerifyDockerOptions {
  readonly corpusDir: string | null;
  readonly port: number;
  readonly maxRssMiB: number;
  readonly startupTimeoutMs: number;
  readonly liveAsk: boolean;
}

export interface MinimalContractEnvOptions {
  readonly corpusDir: string;
  readonly port: number;
  readonly geminiApiKey: string;
  readonly path?: string;
  readonly home?: string;
  readonly tmpDir?: string;
  readonly lang?: string;
}

export interface FixtureCorpusFile {
  readonly relativePath: string;
  readonly payload: Readonly<Record<string, unknown>>;
}

const ContainerCheckSchema = z.object({
  name: z.string().min(1),
  passed: z.boolean(),
  detail: z.string()
});

const ContainerReportSchema = z.object({
  passed: z.boolean(),
  image: z.string().min(1),
  port: z.number().int().min(1).max(65535),
  liveAsk: z.boolean(),
  buildToolsPresent: z.array(z.string()),
  serveEnvKeys: z.array(z.string()),
  observedPreReadyStates: z.array(z.string()),
  observedNotReadyStatus: z.number().int().nullable(),
  coldStartMs: z.number().nonnegative().nullable(),
  peakRssKiB: z.number().int().nonnegative().nullable(),
  peakRssMiB: z.number().nonnegative().nullable(),
  maxRssMiB: z.number().positive(),
  sigtermExitCode: z.number().int().nullable(),
  sigtermSignal: z.string().nullable(),
  shutdownMs: z.number().nonnegative().nullable(),
  leakedPids: z.array(z.number().int().positive()),
  checks: z.array(ContainerCheckSchema),
  serveLogTail: z.string()
});

export type ContainerVerificationReport = z.infer<typeof ContainerReportSchema>;

function requireFlagValue(
  argv: readonly string[],
  index: number,
  flag: string
): string {
  const value = argv[index + 1];
  if (value === undefined || value.trim().length === 0) {
    throw new ConfigError(`${flag} requires a non-empty value`);
  }
  return value;
}

function parsePositiveIntegerFlag(rawValue: string, flag: string): number {
  const parsed = Number(rawValue);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new ConfigError(
      `${flag} requires a positive integer, got: ${rawValue}`
    );
  }
  return parsed;
}

function parsePositiveNumberFlag(rawValue: string, flag: string): number {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new ConfigError(
      `${flag} requires a positive number, got: ${rawValue}`
    );
  }
  return parsed;
}

export function parseVerifyDockerArgs(
  argv: readonly string[],
  cwd = process.cwd()
): VerifyDockerOptions {
  let corpusDir: string | null = null;
  let port = DEFAULT_PORT;
  let maxRssMiB = DEFAULT_MAX_RSS_MIB;
  let startupTimeoutMs = DEFAULT_STARTUP_TIMEOUT_MS;
  let liveAsk = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--corpus-dir") {
      const value = requireFlagValue(argv, i, "--corpus-dir");
      corpusDir = resolve(cwd, value);
      i += 1;
    } else if (arg === "--port") {
      const value = requireFlagValue(argv, i, "--port");
      const parsedPort = parsePositiveIntegerFlag(value, "--port");
      if (parsedPort > 65535) {
        throw new ConfigError(`--port must be between 1 and 65535, got: ${value}`);
      }
      port = parsedPort;
      i += 1;
    } else if (arg === "--max-rss-mib") {
      const value = requireFlagValue(argv, i, "--max-rss-mib");
      maxRssMiB = parsePositiveNumberFlag(value, "--max-rss-mib");
      i += 1;
    } else if (arg === "--startup-timeout-ms") {
      const value = requireFlagValue(argv, i, "--startup-timeout-ms");
      startupTimeoutMs = parsePositiveIntegerFlag(value, "--startup-timeout-ms");
      i += 1;
    } else if (arg === "--live-ask") {
      liveAsk = true;
    } else {
      throw new ConfigError(`Unknown argument: ${arg ?? ""}`);
    }
  }

  return {
    corpusDir,
    port,
    maxRssMiB,
    startupTimeoutMs,
    liveAsk
  };
}

export function parseVmHwmKiB(procStatusText: string): number {
  const match = /^VmHWM:\s*(\d+)\s*kB\s*$/im.exec(procStatusText);
  if (match === null || match[1] === undefined) {
    throw new ContractError("VmHWM field not found in /proc/<pid>/status");
  }
  const value = Number(match[1]);
  if (!Number.isInteger(value) || value < 0) {
    throw new ContractError(`Invalid VmHWM value: ${match[1]}`);
  }
  return value;
}

export function kibToMiB(kib: number): number {
  if (!Number.isFinite(kib) || kib < 0) {
    throw new ContractError(`Invalid KiB value: ${String(kib)}`);
  }
  return Number((kib / 1024).toFixed(2));
}

export function buildMinimalContractEnv(
  options: MinimalContractEnvOptions
): Record<ContractEnvKey, string> {
  if (options.corpusDir.trim().length === 0) {
    throw new ConfigError("CORPUS_DIR must be a non-empty string");
  }
  if (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535) {
    throw new ConfigError(
      `PORT must be an integer between 1 and 65535, got: ${String(options.port)}`
    );
  }
  if (options.geminiApiKey.trim().length === 0) {
    throw new ConfigError("GEMINI_API_KEY must be a non-empty string");
  }

  return {
    CORPUS_DIR: options.corpusDir,
    PORT: String(options.port),
    GEMINI_API_KEY: options.geminiApiKey,
    PATH: options.path ?? DEFAULT_CONTAINER_PATH,
    HOME: options.home ?? DEFAULT_CONTAINER_HOME,
    TMPDIR: options.tmpDir ?? DEFAULT_CONTAINER_TMPDIR,
    LANG: options.lang ?? DEFAULT_CONTAINER_LANG
  };
}

export function buildFixtureCorpusDocuments(): readonly FixtureCorpusFile[] {
  return [
    {
      relativePath: "slack/threads/release-cutover.json",
      payload: {
        dataset_doc_uuid: "dsid_docker_fixture_slack_01",
        title_field_name: "channel",
        content_field_names: ["messages"],
        channel: "#release-ops",
        status: "active",
        created_at: "2025-03-10T14:00:00Z",
        messages: [
          {
            user: "priya",
            ts: "2025-03-10T14:01:00Z",
            text: "Canary deployment for Nimbus gateway completed with zero 5xx errors."
          },
          {
            user: "marcus",
            ts: "2025-03-10T14:05:00Z",
            text: "Cutover window closed at 14:05 UTC after p95 latency stayed at 42ms."
          }
        ]
      }
    },
    {
      relativePath: "confluence/architecture/billing-db.json",
      payload: {
        dataset_doc_uuid: "dsid_docker_fixture_confluence_02",
        title_field_name: "title",
        content_field_names: ["body"],
        title: "Billing Ledger Postgres 16 Runbook",
        author: "dana",
        status: "approved",
        updated_at: "2025-02-18T09:30:00Z",
        body: "The Billing Ledger cluster runs on Postgres 16 with synchronous standby replication and nightly checksum verification."
      }
    },
    {
      relativePath: "jira/platform/2025/OPS-404.json",
      payload: {
        dataset_doc_uuid: "dsid_docker_fixture_jira_03",
        title_field_name: "summary",
        content_field_names: ["description"],
        key: "OPS-404",
        summary: "Rotate Redis cache credentials in staging",
        reporter: "alex",
        priority: "P2",
        status: "Done",
        description: "Rotated staging Redis credentials on 2025-03-01 and verified zero cache miss spikes."
      }
    },
    {
      relativePath: "email/inbox/vendor-contract.json",
      payload: {
        dataset_doc_uuid: "dsid_docker_fixture_email_04",
        title_field_name: "subject",
        content_field_names: ["body"],
        subject: "Nimbus Gateway Annual SLA Renewal",
        sender: "legal@redwood.example",
        recipients: ["procurement@redwood.example"],
        created_at: "2025-01-22T16:15:00Z",
        body: "The Nimbus Gateway SLA renewal locks uptime at 99.95% through December 2026 with a 30-day termination notice."
      }
    }
  ];
}

export function parseContainerReportOutput(
  stdout: string
): ContainerVerificationReport {
  const startIndex = stdout.lastIndexOf(REPORT_START_MARKER);
  const endIndex = stdout.lastIndexOf(REPORT_END_MARKER);
  if (startIndex === -1 || endIndex === -1 || endIndex <= startIndex) {
    throw new ContractError(
      "Container harness did not emit a structured verification report"
    );
  }
  const rawJson = stdout
    .slice(startIndex + REPORT_START_MARKER.length, endIndex)
    .trim();

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch (error) {
    throw new ContractError(
      `Container verification report is not valid JSON: ${error instanceof Error ? error.message : String(error)}`
    );
  }

  const validated = ContainerReportSchema.safeParse(parsed);
  if (!validated.success) {
    throw new ContractError(
      `Container verification report failed schema validation: ${validated.error.message}`
    );
  }
  return validated.data;
}

function stageCleanRepoFromHead(repoRoot: string, stagingDir: string): void {
  const archiveTar = execFileSync("git", ["archive", "--format=tar", "HEAD"], {
    cwd: repoRoot,
    maxBuffer: 512 * 1024 * 1024
  });

  execFileSync("tar", ["-xf", "-", "-C", stagingDir], {
    input: archiveTar
  });

  for (const excludedPath of STAGING_EXCLUDED_PATHS) {
    const fullExcludedPath = join(stagingDir, excludedPath);
    if (existsSync(fullExcludedPath)) {
      rmSync(fullExcludedPath, { recursive: true, force: true });
    }
  }
}

function writeFixtureCorpus(corpusDir: string): void {
  for (const file of buildFixtureCorpusDocuments()) {
    const targetPath = join(corpusDir, file.relativePath);
    mkdirSync(dirname(targetPath), { recursive: true });
    writeFileSync(targetPath, `${JSON.stringify(file.payload, null, 2)}\n`, "utf8");
  }
}

function buildContainerRunnerScript(): string {
  return `import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, readdirSync, readFileSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";

const START_MARKER = ${JSON.stringify(REPORT_START_MARKER)};
const END_MARKER = ${JSON.stringify(REPORT_END_MARKER)};
const FORBIDDEN_TOOLS = ${JSON.stringify(FORBIDDEN_BUILD_TOOLS)};
const SHUTDOWN_TIMEOUT_MS = ${String(SHUTDOWN_TIMEOUT_MS)};
const MAX_ASK_RESPONSE_BYTES = ${String(MAX_ASK_RESPONSE_BYTES)};
const MAX_TAIL_BYTES = 8192;

const config = JSON.parse(readFileSync("/harness/config.json", "utf8"));
const geminiApiKey = process.env.HARNESS_GEMINI_API_KEY ?? "";
if (geminiApiKey.trim().length === 0) {
  throw new Error("HARNESS_GEMINI_API_KEY was not provided to container harness");
}

const serveEnv = {
  CORPUS_DIR: config.serveEnv.CORPUS_DIR,
  PORT: config.serveEnv.PORT,
  GEMINI_API_KEY: geminiApiKey,
  PATH: config.serveEnv.PATH,
  HOME: config.serveEnv.HOME,
  TMPDIR: config.serveEnv.TMPDIR,
  LANG: config.serveEnv.LANG
};

const checks = [];
const observedPreReadyStates = [];
let observedNotReadyStatus = null;
let coldStartMs = null;
let peakRssKiB = null;
let peakRssMiB = null;
let sigtermExitCode = null;
let sigtermSignal = null;
let shutdownMs = null;
let leakedPids = [];

let stdoutTail = Buffer.alloc(0);
let stderrTail = Buffer.alloc(0);

function appendTail(current, chunk) {
  const buf = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
  if (buf.length >= MAX_TAIL_BYTES) {
    return Buffer.from(buf.subarray(buf.length - MAX_TAIL_BYTES));
  }
  const combined = Buffer.concat([current, buf]);
  if (combined.length <= MAX_TAIL_BYTES) {
    return combined;
  }
  return Buffer.from(combined.subarray(combined.length - MAX_TAIL_BYTES));
}

function getLogTail() {
  const parts = [];
  const errText = stderrTail.toString("utf8").trim();
  const outText = stdoutTail.toString("utf8").trim();
  if (errText.length > 0) parts.push("stderr:\\n" + errText);
  if (outText.length > 0) parts.push("stdout:\\n" + outText);
  return parts.join("\\n");
}

function recordPreReadyState(state) {
  if (!observedPreReadyStates.includes(state)) {
    observedPreReadyStates.push(state);
  }
}

function parseVmHwm(statusText) {
  const match = /^VmHWM:\\s*(\\d+)\\s*kB\\s*$/im.exec(statusText);
  if (!match || !match[1]) {
    throw new Error("VmHWM field missing in /proc status");
  }
  return Number(match[1]);
}

function listPidsInProcessGroup(targetPgid) {
  const pids = [];
  let entries;
  try {
    entries = readdirSync("/proc");
  } catch (error) {
    throw new Error("Failed to read /proc: " + String(error));
  }
  for (const entry of entries) {
    if (!/^\\d+$/.test(entry)) continue;
    const pid = Number(entry);
    try {
      const statText = readFileSync("/proc/" + entry + "/stat", "utf8");
      const closeParen = statText.lastIndexOf(")");
      if (closeParen === -1) continue;
      const fields = statText.slice(closeParen + 2).trim().split(/\\s+/);
      const state = fields[0];
      const pgrp = Number(fields[2]);
      if (pgrp === targetPgid && state !== "Z") {
        pids.push(pid);
      }
    } catch (error) {
      if (error && error.code === "ENOENT") {
        continue;
      }
      throw error;
    }
  }
  return pids;
}

function readProcessGroupPeakRssKiB(leaderPid) {
  const pids = new Set([leaderPid, ...listPidsInProcessGroup(leaderPid)]);
  let totalKiB = 0;
  for (const pid of pids) {
    const statusText = readFileSync("/proc/" + String(pid) + "/status", "utf8");
    totalKiB += parseVmHwm(statusText);
  }
  return totalKiB;
}

function emitReportAndExit() {
  const passed = checks.length > 0 && checks.every((c) => c.passed);
  const report = {
    passed,
    image: config.image,
    port: config.port,
    liveAsk: config.liveAsk,
    buildToolsPresent,
    serveEnvKeys: Object.keys(serveEnv),
    observedPreReadyStates,
    observedNotReadyStatus,
    coldStartMs,
    peakRssKiB,
    peakRssMiB,
    maxRssMiB: config.maxRssMiB,
    sigtermExitCode,
    sigtermSignal,
    shutdownMs,
    leakedPids,
    checks,
    serveLogTail: getLogTail()
  };
  process.stdout.write(
    "\\n" + START_MARKER + "\\n" + JSON.stringify(report) + "\\n" + END_MARKER + "\\n"
  );
  process.exit(passed ? 0 : 1);
}

// 1. Check zero C++ build tools
const buildToolsPresent = [];
for (const tool of FORBIDDEN_TOOLS) {
  const result = spawnSync("/bin/sh", ["-c", 'command -v "$1" >/dev/null 2>&1', "--", tool], {
    env: { PATH: serveEnv.PATH }
  });
  if (result.status === 0) {
    buildToolsPresent.push(tool);
  }
}
checks.push({
  name: "build_tools_absent",
  passed: buildToolsPresent.length === 0,
  detail:
    buildToolsPresent.length === 0
      ? "Verified absent: " + FORBIDDEN_TOOLS.join(", ")
      : "Unexpected build tools present: " + buildToolsPresent.join(", ")
});

if (buildToolsPresent.length > 0) {
  emitReportAndExit();
}

// 2. Prepare writable HOME and TMPDIR, verify exact docs/api-contract.md env keys
mkdirSync(serveEnv.HOME, { recursive: true });
mkdirSync(serveEnv.TMPDIR, { recursive: true });

const expectedEnvKeys = ["CORPUS_DIR", "PORT", "GEMINI_API_KEY", "PATH", "HOME", "TMPDIR", "LANG"];
const actualEnvKeys = Object.keys(serveEnv);
const envKeysMatch =
  actualEnvKeys.length === expectedEnvKeys.length &&
  expectedEnvKeys.every((key) => actualEnvKeys.includes(key));
checks.push({
  name: "contract_env_minimal",
  passed: envKeysMatch,
  detail: "Passed exact env keys: " + actualEnvKeys.join(", ")
});

// 3. Start ./serve in its own process group via setsid, stdin /dev/null
const startMs = Date.now();
const child = spawn("setsid", ["./serve"], {
  cwd: "/workspace",
  env: serveEnv,
  stdio: ["ignore", "pipe", "pipe"]
});

let spawnError = null;
let childExited = false;
let earlyExitDetail = null;
const lifecycleAbort = new AbortController();

child.on("error", (error) => {
  spawnError = error;
  lifecycleAbort.abort();
});
child.stdout.on("data", (chunk) => {
  stdoutTail = appendTail(stdoutTail, chunk);
});
child.stderr.on("data", (chunk) => {
  stderrTail = appendTail(stderrTail, chunk);
});
child.once("exit", (code, signal) => {
  childExited = true;
  earlyExitDetail = "exited before ready (code=" + String(code) + ", signal=" + String(signal) + ")";
  lifecycleAbort.abort();
});

const baseUrl = "http://127.0.0.1:" + String(config.port);
const healthUrl = baseUrl + "/health";
const askUrl = baseUrl + "/ask";

// 4. Poll GET /health until 200
async function waitForReady() {
  const deadline = startMs + config.startupTimeoutMs;
  while (Date.now() < deadline) {
    if (spawnError !== null) {
      return { ready: false, detail: "Failed to spawn ./serve: " + spawnError.message };
    }
    if (childExited) {
      return { ready: false, detail: "./serve " + earlyExitDetail };
    }

    try {
      const response = await fetch(healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(Math.min(1000, Math.max(100, deadline - Date.now())))
      });
      if (response.status === 200) {
        coldStartMs = Date.now() - startMs;
        return {
          ready: true,
          detail:
            "GET /health reached 200 in " +
            String(coldStartMs) +
            " ms (pre-ready states: " +
            (observedPreReadyStates.length > 0 ? observedPreReadyStates.join(", ") : "none") +
            ")"
        };
      }
      if (observedNotReadyStatus === null) {
        observedNotReadyStatus = response.status;
      }
      recordPreReadyState("http_" + String(response.status));
    } catch (error) {
      const code = error && error.cause && error.cause.code ? String(error.cause.code) : "connection_refused";
      recordPreReadyState(code);
    }

    if (spawnError !== null || childExited) {
      break;
    }

    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      break;
    }
    try {
      await delay(Math.min(250, remaining), undefined, { signal: lifecycleAbort.signal });
    } catch (error) {
      if (!(error && error.name === "AbortError")) {
        throw error;
      }
    }
  }

  if (spawnError !== null) {
    return { ready: false, detail: "Failed to spawn ./serve: " + spawnError.message };
  }
  if (childExited) {
    return { ready: false, detail: "./serve " + earlyExitDetail };
  }
  return {
    ready: false,
    detail: "GET /health did not return 200 within " + String(config.startupTimeoutMs) + " ms"
  };
}

async function checkAskStatus(name, bodyText, expectedStatus) {
  try {
    const response = await fetch(askUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: bodyText,
      signal: AbortSignal.timeout(15000)
    });
    const passed = response.status === expectedStatus;
    return {
      name,
      passed,
      detail: passed
        ? "POST /ask returned expected status " + String(expectedStatus)
        : "Expected status " + String(expectedStatus) + ", got " + String(response.status)
    };
  } catch (error) {
    return {
      name,
      passed: false,
      detail: "POST /ask request failed: " + (error instanceof Error ? error.message : String(error))
    };
  }
}

async function runLiveAskChecks() {
  const questions = [
    {
      label: "q1_postgres",
      question: "What Postgres version and replication setup does the Billing Ledger cluster use?",
      expectUnanswerable: false
    },
    {
      label: "q2_cutover",
      question: "What was the p95 latency during the Nimbus gateway canary cutover?",
      expectUnanswerable: false
    },
    {
      label: "q3_sla",
      question: "What uptime percentage does the Nimbus Gateway annual SLA renewal specify?",
      expectUnanswerable: false
    },
    {
      label: "q4_unanswerable",
      question: "What is the secret launch code for the fictional Mars Underwater Monorail scheduled for year 2099?",
      expectUnanswerable: true
    }
  ];

  const results = await Promise.all(
    questions.map(async (item) => {
      try {
        const response = await fetch(askUrl, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ question: item.question }),
          signal: AbortSignal.timeout(180000)
        });
        const rawText = await response.text();
        const bytes = Buffer.byteLength(rawText, "utf8");
        if (response.status !== 200) {
          return { ok: false, label: item.label, reason: "status " + String(response.status) };
        }
        if (bytes > MAX_ASK_RESPONSE_BYTES) {
          return { ok: false, label: item.label, reason: "response exceeded 1 MiB (" + String(bytes) + " bytes)" };
        }
        const parsed = JSON.parse(rawText);
        if (typeof parsed.answer !== "string" || parsed.answer.trim().length === 0) {
          return { ok: false, label: item.label, reason: "answer is empty or not a string" };
        }
        if (!Array.isArray(parsed.document_ids) || !parsed.document_ids.every((id) => typeof id === "string" && id.trim().length > 0)) {
          return { ok: false, label: item.label, reason: "document_ids is not a string array" };
        }
        if (item.expectUnanswerable && parsed.document_ids.length !== 0) {
          return {
            ok: false,
            label: item.label,
            reason: "unanswerable query returned non-empty citations: " + JSON.stringify(parsed.document_ids)
          };
        }
        return { ok: true, label: item.label, bytes, citations: parsed.document_ids.length };
      } catch (error) {
        return {
          ok: false,
          label: item.label,
          reason: error instanceof Error ? error.message : String(error)
        };
      }
    })
  );

  const failures = results.filter((r) => !r.ok);
  checks.push({
    name: "live_ask_concurrency_4",
    passed: failures.length === 0,
    detail:
      failures.length === 0
        ? "4/4 concurrent live POST /ask calls returned 200 <= 1 MiB with unanswerable document_ids=[]"
        : failures.map((f) => f.label + ": " + f.reason).join("; ")
  });
}

async function stopProcessGroupAndVerify(pgid) {
  if (childExited) {
    checks.push({
      name: "sigterm_shutdown",
      passed: false,
      detail: "./serve exited before SIGTERM (" + String(earlyExitDetail) + ")"
    });
    return;
  }

  const shutdownStart = Date.now();
  const exitResult = await new Promise((resolveExit) => {
    const timer = setTimeout(() => {
      try {
        process.kill(-pgid, "SIGKILL");
      } catch (error) {
        if (!(error && error.code === "ESRCH")) {
          stderrTail = appendTail(stderrTail, "SIGKILL error: " + String(error) + "\\n");
        }
      }
      resolveExit({ timedOut: true, code: null, signal: "SIGKILL" });
    }, SHUTDOWN_TIMEOUT_MS);

    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      resolveExit({ timedOut: false, code, signal });
    });

    try {
      process.kill(-pgid, "SIGTERM");
    } catch (error) {
      clearTimeout(timer);
      resolveExit({
        timedOut: false,
        code: null,
        signal: null,
        killError: error instanceof Error ? error.message : String(error)
      });
    }
  });

  shutdownMs = Date.now() - shutdownStart;
  sigtermExitCode = exitResult.code;
  sigtermSignal = exitResult.signal;

  if (exitResult.timedOut) {
    checks.push({
      name: "sigterm_shutdown",
      passed: false,
      detail: "Process group -" + String(pgid) + " did not exit within " + String(SHUTDOWN_TIMEOUT_MS) + " ms"
    });
  } else if (exitResult.killError) {
    checks.push({
      name: "sigterm_shutdown",
      passed: false,
      detail: "Failed to send SIGTERM to -" + String(pgid) + ": " + exitResult.killError
    });
  } else {
    const cleanExit = exitResult.code === 0;
    checks.push({
      name: "sigterm_shutdown",
      passed: cleanExit,
      detail:
        "SIGTERM to -" +
        String(pgid) +
        " exited in " +
        String(shutdownMs) +
        " ms (code=" +
        String(exitResult.code) +
        ", signal=" +
        String(exitResult.signal) +
        ")"
    });
  }

  leakedPids = listPidsInProcessGroup(pgid);
  let pgrpAlive = false;
  try {
    process.kill(-pgid, 0);
    pgrpAlive = true;
  } catch (error) {
    if (!(error && error.code === "ESRCH")) {
      pgrpAlive = true;
    }
  }

  if (leakedPids.length > 0 || pgrpAlive) {
    try {
      process.kill(-pgid, "SIGKILL");
    } catch (error) {
      if (!(error && error.code === "ESRCH")) {
        stderrTail = appendTail(stderrTail, "Cleanup SIGKILL error: " + String(error) + "\\n");
      }
    }
  }

  checks.push({
    name: "zero_leaked_processes",
    passed: leakedPids.length === 0 && !pgrpAlive,
    detail:
      leakedPids.length === 0 && !pgrpAlive
        ? "Zero leaked processes in process group " + String(pgid)
        : "Leaked processes in process group " + String(pgid) + ": " + JSON.stringify(leakedPids)
  });
}

async function run() {
  const pgid = child.pid;
  try {
    const readiness = await waitForReady();
    checks.push({
      name: "health_readiness",
      passed: readiness.ready,
      detail: readiness.detail
    });

    if (readiness.ready && pgid !== undefined) {
      checks.push(await checkAskStatus("ask_malformed_json", "{malformed-json", 400));
      checks.push(await checkAskStatus("ask_invalid_schema", "{}", 400));

      if (config.liveAsk) {
        await runLiveAskChecks();
      }

      try {
        peakRssKiB = readProcessGroupPeakRssKiB(pgid);
        peakRssMiB = Number((peakRssKiB / 1024).toFixed(2));
        const withinBudget = peakRssMiB <= config.maxRssMiB;
        checks.push({
          name: "peak_rss_limit",
          passed: withinBudget,
          detail:
            "Peak RSS (VmHWM) = " +
            String(peakRssMiB) +
            " MiB (" +
            String(peakRssKiB) +
            " KiB), limit = " +
            String(config.maxRssMiB) +
            " MiB"
        });
      } catch (error) {
        checks.push({
          name: "peak_rss_limit",
          passed: false,
          detail:
            "Failed to read peak RSS: " +
            (error instanceof Error ? error.message : String(error))
        });
      }
    }
  } finally {
    if (pgid !== undefined) {
      await stopProcessGroupAndVerify(pgid);
    }
  }

  emitReportAndExit();
}

void run().catch((error) => {
  checks.push({
    name: "harness_runtime",
    passed: false,
    detail: error instanceof Error ? error.message : String(error)
  });
  emitReportAndExit();
});
`;
}

function runDockerContainer(args: readonly string[], env: NodeJS.ProcessEnv): Promise<{
  exitCode: number | null;
  stdout: string;
  stderr: string;
}> {
  return new Promise((resolveRun, rejectRun) => {
    const proc = spawn("docker", [...args], {
      env,
      stdio: ["ignore", "pipe", "pipe"]
    });
    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    proc.stdout.on("data", (chunk: Buffer | string) => {
      stdoutChunks.push(typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk);
    });
    proc.stderr.on("data", (chunk: Buffer | string) => {
      stderrChunks.push(typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk);
    });
    proc.on("error", (error: Error) => {
      rejectRun(new ContractError(`Failed to run docker: ${error.message}`, { cause: error }));
    });
    proc.on("close", (code: number | null) => {
      resolveRun({
        exitCode: code,
        stdout: Buffer.concat(stdoutChunks).toString("utf8"),
        stderr: Buffer.concat(stderrChunks).toString("utf8")
      });
    });
  });
}

function formatVerificationReport(
  report: ContainerVerificationReport,
  corpusMode: string
): string {
  const lines = [
    "=== Docker Cold-Start Verification Report ===",
    `Status:          ${report.passed ? "PASS" : "FAIL"}`,
    `Image:           ${report.image}`,
    `Corpus:          ${corpusMode}`,
    `Port:            ${String(report.port)}`,
    `Live /ask:       ${report.liveAsk ? "enabled" : "disabled"}`,
    `Cold start:      ${report.coldStartMs !== null ? `${String(report.coldStartMs)} ms` : "n/a"}`,
    `Pre-ready:       ${report.observedPreReadyStates.length > 0 ? report.observedPreReadyStates.join(", ") : "none"}`,
    `Peak RSS:        ${report.peakRssMiB !== null ? `${String(report.peakRssMiB)} MiB (limit ${String(report.maxRssMiB)} MiB)` : "n/a"}`,
    `SIGTERM exit:    code=${String(report.sigtermExitCode)} signal=${String(report.sigtermSignal)} (${report.shutdownMs !== null ? `${String(report.shutdownMs)} ms` : "n/a"})`,
    `Leaked PIDs:     ${String(report.leakedPids.length)}`,
    "Checks:"
  ];
  for (const check of report.checks) {
    lines.push(`  [${check.passed ? "PASS" : "FAIL"}] ${check.name}: ${check.detail}`);
  }
  if (!report.passed && report.serveLogTail.length > 0) {
    lines.push("Serve log tail:");
    lines.push(report.serveLogTail);
  }
  return lines.join("\n");
}

export async function runVerifyDocker(
  options: VerifyDockerOptions,
  repoRoot = process.cwd()
): Promise<ContainerVerificationReport> {
  const geminiApiKey = options.liveAsk
    ? (process.env.GEMINI_API_KEY?.trim() ?? "")
    : DUMMY_GEMINI_API_KEY;

  if (options.liveAsk && geminiApiKey.length === 0) {
    throw new ConfigError("--live-ask requires GEMINI_API_KEY to be set");
  }

  let hostCorpusDir: string;
  let corpusMode: string;
  const tempRoot = realpathSync(mkdtempSync(join(tmpdir(), "sqlite-rag-bench-docker-")));
  const stagingDir = join(tempRoot, "workspace");
  const harnessDir = join(tempRoot, "harness");

  try {
    mkdirSync(stagingDir, { recursive: true });
    mkdirSync(harnessDir, { recursive: true });

    if (options.corpusDir !== null) {
      if (!existsSync(options.corpusDir) || !statSync(options.corpusDir).isDirectory()) {
        throw new ConfigError(`Corpus directory does not exist: ${options.corpusDir}`);
      }
      hostCorpusDir = realpathSync(options.corpusDir);
      corpusMode = `custom (${hostCorpusDir})`;
    } else {
      hostCorpusDir = join(tempRoot, "corpus");
      mkdirSync(hostCorpusDir, { recursive: true });
      writeFixtureCorpus(hostCorpusDir);
      corpusMode = "built-in 4-document multi-source fixture";
    }

    stageCleanRepoFromHead(repoRoot, stagingDir);

    const serveEnv = buildMinimalContractEnv({
      corpusDir: "/corpus",
      port: options.port,
      geminiApiKey: "REDACTED_IN_CONFIG"
    });

    const harnessConfig = {
      image: DOCKER_IMAGE,
      port: options.port,
      maxRssMiB: options.maxRssMiB,
      startupTimeoutMs: options.startupTimeoutMs,
      liveAsk: options.liveAsk,
      serveEnv: {
        CORPUS_DIR: serveEnv.CORPUS_DIR,
        PORT: serveEnv.PORT,
        PATH: serveEnv.PATH,
        HOME: serveEnv.HOME,
        TMPDIR: serveEnv.TMPDIR,
        LANG: serveEnv.LANG
      }
    };

    writeFileSync(
      join(harnessDir, "config.json"),
      `${JSON.stringify(harnessConfig, null, 2)}\n`,
      "utf8"
    );
    writeFileSync(
      join(harnessDir, "runner.mjs"),
      buildContainerRunnerScript(),
      "utf8"
    );

    const dockerArgs = [
      "run",
      "--rm",
      "-e",
      "HARNESS_GEMINI_API_KEY",
      "-v",
      `${stagingDir}:/workspace:rw`,
      "-v",
      `${hostCorpusDir}:/corpus:ro`,
      "-v",
      `${harnessDir}:/harness:ro`,
      "-w",
      "/workspace",
      DOCKER_IMAGE,
      "node",
      "/harness/runner.mjs"
    ];

    const dockerResult = await runDockerContainer(dockerArgs, {
      ...process.env,
      HARNESS_GEMINI_API_KEY: geminiApiKey
    });

    let report: ContainerVerificationReport;
    try {
      report = parseContainerReportOutput(dockerResult.stdout);
    } catch (error) {
      const details = [
        `docker exit code: ${String(dockerResult.exitCode)}`,
        dockerResult.stderr.trim().length > 0 ? `stderr: ${dockerResult.stderr.trim()}` : "",
        dockerResult.stdout.trim().length > 0 ? `stdout: ${dockerResult.stdout.trim()}` : ""
      ]
        .filter((part) => part.length > 0)
        .join("\n");
      throw new ContractError(
        `${error instanceof Error ? error.message : String(error)}\n${details}`
      );
    }

    console.log(formatVerificationReport(report, corpusMode));

    if (!report.passed || dockerResult.exitCode !== 0) {
      const failedNames = report.checks
        .filter((check) => !check.passed)
        .map((check) => check.name)
        .join(", ");
      throw new ContractError(
        `Docker cold-start verification failed (exit=${String(dockerResult.exitCode)}, failed checks: ${failedNames || "none"})`
      );
    }

    return report;
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  const options = parseVerifyDockerArgs(process.argv.slice(2));
  await runVerifyDocker(options);
}

const currentFilePath = fileURLToPath(import.meta.url);
if (
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === currentFilePath
) {
  void main().catch((error: unknown) => {
    console.error(
      `[verify:docker] ${error instanceof Error ? error.message : String(error)}`
    );
    process.exit(1);
  });
}
