import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildFixtureCorpusDocuments,
  buildMinimalContractEnv,
  CONTRACT_ENV_KEYS,
  DEFAULT_CONTAINER_HOME,
  DEFAULT_CONTAINER_LANG,
  DEFAULT_CONTAINER_PATH,
  DEFAULT_CONTAINER_TMPDIR,
  DEFAULT_MAX_RSS_MIB,
  DEFAULT_PORT,
  DEFAULT_STARTUP_TIMEOUT_MS,
  kibToMiB,
  parseContainerReportOutput,
  parseVerifyDockerArgs,
  parseVmHwmKiB,
  REPORT_END_MARKER,
  REPORT_START_MARKER
} from "../../scripts/verify-docker.js";
import { ConfigError, ContractError } from "../../src/models/errors.js";

describe("parseVerifyDockerArgs", () => {
  it("returns default options when called with no arguments", () => {
    const parsed = parseVerifyDockerArgs([]);
    expect(parsed).toEqual({
      corpusDir: null,
      port: DEFAULT_PORT,
      maxRssMiB: DEFAULT_MAX_RSS_MIB,
      startupTimeoutMs: DEFAULT_STARTUP_TIMEOUT_MS,
      liveAsk: false
    });
  });

  it("parses all supported flags and resolves corpusDir relative to cwd", () => {
    const parsed = parseVerifyDockerArgs(
      [
        "--corpus-dir",
        "fixtures/corpus",
        "--port",
        "9090",
        "--max-rss-mib",
        "2048.5",
        "--startup-timeout-ms",
        "60000",
        "--live-ask"
      ],
      "/tmp/workspace"
    );

    expect(parsed).toEqual({
      corpusDir: resolve("/tmp/workspace", "fixtures/corpus"),
      port: 9090,
      maxRssMiB: 2048.5,
      startupTimeoutMs: 60_000,
      liveAsk: true
    });
  });

  it("throws ConfigError on invalid flags, missing values, and out-of-range numbers", () => {
    expect(() => parseVerifyDockerArgs(["--unknown"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--corpus-dir"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--port", "0"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--port", "70000"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--port", "80.5"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--max-rss-mib", "-10"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--max-rss-mib", "NaN"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--startup-timeout-ms", "0"])).toThrow(ConfigError);
    expect(() => parseVerifyDockerArgs(["--startup-timeout-ms", "abc"])).toThrow(ConfigError);
  });
});

describe("parseVmHwmKiB and kibToMiB", () => {
  it("extracts VmHWM from realistic Linux /proc/<pid>/status output", () => {
    const procStatus = [
      "Name:\tnode",
      "Umask:\t0022",
      "State:\tS (sleeping)",
      "Tgid:\t42",
      "Pid:\t42",
      "VmPeak:\t 1284512 kB",
      "VmSize:\t 1200420 kB",
      "VmHWM:\t  148992 kB",
      "VmRSS:\t  132100 kB",
      "Threads:\t7"
    ].join("\n");

    const kib = parseVmHwmKiB(procStatus);
    expect(kib).toBe(148_992);
    expect(kibToMiB(kib)).toBe(145.5);
  });

  it("throws ContractError when VmHWM is absent or malformed", () => {
    expect(() => parseVmHwmKiB("Name:\tnode\nVmRSS:\t1200 kB\n")).toThrow(ContractError);
    expect(() => parseVmHwmKiB("VmHWM:\tnot-a-number kB\n")).toThrow(ContractError);
    expect(() => kibToMiB(-1)).toThrow(ContractError);
  });
});

describe("buildMinimalContractEnv", () => {
  it("contains only the 7 docs/api-contract.md environment keys with no leaked host secrets", () => {
    const env = buildMinimalContractEnv({
      corpusDir: "/corpus",
      port: 8080,
      geminiApiKey: "dummy-docker-smoke-key"
    });

    expect(Object.keys(env).sort()).toEqual([...CONTRACT_ENV_KEYS].sort());
    expect(env).toEqual({
      CORPUS_DIR: "/corpus",
      PORT: "8080",
      GEMINI_API_KEY: "dummy-docker-smoke-key",
      PATH: DEFAULT_CONTAINER_PATH,
      HOME: DEFAULT_CONTAINER_HOME,
      TMPDIR: DEFAULT_CONTAINER_TMPDIR,
      LANG: DEFAULT_CONTAINER_LANG
    });
    expect("OPENAI_API_KEY" in env).toBe(false);
    expect("BRAIN_PROFILE" in env).toBe(false);
    expect("AWS_SECRET_ACCESS_KEY" in env).toBe(false);
  });

  it("throws ConfigError on invalid contract env fields", () => {
    expect(() =>
      buildMinimalContractEnv({ corpusDir: "", port: 8080, geminiApiKey: "k" })
    ).toThrow(ConfigError);
    expect(() =>
      buildMinimalContractEnv({ corpusDir: "/corpus", port: 0, geminiApiKey: "k" })
    ).toThrow(ConfigError);
    expect(() =>
      buildMinimalContractEnv({ corpusDir: "/corpus", port: 8080, geminiApiKey: "   " })
    ).toThrow(ConfigError);
  });
});

describe("buildFixtureCorpusDocuments and parseContainerReportOutput", () => {
  it("builds 4 nested multi-source fixture documents with unique UUIDs", () => {
    const docs = buildFixtureCorpusDocuments();
    expect(docs).toHaveLength(4);
    const paths = docs.map((doc) => doc.relativePath);
    expect(paths).toEqual([
      "slack/threads/release-cutover.json",
      "confluence/architecture/billing-db.json",
      "jira/platform/2025/OPS-404.json",
      "email/inbox/vendor-contract.json"
    ]);
    const uuids = new Set(docs.map((doc) => doc.payload.dataset_doc_uuid));
    expect(uuids.size).toBe(4);
  });

  it("parses structured container report output and rejects malformed output", () => {
    const payload = {
      passed: true,
      image: "node:20-slim",
      port: 8080,
      liveAsk: false,
      buildToolsPresent: [],
      serveEnvKeys: [...CONTRACT_ENV_KEYS],
      observedPreReadyStates: ["ECONNREFUSED", "http_503"],
      observedNotReadyStatus: 503,
      coldStartMs: 12450,
      peakRssKiB: 102400,
      peakRssMiB: 100,
      maxRssMiB: 4096,
      sigtermExitCode: 0,
      sigtermSignal: null,
      shutdownMs: 42,
      leakedPids: [],
      checks: [{ name: "health_readiness", passed: true, detail: "ok" }],
      serveLogTail: ""
    };

    const stdout = `noise\n${REPORT_START_MARKER}\n${JSON.stringify(payload)}\n${REPORT_END_MARKER}\n`;
    const parsed = parseContainerReportOutput(stdout);
    expect(parsed.passed).toBe(true);
    expect(parsed.coldStartMs).toBe(12450);
    expect(parsed.leakedPids).toEqual([]);

    expect(() => parseContainerReportOutput("no markers here")).toThrow(ContractError);
    expect(() =>
      parseContainerReportOutput(`${REPORT_START_MARKER}\n{bad-json\n${REPORT_END_MARKER}`)
    ).toThrow(ContractError);
  });
});
