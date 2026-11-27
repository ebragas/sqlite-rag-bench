import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type {
  LlmAgentStepOutput,
  LlmProvider,
  LlmSynthesizeInput,
  LlmSynthesizeOutput
} from "../../src/adapters/llm-provider.js";
import { createNoopTelemetryController } from "../../src/adapters/telemetry.js";
import { ConfigError } from "../../src/models/errors.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import {
  buildAnswerJsonPayload,
  formatAnswerSummary,
  formatStartupSummary,
  formatTraceEvent,
  parseAskCliArgs,
  runAskCli
} from "../../src/services/ask-cli.js";
import type { AnswerResult } from "../../src/usecases/answer.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

function createFixtureCorpus(): { root: string; corpusDir: string; dbPath: string } {
  const root = mkdtempSync(join(tmpdir(), "sqlite-rag-bench-ask-"));
  tempDirs.push(root);
  const corpusDir = join(root, "corpus");
  const dbPath = join(root, "cache", "ask.sqlite");

  mkdirSync(join(corpusDir, "confluence", "product"), { recursive: true });
  writeFileSync(
    join(corpusDir, "confluence", "product", "sso.json"),
    JSON.stringify({
      dataset_doc_uuid: "dsid_sso_1",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Enterprise SSO Guide",
      status: "approved",
      body: "Enterprise plan supports SAML 2.0 and OIDC single sign-on."
    }),
    "utf8"
  );

  return { root, corpusDir, dbPath };
}

class FakeAskLlmProvider implements LlmProvider {
  public async synthesize(input: LlmSynthesizeInput): Promise<LlmSynthesizeOutput> {
    return {
      result: {
        answer: "Enterprise plan supports SAML 2.0 and OIDC SSO.",
        abstained: false,
        supporting_doc_ids: ["dsid_sso_1", "dsid_unseen_999"]
      },
      usage: { inputTokens: 200, outputTokens: 50 },
      model: input.model,
      reasoningText: "Confirmed SAML 2.0 and OIDC in Enterprise SSO Guide."
    };
  }

  public async runAgentStep(): Promise<LlmAgentStepOutput> {
    return {
      text: "Done.",
      toolCalls: [],
      usage: { inputTokens: 50, outputTokens: 10 },
      model: "gemini-3.5-flash-lite"
    };
  }
}

describe("parseAskCliArgs", () => {
  it("applies default flags and parses repeatable --set and positional question", () => {
    expect(parseAskCliArgs([])).toEqual({
      profile: "prod",
      setOverrides: [],
      corpusDir: "data/corpus",
      quiet: false,
      json: false
    });

    expect(
      parseAskCliArgs([
        "--profile",
        "l1-filters",
        "--set",
        "topK=20",
        "--set",
        "maxAgentTurns=4",
        "--corpus",
        "tmp/corpus",
        "--db",
        "tmp/brain.sqlite",
        "--quiet",
        "Which",
        "vendor",
        "won?"
      ])
    ).toEqual({
      profile: "l1-filters",
      setOverrides: ["topK=20", "maxAgentTurns=4"],
      corpusDir: "tmp/corpus",
      dbPath: "tmp/brain.sqlite",
      quiet: true,
      json: false,
      question: "Which vendor won?"
    });
  });

  it("rejects unknown flags with ConfigError", () => {
    expect(() => parseAskCliArgs(["--unknown-flag"])).toThrow(ConfigError);
  });
});

describe("ask-cli formatting helpers", () => {
  const profile = RunProfileSchema.parse({
    name: "l3-graph",
    enabledTools: ["search", "fetch_docs", "filter_metadata", "expand_graph"]
  });

  const sampleResult: AnswerResult = {
    response: {
      answer: "Enterprise plan supports SAML 2.0.",
      document_ids: ["dsid_sso_1"]
    },
    abstained: false,
    retrievedDocIds: ["dsid_sso_1"],
    strippedDocIds: ["dsid_hallucinated_1"],
    usage: { inputTokens: 1000, outputTokens: 250 },
    embeddingInputTokens: 120,
    model: "gemini-3.8-flash",
    agentTurns: 2,
    deadlineHit: false,
    traceId: "1234567890abcdef1234567890abcdef"
  };

  it("formats startup banner, live trace events, answer summary, and JSON payload", () => {
    const startup = formatStartupSummary({
      profile,
      indexConfigHash: "abcdef1234567890",
      dbPath: "/tmp/brain.sqlite",
      documentCount: 42,
      chunkCount: 128,
      sourceCount: 5,
      graphEdgeCount: 90,
      graphResolvedEdgeCount: 75,
      vectorCount: 128,
      langfuseEnabled: true,
      langfuseBaseUrl: "https://us.cloud.langfuse.com"
    });
    expect(startup).toContain("l3-graph");
    expect(startup).toContain("docs=42 chunks=128");
    expect(startup).toContain("graph_edges=75/90 vectors=128");
    expect(startup).toContain("ON (https://us.cloud.langfuse.com)");

    expect(
      formatTraceEvent({
        type: "agent_turn_start",
        turn: 1,
        maxTurns: 6,
        model: "gemini-3.5-flash-lite"
      })
    ).toContain("Turn 1/6 (gemini-3.5-flash-lite)");

    expect(
      formatTraceEvent({
        type: "agent_thinking",
        turn: 1,
        text: "Check SSO docs first."
      })
    ).toContain("  Check SSO docs first.");

    expect(
      formatTraceEvent({
        type: "tool_call",
        turn: 1,
        toolCallId: "c1",
        toolName: "search",
        input: { query: "Enterprise SSO" }
      })
    ).toBe('[tool] search({"query":"Enterprise SSO"})');

    expect(
      formatTraceEvent({
        type: "tool_result",
        turn: 1,
        toolCallId: "c1",
        toolName: "search",
        isError: false,
        durationMs: 12.4,
        summary: "15 hits (hybrid)",
        output: {}
      })
    ).toBe("  -> 15 hits (hybrid) (12ms)");

    expect(
      formatTraceEvent({
        type: "citations_validated",
        validDocIds: ["dsid_sso_1"],
        strippedDocIds: ["dsid_hallucinated_1"],
        abstained: false
      })
    ).toContain("stripped 1: dsid_hallucinated_1");

    expect(
      formatTraceEvent({
        type: "citations_validated",
        validDocIds: ["dsid_sso_1"],
        strippedDocIds: [],
        abstained: false
      })
    ).toBeNull();

    const fullSummary = formatAnswerSummary({
      question: "Does Enterprise support SSO?",
      profile,
      result: sampleResult,
      latencyMs: 842
    });
    expect(fullSummary).toContain("Enterprise plan supports SAML 2.0.");
    expect(fullSummary).toContain("Abstained: no");
    expect(fullSummary).toContain("Citations (1): dsid_sso_1");
    expect(fullSummary).toContain("Stripped citations (1): dsid_hallucinated_1");
    expect(fullSummary).toContain("tokens in=1000 out=250 embed=120");
    expect(fullSummary).toContain("Langfuse traceId: 1234567890abcdef1234567890abcdef");

    const quietSummary = formatAnswerSummary({
      question: "Does Enterprise support SSO?",
      profile,
      result: sampleResult,
      latencyMs: 842,
      quiet: true
    });
    expect(quietSummary).toBe(
      "Enterprise plan supports SAML 2.0.\nCitations: dsid_sso_1"
    );

    const payload = buildAnswerJsonPayload({
      question: "Does Enterprise support SSO?",
      profile,
      result: sampleResult,
      latencyMs: 842
    });
    expect(payload).toEqual(
      expect.objectContaining({
        profile: "l3-graph",
        abstained: false,
        document_ids: ["dsid_sso_1"],
        stripped_doc_ids: ["dsid_hallucinated_1"],
        trace_id: "1234567890abcdef1234567890abcdef"
      })
    );
  });
});

describe("runAskCli end-to-end without live Gemini calls", () => {
  it("fails with ConfigError when GEMINI_API_KEY is missing and no provider override is supplied", async () => {
    await expect(
      runAskCli(["--profile", "l0-fts", "What is SSO?"], {
        env: {}
      })
    ).rejects.toThrow(ConfigError);
  });

  it("streams trace events, stripped citations, and JSON output against a real temp SQLite store", async () => {
    const { root, corpusDir, dbPath } = createFixtureCorpus();
    const outChunks: string[] = [];
    const errChunks: string[] = [];

    await runAskCli(
      [
        "--profile",
        "l0-fts",
        "--corpus",
        corpusDir,
        "--db",
        dbPath,
        "What SSO protocols does Enterprise support?"
      ],
      {
        cwd: root,
        env: {},
        llmProvider: new FakeAskLlmProvider(),
        telemetry: createNoopTelemetryController(),
        stdout: {
          write(chunk: string | Uint8Array): boolean {
            outChunks.push(String(chunk));
            return true;
          }
        },
        stderr: {
          write(chunk: string | Uint8Array): boolean {
            errChunks.push(String(chunk));
            return true;
          }
        }
      }
    );

    const output = outChunks.join("");
    expect(errChunks).toEqual([]);
    expect(output).toContain("Enterprise Knowledge CLI");
    expect(output).toContain("[search] pre-search (fts) -> 1 hit");
    expect(output).toContain("Confirmed SAML 2.0 and OIDC in Enterprise SSO Guide.");
    expect(output).toContain("stripped 1: dsid_unseen_999");
    expect(output).toContain("Citations (1): dsid_sso_1");

    const jsonChunks: string[] = [];
    await runAskCli(
      [
        "--profile",
        "l0-fts",
        "--corpus",
        corpusDir,
        "--db",
        dbPath,
        "--json",
        "What SSO protocols does Enterprise support?"
      ],
      {
        cwd: root,
        env: {},
        llmProvider: new FakeAskLlmProvider(),
        telemetry: createNoopTelemetryController(),
        stdout: {
          write(chunk: string | Uint8Array): boolean {
            jsonChunks.push(String(chunk));
            return true;
          }
        }
      }
    );

    const parsedJson = JSON.parse(jsonChunks.join("")) as {
      answer: string;
      document_ids: string[];
      stripped_doc_ids: string[];
    };
    expect(parsedJson.answer).toBe("Enterprise plan supports SAML 2.0 and OIDC SSO.");
    expect(parsedJson.document_ids).toEqual(["dsid_sso_1"]);
    expect(parsedJson.stripped_doc_ids).toEqual(["dsid_unseen_999"]);
  });
});
