import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { JudgeAdapter, JudgeEvaluation, JudgeInput } from "../../src/adapters/judge.js";
import { SpendLedgerStore } from "../../src/adapters/spend-ledger.js";
import {
  SystemOneDecisionJudgeAdapter,
  type SystemOneRequestPayload
} from "../../src/adapters/systemone-judge.js";
import { ConfigError, ContractError, ValidationError } from "../../src/models/errors.js";
import { CalibrationReportSchema } from "../../src/models/eval-harness.js";
import { calibrateJudge } from "../../src/usecases/eval-runner.js";

interface CapturedSystemOneCall {
  readonly url: string | undefined;
  readonly authorization: string | undefined;
  readonly body: SystemOneRequestPayload;
}

const openServers: Server[] = [];
const tempDirs: string[] = [];

afterEach(async () => {
  while (openServers.length > 0) {
    const server = openServers.pop();
    if (server !== undefined) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function startStubServer(
  handler: (
    req: IncomingMessage,
    rawBody: string,
    res: ServerResponse
  ) => void
): Promise<string> {
  const server = createServer((req, res) => {
    readRequestBody(req)
      .then((rawBody) => {
        handler(req, rawBody, res);
      })
      .catch((error: unknown) => {
        res.writeHead(500, { "content-type": "text/plain" });
        res.end(error instanceof Error ? error.message : String(error));
      });
  });
  openServers.push(server);

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const port = (server.address() as AddressInfo).port;
  return `http://127.0.0.1:${port}`;
}

class DeterministicFrontierJudge implements JudgeAdapter {
  public readonly modelId = "gemini-2.5-pro";
  public readonly tier = "frontier" as const;
  private readonly contradictionsByQuestion: ReadonlyMap<string, boolean>;

  public constructor(contradictionsByQuestion: ReadonlyMap<string, boolean>) {
    this.contradictionsByQuestion = contradictionsByQuestion;
  }

  public evaluate(input: JudgeInput): Promise<JudgeEvaluation> {
    const contradicted = this.contradictionsByQuestion.get(input.question) ?? false;
    return Promise.resolve({
      verdict: {
        answerScore: contradicted ? 0 : 1,
        hasFabricatedOrContradictingClaim: contradicted,
        abstained: false,
        reasoning: contradicted ? "Detected contradiction." : "Matches gold answer."
      },
      usage: { inputTokens: 800, outputTokens: 120 }
    });
  }
}

describe("SystemOneDecisionJudgeAdapter", () => {
  it("sends /v1/systemone payload with Bearer auth, computes deterministic per-fact answerScore, and reports zero output tokens", async () => {
    const captured: CapturedSystemOneCall[] = [];
    const baseUrl = await startStubServer((req, rawBody, res) => {
      const parsed = JSON.parse(rawBody) as SystemOneRequestPayload;
      captured.push({
        url: req.url,
        authorization: req.headers.authorization,
        body: parsed
      });

      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          model: "clef:27b-mxfp8",
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", noul: 0.08 },
            abstained: { type: "noul", noul: 0.02 },
            fact_0: { type: "noul", noul: 0.94 },
            fact_1: { type: "noul", probability: 0.82 },
            fact_2: { type: "noul", decision: false }
          },
          usage: { input_tokens: 412, output_tokens: 0 }
        })
      );
    });

    const judge = new SystemOneDecisionJudgeAdapter({
      modelId: "clef:27b-mxfp8",
      baseUrl,
      apiKey: "jev-secret-token"
    });

    const result = await judge.evaluate({
      question: "Which three regions were enabled in Phase 2?",
      goldAnswer: "US-East, EU-West, and AP-South were enabled in Phase 2.",
      answerFacts: ["US-East enabled", "EU-West enabled", "AP-South enabled"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "Phase 2 enabled US-East and EU-West.",
      responderDocIds: ["dsid_1"]
    });

    expect(judge.tier).toBe("decision");
    expect(captured).toHaveLength(1);
    const firstCall = captured[0];
    expect(firstCall?.url).toBe("/v1/systemone");
    expect(firstCall?.authorization).toBe("Bearer jev-secret-token");
    expect(firstCall?.body.model).toBe("clef:27b-mxfp8");
    expect(Object.keys(firstCall?.body.questions ?? {})).toEqual([
      "has_fabricated_or_contradicting_claim",
      "abstained",
      "fact_0",
      "fact_1",
      "fact_2"
    ]);

    expect(result.verdict.hasFabricatedOrContradictingClaim).toBe(false);
    expect(result.verdict.abstained).toBe(false);
    expect(result.verdict.answerScore).toBeCloseTo(2 / 3, 6);
    expect(result.usage).toEqual({ inputTokens: 412, outputTokens: 0 });
  });

  it("applies noul contradiction zeroing and omits Authorization header when apiKey is unset", async () => {
    const capturedAuth: Array<string | undefined> = [];
    const baseUrl = await startStubServer((req, _rawBody, res) => {
      capturedAuth.push(req.headers.authorization);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          model: "clef-flash",
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", noul: 0.91 },
            abstained: { type: "noul", noul: 0.05 },
            fact_0: { type: "noul", noul: 0.95 },
            fact_1: { type: "noul", noul: 0.89 }
          },
          usage: { inputTokens: 320, outputTokens: 0 }
        })
      );
    });

    const judge = new SystemOneDecisionJudgeAdapter({
      modelId: "clef-flash",
      baseUrl,
      apiKey: ""
    });

    const result = await judge.evaluate({
      question: "Who approved the Q4 rollout?",
      goldAnswer: "Naomi approved the Q4 rollout.",
      answerFacts: ["Naomi approved", "Q4 rollout"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "Naomi approved the Q4 rollout on Mars in 1999.",
      responderDocIds: ["dsid_1"]
    });

    expect(capturedAuth).toEqual([undefined]);
    expect(result.verdict.hasFabricatedOrContradictingClaim).toBe(true);
    expect(result.verdict.answerScore).toBe(0);
    expect(result.usage).toEqual({ inputTokens: 320, outputTokens: 0 });
  });

  it("handles unanswerable questions and score fallback when answerFacts is empty", async () => {
    const baseUrl = await startStubServer((_req, rawBody, res) => {
      const parsed = JSON.parse(rawBody) as SystemOneRequestPayload;
      if (parsed.state.question_type === "unanswerable") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            answers: {
              has_fabricated_or_contradicting_claim: { type: "noul", decision: false },
              abstained: { type: "noul", decision: true }
            },
            usage: { input_tokens: 180, output_tokens: 0 }
          })
        );
        return;
      }

      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", probability: 0.1 },
            abstained: { type: "noul", probability: 0.05 },
            gold_answer_score: { type: "score", score: 0.75 }
          },
          usage: { input_tokens: 210, output_tokens: 0 }
        })
      );
    });

    const judge = new SystemOneDecisionJudgeAdapter({
      modelId: "jev",
      baseUrl: `${baseUrl}/v1/systemone`
    });

    const unanswerableEval = await judge.evaluate({
      question: "What is the unrecorded Mars launch date?",
      goldAnswer: "Not in corpus.",
      answerFacts: [],
      expectedDocIds: [],
      responderAnswer: "The corpus does not contain this information.",
      responderDocIds: []
    });

    expect(unanswerableEval.verdict.abstained).toBe(true);
    expect(unanswerableEval.verdict.hasFabricatedOrContradictingClaim).toBe(false);
    expect(unanswerableEval.verdict.answerScore).toBe(1);
    expect(unanswerableEval.usage.outputTokens).toBe(0);

    const scoreFallbackEval = await judge.evaluate({
      question: "Summarize the outage root cause.",
      goldAnswer: "Expired TLS certificate on edge proxy.",
      answerFacts: [],
      expectedDocIds: ["dsid_9"],
      responderAnswer: "An expired TLS certificate caused the outage.",
      responderDocIds: ["dsid_9"]
    });

    expect(scoreFallbackEval.verdict.answerScore).toBeCloseTo(0.75, 6);
    expect(scoreFallbackEval.usage.outputTokens).toBe(0);
  });

  it("raises ContractError on non-2xx status and ValidationError on malformed JSON or invalid schema", async () => {
    let mode: "status500" | "badJson" | "badSchema" | "missingFact" = "status500";
    const baseUrl = await startStubServer((_req, _rawBody, res) => {
      if (mode === "status500") {
        res.writeHead(503, { "content-type": "text/plain" });
        res.end("upstream overloaded");
        return;
      }
      if (mode === "badJson") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end("{not-valid-json");
        return;
      }
      if (mode === "badSchema") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            answers: {
              has_fabricated_or_contradicting_claim: { type: "noul", noul: 1.5 }
            }
          })
        );
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", noul: 0.1 },
            abstained: { type: "noul", noul: 0.1 }
          }
        })
      );
    });

    const judge = new SystemOneDecisionJudgeAdapter({
      modelId: "clef:27b-mxfp8",
      baseUrl
    });

    const sampleInput: JudgeInput = {
      question: "Which vendor won?",
      goldAnswer: "AetherAI won.",
      answerFacts: ["AetherAI won"],
      expectedDocIds: ["dsid_1"],
      responderAnswer: "AetherAI won.",
      responderDocIds: ["dsid_1"]
    };

    await expect(judge.evaluate(sampleInput)).rejects.toThrow(ContractError);

    mode = "badJson";
    await expect(judge.evaluate(sampleInput)).rejects.toThrow(ValidationError);

    mode = "badSchema";
    await expect(judge.evaluate(sampleInput)).rejects.toThrow(ValidationError);

    mode = "missingFact";
    await expect(judge.evaluate(sampleInput)).rejects.toThrow(ValidationError);
  });

  it("validates constructor config eagerly", () => {
    expect(
      () => new SystemOneDecisionJudgeAdapter({ modelId: "  ", baseUrl: "http://127.0.0.1:11434" })
    ).toThrow(ConfigError);
    expect(
      () =>
        new SystemOneDecisionJudgeAdapter({
          modelId: "clef:27b-mxfp8",
          baseUrl: "http://127.0.0.1:11434",
          contradictionThreshold: 1.2
        })
    ).toThrow(ConfigError);
  });

  it("integrates with calibrateJudge for both passing (>= 90%) and failing (< 90%) agreement", async () => {
    const items: JudgeInput[] = [];
    const frontierContradictions = new Map<string, boolean>();
    const decisionProbs = new Map<string, number>();

    for (let index = 0; index < 10; index += 1) {
      const question = `Calibration question ${index}`;
      const isContradiction = index % 2 === 0;
      items.push({
        question,
        goldAnswer: `Gold ${index}`,
        answerFacts: [`Fact ${index}`],
        expectedDocIds: [`dsid_${index}`],
        responderAnswer: `Answer ${index}`,
        responderDocIds: [`dsid_${index}`]
      });
      frontierContradictions.set(question, isContradiction);
      decisionProbs.set(question, isContradiction ? 0.92 : 0.08);
    }

    const baseUrl = await startStubServer((_req, rawBody, res) => {
      const parsed = JSON.parse(rawBody) as SystemOneRequestPayload;
      const prob = decisionProbs.get(parsed.state.question) ?? 0.05;
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          model: "clef:27b-mxfp8",
          answers: {
            has_fabricated_or_contradicting_claim: { type: "noul", noul: prob },
            abstained: { type: "noul", noul: 0.02 },
            fact_0: { type: "noul", noul: prob >= 0.5 ? 0.1 : 0.95 }
          },
          usage: { input_tokens: 250, output_tokens: 0 }
        })
      );
    });

    const root = mkdtempSync(join(tmpdir(), "srb-systemone-cal-"));
    tempDirs.push(root);
    const ledger = new SpendLedgerStore(join(root, "spend-ledger.jsonl"));
    const passReportPath = join(root, "calibrations", "pass.json");

    const systemOneJudge = new SystemOneDecisionJudgeAdapter({
      modelId: "clef:27b-mxfp8",
      baseUrl
    });
    const frontierJudge = new DeterministicFrontierJudge(frontierContradictions);

    const passReport = await calibrateJudge(systemOneJudge, frontierJudge, items, {
      spendLedger: ledger,
      split: "dev",
      outputPath: passReportPath
    });

    expect(passReport.passed).toBe(true);
    expect(passReport.contradictionAgreement).toBe(1);
    expect(passReport.sampleSize).toBe(10);
    const persisted = CalibrationReportSchema.parse(
      JSON.parse(readFileSync(passReportPath, "utf8"))
    );
    expect(persisted.passed).toBe(true);

    const systemOneEntries = ledger
      .readEntries()
      .filter((entry) => entry.model === "clef:27b-mxfp8");
    expect(systemOneEntries).toHaveLength(10);
    expect(systemOneEntries.every((entry) => entry.outputTokens === 0 && entry.usd === 0)).toBe(
      true
    );

    decisionProbs.set("Calibration question 0", 0.05);
    decisionProbs.set("Calibration question 1", 0.95);

    const failReport = await calibrateJudge(systemOneJudge, frontierJudge, items, {
      spendLedger: ledger,
      split: "dev"
    });

    expect(failReport.passed).toBe(false);
    expect(failReport.contradictionAgreement).toBeCloseTo(0.8, 6);
  });
});
