import { afterEach, describe, expect, it, vi } from "vitest";
import { KnowledgeStore } from "../../src/adapters/knowledge-store.js";
import type {
  LlmAgentStepInput,
  LlmAgentStepOutput,
  LlmProvider,
  LlmSynthesizeInput,
  LlmSynthesizeOutput
} from "../../src/adapters/llm-provider.js";
import type {
  TelemetryObservationContext,
  TelemetryObservationOptions,
  TelemetryProvider
} from "../../src/adapters/telemetry.js";
import {
  ProviderError,
  type AgentSynthesis,
  type NormalizedDocument
} from "../../src/models/corpus.js";
import { ValidationError } from "../../src/models/errors.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import {
  answerQuestion,
  type AnswerTraceEvent
} from "../../src/usecases/answer.js";
import { extractGraphArtifacts } from "../../src/utils/graph.js";

interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(error: Error): void;
}

function createDeferred<T>(): Deferred<T> {
  let resolveFn: (value: T) => void = () => undefined;
  let rejectFn: (error: Error) => void = () => undefined;
  const promise = new Promise<T>((resolve, reject) => {
    resolveFn = resolve;
    rejectFn = reject;
  });
  return { promise, resolve: resolveFn, reject: rejectFn };
}

type StepHandler = (
  input: LlmAgentStepInput,
  signal?: AbortSignal
) => Promise<LlmAgentStepOutput> | LlmAgentStepOutput;

class ScriptedLlmProvider implements LlmProvider {
  public readonly receivedInputs: LlmSynthesizeInput[] = [];
  public readonly receivedStepInputs: LlmAgentStepInput[] = [];
  private readonly respond: (input: LlmSynthesizeInput) => AgentSynthesis;
  private readonly stepHandlers: StepHandler[];
  private readonly synthesisReasoningText: string | undefined;
  private stepCursor = 0;

  public constructor(
    respond: (input: LlmSynthesizeInput) => AgentSynthesis,
    stepHandlers: StepHandler[] = [],
    synthesisReasoningText?: string
  ) {
    this.respond = respond;
    this.stepHandlers = stepHandlers;
    this.synthesisReasoningText = synthesisReasoningText;
  }

  public async synthesize(
    input: LlmSynthesizeInput
  ): Promise<LlmSynthesizeOutput> {
    this.receivedInputs.push(input);
    return {
      result: this.respond(input),
      usage: { inputTokens: 120, outputTokens: 35 },
      model: input.model,
      ...(this.synthesisReasoningText !== undefined
        ? { reasoningText: this.synthesisReasoningText }
        : {})
    };
  }

  public async runAgentStep(
    input: LlmAgentStepInput,
    signal?: AbortSignal
  ): Promise<LlmAgentStepOutput> {
    this.receivedStepInputs.push({
      model: input.model,
      system: input.system,
      messages: structuredClone(input.messages),
      tools: [...input.tools]
    });

    const handler = this.stepHandlers[this.stepCursor];
    this.stepCursor += 1;
    if (handler === undefined) {
      return {
        text: "Done gathering evidence.",
        toolCalls: [],
        usage: { inputTokens: 40, outputTokens: 10 },
        model: input.model
      };
    }
    return await handler(input, signal);
  }
}

const openStores: KnowledgeStore[] = [];

function createSeededStore(): KnowledgeStore {
  const store = new KnowledgeStore(":memory:");
  openStores.push(store);

  const docs: NormalizedDocument[] = [
    {
      datasetDocUuid: "dsid_vendor_choice",
      contentHash: "hash-1",
      source: "hubspot",
      title: "Magnetar Edgeworks Deal Notes",
      author: "sam",
      status: "closed_lost",
      createdAt: "2025-01-15",
      updatedAt: "2025-01-20",
      metadataJson: JSON.stringify({ account: "Magnetar Edgeworks", region: "west" }),
      metadataText: "source: hubspot\naccount: Magnetar Edgeworks",
      fullText:
        "Magnetar Edgeworks selected AetherAI after an executive demo citing turnkey SSO/SAML mapping."
    },
    {
      datasetDocUuid: "dsid_unrelated_policy",
      contentHash: "hash-2",
      source: "confluence",
      title: "Travel Reimbursement Policy",
      author: "hr",
      status: "approved",
      createdAt: "2024-06-01",
      updatedAt: "2024-06-01",
      metadataJson: JSON.stringify({ team: "finance" }),
      metadataText: "source: confluence\nteam: finance",
      fullText:
        "Domestic meal expenses are capped at 75 dollars per travel day."
    },
    {
      datasetDocUuid: "dsid_deal_2",
      contentHash: "hash-3",
      source: "hubspot",
      title: "Orion Health Deal Notes",
      author: "sam",
      status: "closed_lost",
      createdAt: "2025-02-01",
      updatedAt: "2025-02-10",
      metadataJson: JSON.stringify({ account: "Orion Health", region: "west" }),
      metadataText: "source: hubspot\naccount: Orion Health",
      fullText:
        "Orion Health paused procurement due to budget freeze."
    },
    {
      datasetDocUuid: "dsid_deal_3",
      contentHash: "hash-4",
      source: "hubspot",
      title: "Vega Systems Deal Notes",
      author: "sam",
      status: "closed_lost",
      createdAt: "2025-02-15",
      updatedAt: "2025-02-22",
      metadataJson: JSON.stringify({ account: "Vega Systems", region: "west" }),
      metadataText: "source: hubspot\naccount: Vega Systems",
      fullText:
        "Vega Systems chose an in-house build instead of purchasing."
    }
  ];

  for (const doc of docs) {
    store.upsertDocument(doc, [
      {
        datasetDocUuid: doc.datasetDocUuid,
        contentHash: doc.contentHash,
        chunkIndex: 0,
        title: doc.title,
        metadataText: doc.metadataText,
        body: doc.fullText
      }
    ]);
  }
  store.rebuildSchemaCatalog();
  return store;
}

afterEach(() => {
  vi.useRealTimers();
  while (openStores.length > 0) {
    openStores.pop()?.close();
  }
});

describe("answerQuestion (L0 single-pass FTS5)", () => {
  const profile = RunProfileSchema.parse({ name: "l0-fts" });

  it("strips hallucinated and unread citations while preserving valid read citations", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(() => ({
      answer: "Magnetar Edgeworks chose AetherAI for turnkey SSO/SAML mapping.",
      abstained: false,
      supporting_doc_ids: [
        "dsid_vendor_choice",
        "dsid_vendor_choice",
        "dsid_hallucinated_999",
        "dsid_unrelated_policy",
        "   "
      ]
    }));

    const outcome = await answerQuestion(
      store,
      llm,
      profile,
      "Which competitor did Magnetar Edgeworks choose?"
    );

    expect(outcome.abstained).toBe(false);
    expect(outcome.agentTurns).toBe(0);
    expect(outcome.deadlineHit).toBe(false);
    expect(outcome.retrievedDocIds).toEqual(["dsid_vendor_choice"]);
    expect(outcome.response).toEqual({
      answer: "Magnetar Edgeworks chose AetherAI for turnkey SSO/SAML mapping.",
      document_ids: ["dsid_vendor_choice"]
    });
    expect(outcome.usage).toEqual({ inputTokens: 120, outputTokens: 35 });
    expect(llm.receivedStepInputs).toHaveLength(0);
    expect(llm.receivedInputs[0]?.system).toContain("Governing Record");
    expect(llm.receivedInputs[0]?.system).toContain("Abstention");
  });

  it("forces empty document_ids whenever the model abstains", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(() => ({
      answer: "The corpus does not specify the annual contract discount.",
      abstained: true,
      supporting_doc_ids: ["dsid_vendor_choice"]
    }));

    const outcome = await answerQuestion(
      store,
      llm,
      profile,
      "What discount did Magnetar Edgeworks receive?"
    );

    expect(outcome.abstained).toBe(true);
    expect(outcome.response.document_ids).toEqual([]);
  });

  it("forces empty document_ids when no documents are retrieved", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(() => ({
      answer: "No corpus documents mention quantum flux capacitors.",
      abstained: false,
      supporting_doc_ids: ["dsid_vendor_choice"]
    }));

    const outcome = await answerQuestion(
      store,
      llm,
      profile,
      "What is the warranty on quantum flux capacitors?"
    );

    expect(outcome.retrievedDocIds).toEqual([]);
    expect(outcome.response.document_ids).toEqual([]);
  });

  it("rejects empty questions with ValidationError", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(() => ({
      answer: "unused",
      abstained: true,
      supporting_doc_ids: []
    }));

    await expect(answerQuestion(store, llm, profile, "   ")).rejects.toThrow(
      ValidationError
    );
  });
});

describe("answerQuestion (L1 bounded agent loop)", () => {
  const l1Profile = RunProfileSchema.parse({
    name: "l1-filters",
    enabledTools: ["search", "fetch_docs", "filter_metadata"],
    maxAgentTurns: 6,
    maxFetchedDocs: 15,
    softDeadlineMs: 90000
  });

  it("runs search -> fetch_docs -> synthesis with Schema Catalog injection and citation validation", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Magnetar Edgeworks selected AetherAI for turnkey SSO/SAML mapping.",
        abstained: false,
        supporting_doc_ids: [
          "dsid_vendor_choice",
          "dsid_unrelated_policy",
          "dsid_nonexistent_999"
        ]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "call-1",
              toolName: "search",
              input: { query: "Magnetar Edgeworks competitor" }
            }
          ],
          usage: { inputTokens: 80, outputTokens: 20 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "call-2",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_vendor_choice"] }
            }
          ],
          usage: { inputTokens: 100, outputTokens: 25 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "Found Magnetar Edgeworks decision.",
          toolCalls: [],
          usage: { inputTokens: 110, outputTokens: 15 },
          model: l1Profile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l1Profile,
      "Which competitor did Magnetar Edgeworks select?"
    );

    expect(outcome.abstained).toBe(false);
    expect(outcome.agentTurns).toBe(2);
    expect(outcome.deadlineHit).toBe(false);
    expect(outcome.response).toEqual({
      answer: "Magnetar Edgeworks selected AetherAI for turnkey SSO/SAML mapping.",
      document_ids: ["dsid_vendor_choice"]
    });
    expect(outcome.usage).toEqual({ inputTokens: 410, outputTokens: 95 });

    const synthPrompt = llm.receivedInputs[0]?.prompt ?? "";
    expect(synthPrompt).toContain("dsid_vendor_choice");
    expect(synthPrompt).toContain("turnkey SSO/SAML mapping");

    const firstStep = llm.receivedStepInputs[0];
    expect(firstStep?.tools.map((tool) => tool.name)).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata"
    ]);
    expect(firstStep?.system).toContain('source="hubspot"');
    expect(firstStep?.system).toContain("account");
    expect(firstStep?.system).toContain("Governing Record");
  });

  it("supports filter_metadata exact totalCount and verifies citations returned by filter_metadata", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Sam owns 3 closed_lost hubspot deals in the west region.",
        abstained: false,
        supporting_doc_ids: [
          "dsid_deal_3",
          "dsid_deal_2",
          "dsid_vendor_choice",
          "dsid_fabricated_id"
        ]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "filter-1",
              toolName: "filter_metadata",
              input: {
                source: "hubspot",
                author: "sam",
                status: "closed_lost",
                metadataFilters: [{ key: "region", op: "eq", value: "west" }],
                limit: 2
              }
            }
          ],
          usage: { inputTokens: 70, outputTokens: 20 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "Total count is 3 with 2 returned rows.",
          toolCalls: [],
          usage: { inputTokens: 90, outputTokens: 15 },
          model: l1Profile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l1Profile,
      "How many closed_lost hubspot deals does Sam own in the west region?"
    );

    const synthPrompt = llm.receivedInputs[0]?.prompt ?? "";
    expect(synthPrompt).toContain("totalCount: 3");
    expect(synthPrompt).toContain("dsid_deal_3");
    expect(synthPrompt).toContain("dsid_deal_2");
    expect(outcome.abstained).toBe(false);
    expect(outcome.agentTurns).toBe(1);
    expect(outcome.response.document_ids).toEqual(["dsid_deal_3", "dsid_deal_2"]);
  });

  it("seeds initial candidate snippets before turn 1 when preSearchEnabled is true", async () => {
    const store = createSeededStore();
    const preSearchProfile = RunProfileSchema.parse({
      name: "l1-presearch",
      preSearchEnabled: true,
      enabledTools: ["search", "fetch_docs"]
    });

    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Domestic meal expenses are capped at 75 dollars per travel day.",
        abstained: false,
        supporting_doc_ids: ["dsid_unrelated_policy"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "fetch-1",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_unrelated_policy"] }
            }
          ],
          usage: { inputTokens: 90, outputTokens: 15 },
          model: preSearchProfile.routingModel
        }),
        () => ({
          text: "Read policy doc.",
          toolCalls: [],
          usage: { inputTokens: 80, outputTokens: 10 },
          model: preSearchProfile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      preSearchProfile,
      "What is the domestic meal expense cap?"
    );

    const initialUser = llm.receivedStepInputs[0]?.messages[0];
    expect(initialUser).toEqual(
      expect.objectContaining({
        role: "user",
        content: expect.stringContaining("Pre-search candidate snippets")
      })
    );
    expect(initialUser).toEqual(
      expect.objectContaining({
        role: "user",
        content: expect.stringContaining("dsid_unrelated_policy")
      })
    );
    expect(outcome.agentTurns).toBe(1);
    expect(outcome.response.document_ids).toEqual(["dsid_unrelated_policy"]);
    expect(llm.receivedStepInputs[0]?.tools.map((tool) => tool.name)).toEqual([
      "search",
      "fetch_docs"
    ]);
  });

  it("enforces maxAgentTurns cap and synthesizes immediately after reaching the turn limit", async () => {
    const store = createSeededStore();
    const cappedTurnsProfile = RunProfileSchema.parse({
      name: "l1-turn-cap",
      enabledTools: ["search", "fetch_docs"],
      maxAgentTurns: 2
    });

    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Magnetar Edgeworks chose AetherAI.",
        abstained: false,
        supporting_doc_ids: ["dsid_vendor_choice"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "t1",
              toolName: "search",
              input: { query: "Magnetar" }
            }
          ],
          usage: { inputTokens: 50, outputTokens: 10 },
          model: cappedTurnsProfile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "t2",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_vendor_choice"] }
            }
          ],
          usage: { inputTokens: 60, outputTokens: 10 },
          model: cappedTurnsProfile.routingModel
        }),
        () => {
          throw new Error("Should not execute a 3rd turn past maxAgentTurns");
        }
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      cappedTurnsProfile,
      "Who won Magnetar Edgeworks?"
    );

    expect(outcome.agentTurns).toBe(2);
    expect(llm.receivedStepInputs).toHaveLength(2);
    expect(llm.receivedInputs).toHaveLength(1);
    expect(outcome.response.document_ids).toEqual(["dsid_vendor_choice"]);
  });

  it("enforces maxFetchedDocs cap and blocks reading or citing docs beyond budget", async () => {
    const store = createSeededStore();
    const cappedDocsProfile = RunProfileSchema.parse({
      name: "l1-doc-cap",
      enabledTools: ["fetch_docs"],
      maxAgentTurns: 4,
      maxFetchedDocs: 2
    });

    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Reviewed two deals within budget.",
        abstained: false,
        supporting_doc_ids: ["dsid_vendor_choice", "dsid_deal_2", "dsid_deal_3"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "f1",
              toolName: "fetch_docs",
              input: {
                doc_ids: ["dsid_vendor_choice", "dsid_deal_2", "dsid_deal_3"]
              }
            }
          ],
          usage: { inputTokens: 40, outputTokens: 10 },
          model: cappedDocsProfile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "f2",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_deal_3"] }
            }
          ],
          usage: { inputTokens: 40, outputTokens: 10 },
          model: cappedDocsProfile.routingModel
        }),
        () => ({
          text: "Budget exhausted.",
          toolCalls: [],
          usage: { inputTokens: 30, outputTokens: 10 },
          model: cappedDocsProfile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      cappedDocsProfile,
      "Summarize the deals."
    );

    const step2Messages = llm.receivedStepInputs[1]?.messages ?? [];
    const firstToolMsg = step2Messages[step2Messages.length - 1];
    expect(firstToolMsg).toEqual(
      expect.objectContaining({
        role: "tool",
        results: [
          expect.objectContaining({
            toolName: "fetch_docs",
            output: expect.objectContaining({
              totalFetchedSoFar: 2,
              remainingDocBudget: 0,
              truncatedByBudget: true
            })
          })
        ]
      })
    );
    expect(outcome.retrievedDocIds).toEqual(["dsid_vendor_choice", "dsid_deal_2"]);
    expect(outcome.response.document_ids).toEqual([
      "dsid_vendor_choice",
      "dsid_deal_2"
    ]);
  });

  it("aborts remaining tool turns when softDeadlineMs fires and runs one final synthesis", async () => {
    vi.useFakeTimers();
    const store = createSeededStore();
    const deadlineProfile = RunProfileSchema.parse({
      name: "l1-deadline",
      enabledTools: ["fetch_docs", "search"],
      maxAgentTurns: 6,
      softDeadlineMs: 500
    });

    const turn2Started = createDeferred<void>();

    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Magnetar Edgeworks selected AetherAI.",
        abstained: false,
        supporting_doc_ids: ["dsid_vendor_choice"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "fetch-before-deadline",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_vendor_choice"] }
            }
          ],
          usage: { inputTokens: 50, outputTokens: 15 },
          model: deadlineProfile.routingModel
        }),
        (_stepInput, signal) =>
          new Promise<LlmAgentStepOutput>((_resolve, reject) => {
            turn2Started.resolve();
            signal?.addEventListener(
              "abort",
              () => {
                reject(
                  new ProviderError("Aborted by soft deadline", {
                    cause: signal.reason
                  })
                );
              },
              { once: true }
            );
          }),
        () => {
          throw new Error("Must never start another tool turn after deadline");
        }
      ]
    );

    const pendingAnswer = answerQuestion(
      store,
      llm,
      deadlineProfile,
      "Which vendor did Magnetar Edgeworks pick?"
    );

    await turn2Started.promise;
    await vi.advanceTimersByTimeAsync(deadlineProfile.softDeadlineMs);

    const outcome = await pendingAnswer;
    const synthPrompt = llm.receivedInputs[0]?.prompt ?? "";
    expect(synthPrompt).toContain("dsid_vendor_choice");
    expect(synthPrompt).toContain("time budget elapsed");
    expect(outcome.deadlineHit).toBe(true);
    expect(outcome.agentTurns).toBe(1);
    expect(llm.receivedStepInputs).toHaveLength(2);
    expect(llm.receivedInputs).toHaveLength(1);
    expect(outcome.response).toEqual({
      answer: "Magnetar Edgeworks selected AetherAI.",
      document_ids: ["dsid_vendor_choice"]
    });
  });

  it("forces empty document_ids when L1 synthesis abstains", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "The corpus does not contain pricing discount percentages.",
        abstained: true,
        supporting_doc_ids: ["dsid_vendor_choice"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "fetch-1",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_vendor_choice"] }
            }
          ],
          usage: { inputTokens: 40, outputTokens: 10 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "No discount mentioned.",
          toolCalls: [],
          usage: { inputTokens: 40, outputTokens: 10 },
          model: l1Profile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l1Profile,
      "What discount percentage was offered to Magnetar Edgeworks?"
    );

    expect(outcome.abstained).toBe(true);
    expect(outcome.response.document_ids).toEqual([]);
  });

  it("returns structured tool errors for disabled tools or malformed filter_metadata keys so the agent can recover", async () => {
    const store = createSeededStore();
    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "Magnetar Edgeworks selected AetherAI.",
        abstained: false,
        supporting_doc_ids: ["dsid_vendor_choice"]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "bad-tool",
              toolName: "expand_graph",
              input: { doc_ids: ["dsid_vendor_choice"] }
            },
            {
              toolCallId: "bad-key",
              toolName: "filter_metadata",
              input: {
                metadataFilters: [
                  { key: "account') OR 1=1 --", op: "eq", value: "x" }
                ]
              }
            }
          ],
          usage: { inputTokens: 50, outputTokens: 15 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "recover-fetch",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_vendor_choice"] }
            }
          ],
          usage: { inputTokens: 60, outputTokens: 15 },
          model: l1Profile.routingModel
        }),
        () => ({
          text: "Recovered after tool errors.",
          toolCalls: [],
          usage: { inputTokens: 60, outputTokens: 10 },
          model: l1Profile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l1Profile,
      "Which vendor did Magnetar Edgeworks select?"
    );

    const step2Messages = llm.receivedStepInputs[1]?.messages ?? [];
    const errorToolMsg = step2Messages[step2Messages.length - 1];
    expect(errorToolMsg).toEqual(
      expect.objectContaining({
        role: "tool",
        results: [
          expect.objectContaining({
            toolCallId: "bad-tool",
            isError: true
          }),
          expect.objectContaining({
            toolCallId: "bad-key",
            isError: true
          })
        ]
      })
    );
    expect(outcome.agentTurns).toBe(2);
    expect(outcome.response.document_ids).toEqual(["dsid_vendor_choice"]);
  });
});

describe("answerQuestion (L3 Link Graph expand_graph tool)", () => {
  const l3Profile = RunProfileSchema.parse({
    name: "l3-graph-test",
    enableEmbeddings: false,
    enabledTools: ["search", "fetch_docs", "filter_metadata", "expand_graph"],
    maxAgentTurns: 6,
    maxFetchedDocs: 15,
    softDeadlineMs: 90000
  });

  function createGraphSeededStore(): KnowledgeStore {
    const store = new KnowledgeStore(":memory:");
    openStores.push(store);

    const rawDocs = [
      {
        source: "jira",
        raw: {
          dataset_doc_uuid: "dsid_jira_saml",
          key: "ENG-701",
          summary: "SAML group sync escalation",
          author: "Aisha Khan",
          status: "open",
          updated_at: "2026-03-10",
          linked_issues: ["ENG-702"],
          company_name: "Magnetar Edgeworks",
          description: "Customer requires SAML group sync before go-live."
        },
        doc: {
          datasetDocUuid: "dsid_jira_saml",
          contentHash: "hash-g1",
          source: "jira",
          title: "SAML group sync escalation",
          author: "Aisha Khan",
          status: "open",
          createdAt: "2026-03-09",
          updatedAt: "2026-03-10",
          metadataJson: JSON.stringify({
            key: "ENG-701",
            company_name: "Magnetar Edgeworks"
          }),
          metadataText: "source: jira\nkey: ENG-701",
          fullText: "Customer requires SAML group sync before go-live."
        }
      },
      {
        source: "jira",
        raw: {
          dataset_doc_uuid: "dsid_jira_fix",
          key: "ENG-702",
          summary: "Shipped SAML group mapping patch",
          author: "Elena Park",
          status: "done",
          updated_at: "2026-03-12",
          company_name: "Magnetar Edgeworks",
          description:
            "Patch v2.4.1 resolved SAML group mapping with zero downtime."
        },
        doc: {
          datasetDocUuid: "dsid_jira_fix",
          contentHash: "hash-g2",
          source: "jira",
          title: "Shipped SAML group mapping patch",
          author: "Elena Park",
          status: "done",
          createdAt: "2026-03-11",
          updatedAt: "2026-03-12",
          metadataJson: JSON.stringify({
            key: "ENG-702",
            company_name: "Magnetar Edgeworks"
          }),
          metadataText: "source: jira\nkey: ENG-702",
          fullText:
            "Patch v2.4.1 resolved SAML group mapping with zero downtime."
        }
      },
      {
        source: "hubspot",
        raw: {
          dataset_doc_uuid: "dsid_hs_unfetched",
          company_id: "HS-901",
          company_name: "Magnetar Edgeworks",
          owner: "Markus Klein",
          stage: "closed_won",
          updated_at: "2026-03-15",
          crm_notes: "Account notes."
        },
        doc: {
          datasetDocUuid: "dsid_hs_unfetched",
          contentHash: "hash-g3",
          source: "hubspot",
          title: "Magnetar Edgeworks",
          author: "Markus Klein",
          status: "closed_won",
          createdAt: "2026-03-01",
          updatedAt: "2026-03-15",
          metadataJson: JSON.stringify({
            company_id: "HS-901",
            company_name: "Magnetar Edgeworks"
          }),
          metadataText: "source: hubspot\ncompany_name: Magnetar Edgeworks",
          fullText: "Account notes."
        }
      }
    ];

    for (const entry of rawDocs) {
      const artifacts = extractGraphArtifacts({
        datasetDocUuid: entry.doc.datasetDocUuid,
        contentHash: entry.doc.contentHash,
        source: entry.source,
        raw: entry.raw,
        title: entry.doc.title,
        bodyText: entry.doc.fullText
      });
      store.upsertDocument(
        entry.doc,
        [
          {
            datasetDocUuid: entry.doc.datasetDocUuid,
            contentHash: entry.doc.contentHash,
            chunkIndex: 0,
            title: entry.doc.title,
            metadataText: entry.doc.metadataText,
            body: entry.doc.fullText
          }
        ],
        artifacts
      );
    }
    store.resolveGraphEdges();
    store.rebuildSchemaCatalog();
    return store;
  }

  it("exposes expand_graph only when enabled, discovers a linked 2nd doc, and requires fetch_docs before citing", async () => {
    const store = createGraphSeededStore();
    const llm = new ScriptedLlmProvider(
      () => ({
        answer:
          "The SAML escalation (ENG-701) was fixed in patch v2.4.1 (ENG-702).",
        abstained: false,
        supporting_doc_ids: [
          "dsid_jira_saml",
          "dsid_jira_fix",
          "dsid_hs_unfetched"
        ]
      }),
      [
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "l3-search",
              toolName: "search",
              input: { query: "SAML group sync escalation" }
            }
          ],
          usage: { inputTokens: 60, outputTokens: 15 },
          model: l3Profile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "l3-expand-invalid",
              toolName: "expand_graph",
              input: { maxHops: 1 }
            },
            {
              toolCallId: "l3-expand",
              toolName: "expand_graph",
              input: { docId: "dsid_jira_saml", maxHops: 1, limit: 10 }
            }
          ],
          usage: { inputTokens: 80, outputTokens: 20 },
          model: l3Profile.routingModel
        }),
        () => ({
          text: "",
          toolCalls: [
            {
              toolCallId: "l3-fetch",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_jira_saml", "dsid_jira_fix"] }
            }
          ],
          usage: { inputTokens: 90, outputTokens: 20 },
          model: l3Profile.routingModel
        }),
        () => ({
          text: "Gathered escalation ticket and linked fix ticket.",
          toolCalls: [],
          usage: { inputTokens: 100, outputTokens: 15 },
          model: l3Profile.routingModel
        })
      ]
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l3Profile,
      "Which patch resolved the SAML group sync escalation?"
    );

    const firstStep = llm.receivedStepInputs[0];
    expect(firstStep?.tools.map((tool) => tool.name)).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata",
      "expand_graph"
    ]);
    expect(firstStep?.system).toContain("expand_graph");

    const step3Messages = llm.receivedStepInputs[2]?.messages ?? [];
    const expandToolMsg = step3Messages[step3Messages.length - 1];
    expect(expandToolMsg).toEqual(
      expect.objectContaining({
        role: "tool",
        results: [
          expect.objectContaining({
            toolCallId: "l3-expand-invalid",
            isError: true
          }),
          expect.objectContaining({
            toolCallId: "l3-expand",
            output: expect.objectContaining({
              totalNeighbors: 2,
              neighbors: expect.arrayContaining([
                expect.objectContaining({
                  dataset_doc_uuid: "dsid_jira_fix",
                  hop: 1
                }),
                expect.objectContaining({
                  dataset_doc_uuid: "dsid_hs_unfetched",
                  hop: 1
                })
              ])
            })
          })
        ]
      })
    );

    // dsid_hs_unfetched was returned by expand_graph but never fetched via fetch_docs, so it must be stripped!
    expect(outcome.abstained).toBe(false);
    expect(outcome.agentTurns).toBe(3);
    expect(outcome.response.document_ids).toEqual([
      "dsid_jira_saml",
      "dsid_jira_fix"
    ]);
    expect(outcome.strippedDocIds).toEqual(["dsid_hs_unfetched"]);
  });

  it("streams live AnswerTraceEvent sequence and records typed Langfuse observations", async () => {
    const store = createGraphSeededStore();
    const events: AnswerTraceEvent[] = [];
    const recordedObservations: Array<{
      name: string;
      asType: string;
      input: unknown;
      metadata: Readonly<Record<string, unknown>> | undefined;
      output: unknown;
    }> = [];
    const fakeTraceId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    const recordingTelemetry: TelemetryProvider = {
      enabled: true,
      getAiSdkTelemetry(functionId) {
        return { isEnabled: true, functionId };
      },
      getActiveTraceId() {
        return fakeTraceId;
      },
      async startObservation<T>(
        obsOptions: TelemetryObservationOptions,
        fn: (ctx: TelemetryObservationContext) => Promise<T>
      ): Promise<T> {
        const entry = {
          name: obsOptions.name,
          asType: obsOptions.asType,
          input: obsOptions.input,
          metadata: obsOptions.metadata,
          output: undefined as unknown
        };
        recordedObservations.push(entry);
        return fn({
          traceId: fakeTraceId,
          setOutput(output: unknown): void {
            entry.output = output;
          }
        });
      }
    };

    const llm = new ScriptedLlmProvider(
      () => ({
        answer: "ENG-701 was resolved by patch v2.4.1 in ENG-702.",
        abstained: false,
        supporting_doc_ids: [
          "dsid_jira_saml",
          "dsid_jira_fix",
          "dsid_hs_unfetched",
          "dsid_fabricated"
        ]
      }),
      [
        () => ({
          text: "",
          reasoningText: "Need to find SAML escalation and linked fix.",
          toolCalls: [
            {
              toolCallId: "ev-search",
              toolName: "search",
              input: { query: "SAML group sync escalation" }
            }
          ],
          usage: { inputTokens: 60, outputTokens: 15 },
          model: l3Profile.routingModel
        }),
        () => ({
          text: "",
          reasoningText: "Expand graph from ENG-701 and check Jira metadata.",
          toolCalls: [
            {
              toolCallId: "ev-expand",
              toolName: "expand_graph",
              input: { docId: "dsid_jira_saml", maxHops: 1, limit: 10 }
            },
            {
              toolCallId: "ev-filter",
              toolName: "filter_metadata",
              input: { source: "jira", status: "done" }
            },
            {
              toolCallId: "ev-fetch",
              toolName: "fetch_docs",
              input: { doc_ids: ["dsid_jira_saml", "dsid_jira_fix"] }
            }
          ],
          usage: { inputTokens: 90, outputTokens: 20 },
          model: l3Profile.routingModel
        }),
        () => ({
          text: "Verified both tickets.",
          toolCalls: [],
          usage: { inputTokens: 70, outputTokens: 10 },
          model: l3Profile.routingModel
        })
      ],
      "Synthesizing final grounded answer from both Jira tickets."
    );

    const outcome = await answerQuestion(
      store,
      llm,
      l3Profile,
      "Which patch resolved the SAML group sync escalation?",
      {
        telemetry: recordingTelemetry,
        onEvent(event) {
          events.push(event);
        }
      }
    );

    expect(outcome.traceId).toBe(fakeTraceId);
    expect(outcome.strippedDocIds).toEqual(["dsid_hs_unfetched", "dsid_fabricated"]);

    const eventTypes = events.map((event) => event.type);
    expect(eventTypes).toEqual([
      "agent_turn_start",
      "agent_thinking",
      "tool_call",
      "tool_result",
      "agent_turn_start",
      "agent_thinking",
      "tool_call",
      "tool_result",
      "tool_call",
      "tool_result",
      "tool_call",
      "tool_result",
      "agent_turn_start",
      "synthesis_start",
      "synthesis_thinking",
      "citations_validated",
      "answer_complete"
    ]);

    const citationsEvent = events.find((event) => event.type === "citations_validated");
    expect(citationsEvent).toEqual({
      type: "citations_validated",
      validDocIds: ["dsid_jira_saml", "dsid_jira_fix"],
      strippedDocIds: ["dsid_hs_unfetched", "dsid_fabricated"],
      abstained: false
    });

    const completeEvent = events.find((event) => event.type === "answer_complete");
    expect(completeEvent).toEqual(
      expect.objectContaining({
        type: "answer_complete",
        abstained: false,
        documentIds: ["dsid_jira_saml", "dsid_jira_fix"],
        strippedDocIds: ["dsid_hs_unfetched", "dsid_fabricated"],
        agentTurns: 2,
        deadlineHit: false,
        traceId: fakeTraceId
      })
    );

    expect(
      recordedObservations.map((obs) => ({ name: obs.name, asType: obs.asType }))
    ).toEqual([
      { name: "answer-question", asType: "agent" },
      { name: "retrieve-evidence", asType: "retriever" },
      { name: "expand-graph", asType: "tool" },
      { name: "filter-metadata", asType: "tool" },
      { name: "fetch-docs", asType: "tool" }
    ]);

    const rootObs = recordedObservations[0];
    expect(rootObs?.input).toEqual({
      question: "Which patch resolved the SAML group sync escalation?"
    });
    expect(rootObs?.metadata).toEqual(
      expect.objectContaining({
        profile: "l3-graph-test",
        enabledTools: ["search", "fetch_docs", "filter_metadata", "expand_graph"]
      })
    );
    expect(rootObs?.output).toEqual({
      answer: "ENG-701 was resolved by patch v2.4.1 in ENG-702.",
      abstained: false,
      document_ids: ["dsid_jira_saml", "dsid_jira_fix"],
      agentTurns: 2,
      deadlineHit: false
    });
  });
});
