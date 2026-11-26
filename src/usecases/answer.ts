import { performance } from "node:perf_hooks";
import { z } from "zod";
import type { Embedder } from "../adapters/embedder.js";
import type { KnowledgeStore } from "../adapters/knowledge-store.js";
import type {
  LlmAgentMessage,
  LlmProvider,
  LlmTokenUsage,
  LlmToolCall,
  LlmToolDefinition,
  LlmToolResult
} from "../adapters/llm-provider.js";
import type {
  TelemetryObservationContext,
  TelemetryObservationOptions,
  TelemetryProvider
} from "../adapters/telemetry.js";
import {
  AskResponseSchema,
  ExpandGraphQuerySchema,
  MetadataFilterQuerySchema,
  ProviderError,
  type AgentSynthesis,
  type AskResponse,
  type MetadataFilterResult,
  type SchemaCatalog,
  type StoredDocument
} from "../models/corpus.js";
import { ValidationError } from "../models/errors.js";
import {
  computeIndexConfigHash,
  type AgentTool,
  type RunProfile
} from "../models/profile.js";
import type { FusedSearchHit } from "../utils/vector.js";
import { retrieveEvidence } from "./retrieve.js";

export type AnswerTraceEvent =
  | {
      readonly type: "pre_search_done";
      readonly query: string;
      readonly mode: "fts" | "hybrid";
      readonly hitCount: number;
      readonly topDocIds: readonly string[];
      readonly durationMs: number;
    }
  | {
      readonly type: "agent_turn_start";
      readonly turn: number;
      readonly maxTurns: number;
      readonly model: string;
    }
  | {
      readonly type: "agent_thinking";
      readonly turn: number;
      readonly text: string;
    }
  | {
      readonly type: "tool_call";
      readonly turn: number;
      readonly toolCallId: string;
      readonly toolName: string;
      readonly input: unknown;
    }
  | {
      readonly type: "tool_result";
      readonly turn: number;
      readonly toolCallId: string;
      readonly toolName: string;
      readonly isError: boolean;
      readonly durationMs: number;
      readonly summary: string;
      readonly output: unknown;
    }
  | {
      readonly type: "deadline_hit";
      readonly turnsCompleted: number;
      readonly softDeadlineMs: number;
    }
  | {
      readonly type: "synthesis_start";
      readonly model: string;
      readonly fetchedDocCount: number;
      readonly filterSnapshotCount: number;
    }
  | {
      readonly type: "synthesis_thinking";
      readonly text: string;
    }
  | {
      readonly type: "citations_validated";
      readonly validDocIds: readonly string[];
      readonly strippedDocIds: readonly string[];
      readonly abstained: boolean;
    }
  | {
      readonly type: "answer_complete";
      readonly abstained: boolean;
      readonly documentIds: readonly string[];
      readonly strippedDocIds: readonly string[];
      readonly agentTurns: number;
      readonly deadlineHit: boolean;
      readonly usage: LlmTokenUsage;
      readonly embeddingInputTokens: number;
      readonly durationMs: number;
      readonly traceId?: string;
    };

export interface AnswerQuestionOptions {
  readonly signal?: AbortSignal;
  readonly embedder?: Embedder;
  readonly onEvent?: (event: AnswerTraceEvent) => void;
  readonly telemetry?: TelemetryProvider;
}

export interface AnswerResult {
  response: AskResponse;
  abstained: boolean;
  retrievedDocIds: string[];
  strippedDocIds: string[];
  usage: LlmTokenUsage;
  embeddingInputTokens: number;
  model: string;
  agentTurns: number;
  deadlineHit: boolean;
  traceId?: string;
}

interface FilterSnapshot {
  readonly query: Record<string, unknown>;
  readonly result: MetadataFilterResult;
}

const MAX_PROMPT_DOC_CHARS = 24000;
const MAX_TOOL_DOC_CHARS = 12000;
const MAX_SEARCH_TOOL_TOP_K = 50;
const MAX_SCHEMA_CATALOG_CHARS = 14000;
const FALLBACK_ABSTENTION_TEXT =
  "The corpus does not contain sufficient evidence to answer this question.";

const SearchToolInputSchema = z.object({
  query: z
    .string()
    .trim()
    .min(1)
    .describe("Search keywords or phrases matching titles, metadata, or body text"),
  topK: z
    .number()
    .int()
    .min(1)
    .max(MAX_SEARCH_TOOL_TOP_K)
    .optional()
    .describe("Maximum number of search hits to return")
});

const FetchDocsToolInputSchema = z.object({
  doc_ids: z
    .array(z.string().trim().min(1))
    .min(1)
    .max(25)
    .describe("Exact dataset_doc_uuid values to read in full")
});

const SYNTHESIS_SYSTEM_PROMPT = [
  "You are an enterprise knowledge assistant answering questions strictly from verified Corpus Documents and metadata query results.",
  "Follow these rules:",
  "1. Ground every claim in the provided evidence. Never fabricate facts or infer unsupported details.",
  "2. Governing Record rule: when documents conflict or evolve over time, prefer the most authoritative, up-to-date record based on timestamps (updated_at, created_at), finalized or approved status, and official sources over drafts or informal chat.",
  "3. Strict Abstention rule: if the provided evidence does not contain enough information to answer the question, set abstained to true, state clearly in answer that the corpus lacks sufficient evidence, and return an empty supporting_doc_ids array.",
  "4. Partial unanswerability rule: if only part of a multi-part question is supported, answer the supported part accurately, state clearly which part lacks evidence in the corpus without guessing, and cite only the documents supporting the answered part. If the core question cannot be answered, abstain with an empty supporting_doc_ids array.",
  "5. Citation precision: in supporting_doc_ids, include only the exact dataset_doc_uuid values of verified documents that directly support your answer. Do not cite irrelevant documents, file paths, or titles. When a question asks to count or enumerate records matching a filter (via filter_metadata), include all matching dataset_doc_uuid values returned by that filter that support the count, not only a subset described in detail."
].join("\n");

function buildAgentSystemPrompt(
  schemaCatalogText: string,
  profile: RunProfile
): string {
  const hasExpandGraph = profile.enabledTools.includes("expand_graph");
  const toolGuidance = hasExpandGraph
    ? "1. Use search to find candidate documents, expand_graph to follow deterministic cross-source links or shared entities (accounts, deals, people, competitors, products, labels), fetch_docs to read full document contents before relying on or citing details, and filter_metadata for structured counts, date ranges, owners, statuses, sources, or metadata fields."
    : "1. Use search to find candidate documents, fetch_docs to read full document contents before relying on details, and filter_metadata for structured counts, date ranges, owners, statuses, sources, or metadata fields.";

  return [
    "You are an enterprise knowledge retrieval agent. Use the provided tools to gather verified evidence from the corpus before answering.",
    "Follow these operating rules:",
    toolGuidance,
    "2. Stay strictly within the budget: at most " +
      `${profile.maxAgentTurns} tool turns and at most ${profile.maxFetchedDocs} full documents fetched via fetch_docs.`,
    "3. Governing Record rule: resolve conflicting or evolving facts by checking timestamps, finalized status, and official records over casual chat or drafts.",
    "4. Abstention and partial unanswerability: if the corpus lacks evidence for the question (or for part of a multi-part question), stop searching once you have confirmed the gap so synthesis can abstain or state the unsupported part clearly.",
    "5. Before stopping, verify that you have checked every sub-question or distinct aspect of the user's request. When you have enough verified evidence (or have confirmed the corpus lacks the answer), stop calling tools and summarize your findings briefly so the final synthesis step can produce the structured answer.",
    "",
    "Corpus Schema Catalog (discovered exact sources and metadata_json keys):",
    schemaCatalogText
  ].join("\n");
}

function formatSchemaCatalog(catalog: SchemaCatalog): string {
  if (catalog.sources.length === 0) {
    return "(Empty corpus: no sources indexed.)";
  }

  const perSourceBudget = Math.max(
    450,
    Math.floor(MAX_SCHEMA_CATALOG_CHARS / catalog.sources.length)
  );
  const lines: string[] = [];

  for (const entry of catalog.sources) {
    const prefix = `- source="${entry.source}" (${entry.docCount} docs): keys=[`;
    const suffix = "]";
    const keyParts: string[] = [];
    let currentLen = prefix.length + suffix.length;

    for (const key of entry.metadataKeys) {
      const samples = entry.sampleValues[key];
      const renderedKey =
        samples !== undefined && samples.length > 0
          ? `${key} (${samples.slice(0, 12).join(" | ")})`
          : key;
      const separatorLen = keyParts.length > 0 ? 2 : 0;
      if (keyParts.length > 0 && currentLen + separatorLen + renderedKey.length + 5 > perSourceBudget) {
        keyParts.push("...");
        break;
      }
      keyParts.push(renderedKey);
      currentLen += separatorLen + renderedKey.length;
    }

    lines.push(`${prefix}${keyParts.join(", ")}${suffix}`);
  }

  return lines.join("\n");
}

function formatDocForPrompt(
  doc: StoredDocument,
  index: number,
  maxChars = MAX_PROMPT_DOC_CHARS
): string {
  const truncatedText =
    doc.fullText.length > maxChars
      ? `${doc.fullText.slice(0, maxChars)}\n[truncated]`
      : doc.fullText;

  const metaLines = [
    `[Document ${index + 1}]`,
    `dataset_doc_uuid: ${doc.datasetDocUuid}`,
    `source: ${doc.source}`,
    `title: ${doc.title}`
  ];
  if (doc.author !== null) {
    metaLines.push(`author: ${doc.author}`);
  }
  if (doc.status !== null) {
    metaLines.push(`status: ${doc.status}`);
  }
  if (doc.createdAt !== null) {
    metaLines.push(`created_at: ${doc.createdAt}`);
  }
  if (doc.updatedAt !== null) {
    metaLines.push(`updated_at: ${doc.updatedAt}`);
  }

  return `${metaLines.join("\n")}\n---\n${truncatedText}`;
}

function formatPreSearchHits(hits: readonly FusedSearchHit[]): string {
  if (hits.length === 0) {
    return "(Pre-search returned no candidate snippets.)";
  }
  return hits
    .map(
      (hit, index) =>
        `${index + 1}. dataset_doc_uuid=${hit.datasetDocUuid} | source=${hit.source} | title=${hit.title} | excerpt=${hit.excerpt}`
    )
    .join("\n");
}

function formatFilterSnapshots(snapshots: readonly FilterSnapshot[]): string {
  if (snapshots.length === 0) {
    return "";
  }

  const sections = snapshots.map((snapshot, index) => {
    const rows = snapshot.result.documents.map((doc) => {
      const parts = [
        `dataset_doc_uuid=${doc.datasetDocUuid}`,
        `source=${doc.source}`,
        `title=${doc.title}`
      ];
      if (doc.author !== null) {
        parts.push(`author=${doc.author}`);
      }
      if (doc.status !== null) {
        parts.push(`status=${doc.status}`);
      }
      if (doc.createdAt !== null) {
        parts.push(`created_at=${doc.createdAt}`);
      }
      if (doc.updatedAt !== null) {
        parts.push(`updated_at=${doc.updatedAt}`);
      }
      return `- ${parts.join(" | ")}`;
    });

    return [
      `[Metadata Filter ${index + 1}]`,
      `query: ${JSON.stringify(snapshot.query)}`,
      `totalCount: ${snapshot.result.totalCount}`,
      `returnedRows: ${snapshot.result.documents.length}`,
      ...(rows.length > 0 ? rows : ["(No matching documents)"])
    ].join("\n");
  });

  return sections.join("\n\n");
}

function buildL0SynthesisPrompt(
  question: string,
  docs: readonly StoredDocument[]
): string {
  const evidenceBlock =
    docs.length === 0
      ? "(No matching Corpus Documents were retrieved. You must abstain and return supporting_doc_ids: [].)"
      : docs.map((doc, index) => formatDocForPrompt(doc, index)).join("\n\n");

  return `Question:\n${question}\n\nRetrieved Corpus Documents:\n${evidenceBlock}`;
}

function buildAgentSynthesisPrompt(params: {
  question: string;
  fetchedDocs: readonly StoredDocument[];
  filterSnapshots: readonly FilterSnapshot[];
  agentNotes: string;
  deadlineHit: boolean;
  hasObservedEvidence: boolean;
}): string {
  const sections: string[] = [`Question:\n${params.question}`];

  if (params.deadlineHit) {
    sections.push(
      "Notice: The retrieval time budget elapsed before further tool calls. Synthesize strictly from the verified evidence already gathered below, or abstain if insufficient."
    );
  }

  if (params.fetchedDocs.length > 0) {
    sections.push(
      `Fetched Corpus Documents (${params.fetchedDocs.length}):\n${params.fetchedDocs
        .map((doc, index) => formatDocForPrompt(doc, index))
        .join("\n\n")}`
    );
  }

  if (params.filterSnapshots.length > 0) {
    sections.push(
      `Structured Metadata Query Results:\n${formatFilterSnapshots(params.filterSnapshots)}`
    );
  }

  if (!params.hasObservedEvidence) {
    sections.push(
      "(No verified Corpus Documents or matching metadata rows were gathered. You must abstain and return supporting_doc_ids: [].)"
    );
  }

  const trimmedNotes = params.agentNotes.trim();
  if (trimmedNotes.length > 0) {
    sections.push(`Agent Retrieval Notes:\n${trimmedNotes}`);
  }

  return sections.join("\n\n");
}

function collectTopUniqueDocIds(
  hitDocIds: readonly string[],
  maxDocs: number
): string[] {
  const unique: string[] = [];
  const seen = new Set<string>();
  for (const docId of hitDocIds) {
    if (seen.has(docId)) {
      continue;
    }
    seen.add(docId);
    unique.push(docId);
    if (unique.length >= maxDocs) {
      break;
    }
  }
  return unique;
}

function partitionCitations(
  candidateIds: readonly string[],
  observedDocIds: ReadonlySet<string>,
  store: KnowledgeStore,
  abstained: boolean
): { validDocIds: string[]; strippedDocIds: string[] } {
  const validDocIds: string[] = [];
  const strippedDocIds: string[] = [];
  const seenValid = new Set<string>();
  const seenStripped = new Set<string>();

  for (const rawId of candidateIds) {
    const id = rawId.trim();
    if (id.length === 0) {
      continue;
    }
    if (
      abstained ||
      observedDocIds.size === 0 ||
      !observedDocIds.has(id) ||
      !store.hasDocumentId(id)
    ) {
      if (!seenStripped.has(id)) {
        seenStripped.add(id);
        strippedDocIds.push(id);
      }
      continue;
    }
    if (!seenValid.has(id)) {
      seenValid.add(id);
      validDocIds.push(id);
    }
  }

  return { validDocIds, strippedDocIds };
}

function runObserved<T>(
  telemetry: TelemetryProvider | undefined,
  options: TelemetryObservationOptions,
  fn: (ctx: TelemetryObservationContext) => Promise<T>
): Promise<T> {
  if (telemetry === undefined) {
    return fn({ setOutput() {} });
  }
  return telemetry.startObservation(options, fn);
}

function finalizeAnswer(params: {
  synthesis: AgentSynthesis;
  observedDocIds: ReadonlySet<string>;
  store: KnowledgeStore;
  usage: LlmTokenUsage;
  embeddingInputTokens: number;
  model: string;
  agentTurns: number;
  deadlineHit: boolean;
  onEvent?: (event: AnswerTraceEvent) => void;
}): AnswerResult {
  const trimmedAnswer = params.synthesis.answer.trim();
  const hasAnswerText = trimmedAnswer.length > 0;
  const abstained = params.synthesis.abstained || !hasAnswerText;
  const finalAnswer = hasAnswerText ? trimmedAnswer : FALLBACK_ABSTENTION_TEXT;
  const { validDocIds, strippedDocIds } = partitionCitations(
    params.synthesis.supporting_doc_ids,
    params.observedDocIds,
    params.store,
    abstained
  );

  params.onEvent?.({
    type: "citations_validated",
    validDocIds,
    strippedDocIds,
    abstained
  });

  const response = AskResponseSchema.parse({
    answer: finalAnswer,
    document_ids: validDocIds
  });

  return {
    response,
    abstained,
    retrievedDocIds: [...params.observedDocIds],
    strippedDocIds,
    usage: params.usage,
    embeddingInputTokens: params.embeddingInputTokens,
    model: params.model,
    agentTurns: params.agentTurns,
    deadlineHit: params.deadlineHit
  };
}

function buildToolDefinitions(
  enabledTools: readonly AgentTool[],
  enableEmbeddings: boolean
): LlmToolDefinition[] {
  const definitions: LlmToolDefinition[] = [];
  const seen = new Set<AgentTool>();

  for (const toolName of enabledTools) {
    if (seen.has(toolName)) {
      continue;
    }
    seen.add(toolName);

    switch (toolName) {
      case "search":
        definitions.push({
          name: "search",
          description: enableEmbeddings
            ? "Search Corpus Documents using hybrid BM25 full-text and dense semantic search fused with Reciprocal Rank Fusion. Returns ranked candidate snippets with dataset_doc_uuid values."
            : "Search Corpus Documents using full-text BM25 search across titles, metadata, and text. Returns ranked candidate snippets with dataset_doc_uuid values.",
          inputSchema: SearchToolInputSchema
        });
        break;
      case "fetch_docs":
        definitions.push({
          name: "fetch_docs",
          description:
            "Read full contents and metadata for specific dataset_doc_uuid values returned by search or filter_metadata.",
          inputSchema: FetchDocsToolInputSchema
        });
        break;
      case "filter_metadata":
        definitions.push({
          name: "filter_metadata",
          description:
            "Filter and count Corpus Documents by source, author, status, date bounds, title substring, or metadata_json key filters. Always returns exact totalCount and a page of matching document metadata rows.",
          inputSchema: MetadataFilterQuerySchema
        });
        break;
      case "expand_graph":
        definitions.push({
          name: "expand_graph",
          description:
            "Expand deterministic Link Graph neighbors from a seed docId or an entityName (optional entityType). Returns connected Corpus Document metadata and link reasons; call fetch_docs on promising neighbors to inspect full contents before citing.",
          inputSchema: ExpandGraphQuerySchema
        });
        break;
    }
  }

  return definitions;
}

function addUsage(target: LlmTokenUsage, delta: LlmTokenUsage): void {
  target.inputTokens += delta.inputTokens;
  target.outputTokens += delta.outputTokens;
}

interface ExecutedToolCall {
  readonly result: LlmToolResult;
  readonly embeddingInputTokens: number;
  readonly summary: string;
}

function formatErrorSummary(output: unknown): string {
  if (
    typeof output === "object" &&
    output !== null &&
    "error" in output &&
    typeof (output as { error: unknown }).error === "string"
  ) {
    return `error: ${(output as { error: string }).error}`;
  }
  return "error";
}

function formatFetchDocsSummary(docs: readonly StoredDocument[]): string {
  if (docs.length === 0) {
    return "0 docs fetched";
  }
  const preview = docs
    .slice(0, 4)
    .map((doc) => doc.title)
    .join(", ");
  const suffix = docs.length > 4 ? `, +${docs.length - 4} more` : "";
  return `${docs.length} ${docs.length === 1 ? "doc" : "docs"}: ${preview}${suffix}`;
}

async function executeAgentToolCall(params: {
  call: LlmToolCall;
  enabledToolsSet: ReadonlySet<string>;
  store: KnowledgeStore;
  profile: RunProfile;
  embedder?: Embedder;
  signal?: AbortSignal;
  telemetry?: TelemetryProvider;
  fetchedDocsById: Map<string, StoredDocument>;
  observedDocIds: Set<string>;
  filterSnapshots: FilterSnapshot[];
}): Promise<ExecutedToolCall> {
  const {
    call,
    enabledToolsSet,
    store,
    profile,
    embedder,
    signal,
    telemetry,
    fetchedDocsById,
    observedDocIds,
    filterSnapshots
  } = params;

  if (!enabledToolsSet.has(call.toolName)) {
    const output = { error: `Tool "${call.toolName}" is not enabled in this profile.` };
    return {
      embeddingInputTokens: 0,
      summary: formatErrorSummary(output),
      result: {
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        isError: true,
        output
      }
    };
  }

  try {
    switch (call.toolName) {
      case "search": {
        const parsed = SearchToolInputSchema.safeParse(call.input);
        if (!parsed.success) {
          const output = { error: `Invalid search input: ${parsed.error.message}` };
          return {
            embeddingInputTokens: 0,
            summary: formatErrorSummary(output),
            result: {
              toolCallId: call.toolCallId,
              toolName: call.toolName,
              isError: true,
              output
            }
          };
        }

        const topK = Math.min(
          parsed.data.topK ?? profile.topK,
          MAX_SEARCH_TOOL_TOP_K
        );
        const retrieval = await runObserved(
          telemetry,
          {
            name: "retrieve-evidence",
            asType: "retriever",
            input: { query: parsed.data.query, topK }
          },
          async (obsCtx) => {
            const res = await retrieveEvidence(store, profile, parsed.data.query, {
              topK,
              ...(embedder !== undefined ? { embedder } : {}),
              ...(signal !== undefined ? { signal } : {})
            });
            obsCtx.setOutput({
              hitCount: res.hits.length,
              topDocIds: res.hits.slice(0, 10).map((hit) => hit.datasetDocUuid)
            });
            return res;
          }
        );

        return {
          embeddingInputTokens: retrieval.embeddingInputTokens,
          summary: `${retrieval.hits.length} ${retrieval.hits.length === 1 ? "hit" : "hits"} (${retrieval.mode})`,
          result: {
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            output: {
              query: parsed.data.query,
              mode: retrieval.mode,
              totalHits: retrieval.hits.length,
              hits: retrieval.hits.map((hit) => ({
                dataset_doc_uuid: hit.datasetDocUuid,
                source: hit.source,
                title: hit.title,
                score: hit.rrfScore,
                fts_rank: hit.ftsRank,
                dense_rank: hit.denseRank,
                dense_score: hit.denseScore,
                excerpt: hit.excerpt
              }))
            }
          }
        };
      }

      case "fetch_docs": {
        const parsed = FetchDocsToolInputSchema.safeParse(call.input);
        if (!parsed.success) {
          const output = { error: `Invalid fetch_docs input: ${parsed.error.message}` };
          return {
            embeddingInputTokens: 0,
            summary: formatErrorSummary(output),
            result: {
              toolCallId: call.toolCallId,
              toolName: call.toolName,
              isError: true,
              output
            }
          };
        }

        return await runObserved(
          telemetry,
          {
            name: "fetch-docs",
            asType: "tool",
            input: { doc_ids: parsed.data.doc_ids }
          },
          async (obsCtx) => {
            const uniqueRequested: string[] = [];
            const seenRequested = new Set<string>();
            for (const docId of parsed.data.doc_ids) {
              if (!seenRequested.has(docId)) {
                seenRequested.add(docId);
                uniqueRequested.push(docId);
              }
            }

            const alreadyFetched: StoredDocument[] = [];
            const newIds: string[] = [];
            for (const docId of uniqueRequested) {
              const existing = fetchedDocsById.get(docId);
              if (existing !== undefined) {
                alreadyFetched.push(existing);
              } else {
                newIds.push(docId);
              }
            }

            const remainingBefore = Math.max(
              0,
              profile.maxFetchedDocs - fetchedDocsById.size
            );
            const allowedNewIds = newIds.slice(0, remainingBefore);
            const truncatedByBudget = newIds.length > allowedNewIds.length;

            const newlyFetched =
              allowedNewIds.length > 0 ? store.fetchDocs(allowedNewIds) : [];
            const fetchedIdSet = new Set<string>();
            for (const doc of newlyFetched) {
              fetchedDocsById.set(doc.datasetDocUuid, doc);
              observedDocIds.add(doc.datasetDocUuid);
              fetchedIdSet.add(doc.datasetDocUuid);
            }

            const missingIds = allowedNewIds.filter((id) => !fetchedIdSet.has(id));
            const returnedDocs = [...alreadyFetched, ...newlyFetched];
            const remainingDocBudget = Math.max(
              0,
              profile.maxFetchedDocs - fetchedDocsById.size
            );

            obsCtx.setOutput({
              fetchedCount: returnedDocs.length,
              totalFetchedSoFar: fetchedDocsById.size,
              remainingDocBudget,
              truncatedByBudget,
              missingIds,
              documents: returnedDocs.map((doc) => ({
                dataset_doc_uuid: doc.datasetDocUuid,
                source: doc.source,
                title: doc.title
              }))
            });

            return {
              embeddingInputTokens: 0,
              summary: formatFetchDocsSummary(returnedDocs),
              result: {
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                output: {
                  fetchedCount: returnedDocs.length,
                  totalFetchedSoFar: fetchedDocsById.size,
                  remainingDocBudget,
                  truncatedByBudget,
                  missingIds,
                  documents: returnedDocs.map((doc) => ({
                    dataset_doc_uuid: doc.datasetDocUuid,
                    source: doc.source,
                    title: doc.title,
                    author: doc.author,
                    status: doc.status,
                    created_at: doc.createdAt,
                    updated_at: doc.updatedAt,
                    full_text:
                      doc.fullText.length > MAX_TOOL_DOC_CHARS
                        ? `${doc.fullText.slice(0, MAX_TOOL_DOC_CHARS)}\n[truncated]`
                        : doc.fullText
                  }))
                }
              }
            };
          }
        );
      }

      case "filter_metadata": {
        const parsed = MetadataFilterQuerySchema.safeParse(call.input);
        if (!parsed.success) {
          const output = {
            error: `Invalid filter_metadata input: ${parsed.error.message}`
          };
          return {
            embeddingInputTokens: 0,
            summary: formatErrorSummary(output),
            result: {
              toolCallId: call.toolCallId,
              toolName: call.toolName,
              isError: true,
              output
            }
          };
        }

        return await runObserved(
          telemetry,
          {
            name: "filter-metadata",
            asType: "tool",
            input: parsed.data
          },
          async (obsCtx) => {
            const result = store.filterMetadata(parsed.data);
            for (const doc of result.documents) {
              observedDocIds.add(doc.datasetDocUuid);
            }
            filterSnapshots.push({
              query: parsed.data as Record<string, unknown>,
              result
            });

            obsCtx.setOutput({
              totalCount: result.totalCount,
              returnedCount: result.documents.length,
              docIds: result.documents.map((doc) => doc.datasetDocUuid)
            });

            return {
              embeddingInputTokens: 0,
              summary: `totalCount=${result.totalCount} (returned ${result.documents.length})`,
              result: {
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                output: {
                  totalCount: result.totalCount,
                  returnedCount: result.documents.length,
                  documents: result.documents.map((doc) => ({
                    dataset_doc_uuid: doc.datasetDocUuid,
                    source: doc.source,
                    title: doc.title,
                    author: doc.author,
                    status: doc.status,
                    created_at: doc.createdAt,
                    updated_at: doc.updatedAt
                  }))
                }
              }
            };
          }
        );
      }

      case "expand_graph": {
        const parsed = ExpandGraphQuerySchema.safeParse(call.input);
        if (!parsed.success) {
          const output = {
            error: `Invalid expand_graph input: ${parsed.error.message}`
          };
          return {
            embeddingInputTokens: 0,
            summary: formatErrorSummary(output),
            result: {
              toolCallId: call.toolCallId,
              toolName: call.toolName,
              isError: true,
              output
            }
          };
        }

        return await runObserved(
          telemetry,
          {
            name: "expand-graph",
            asType: "tool",
            input: parsed.data
          },
          async (obsCtx) => {
            const neighbors = store.expandGraph(parsed.data);
            obsCtx.setOutput({
              totalNeighbors: neighbors.length,
              neighborDocIds: neighbors.map((neighbor) => neighbor.datasetDocUuid)
            });

            return {
              embeddingInputTokens: 0,
              summary: `${neighbors.length} ${neighbors.length === 1 ? "neighbor" : "neighbors"}`,
              result: {
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                output: {
                  query: parsed.data,
                  totalNeighbors: neighbors.length,
                  neighbors: neighbors.map((neighbor) => ({
                    dataset_doc_uuid: neighbor.datasetDocUuid,
                    source: neighbor.source,
                    title: neighbor.title,
                    author: neighbor.author,
                    status: neighbor.status,
                    updated_at: neighbor.updatedAt,
                    hop: neighbor.hop,
                    score: Number(neighbor.score.toFixed(4)),
                    reasons: neighbor.reasons
                  }))
                }
              }
            };
          }
        );
      }

      default: {
        const output = { error: `Unknown tool "${call.toolName}".` };
        return {
          embeddingInputTokens: 0,
          summary: formatErrorSummary(output),
          result: {
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            isError: true,
            output
          }
        };
      }
    }
  } catch (error) {
    if (error instanceof ValidationError) {
      const output = { error: error.message };
      return {
        embeddingInputTokens: 0,
        summary: formatErrorSummary(output),
        result: {
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          isError: true,
          output
        }
      };
    }
    throw error;
  }
}

async function answerWithSinglePassRetrieval(
  store: KnowledgeStore,
  llm: LlmProvider,
  profile: RunProfile,
  trimmedQuestion: string,
  options: AnswerQuestionOptions
): Promise<AnswerResult> {
  const { embedder, signal, onEvent, telemetry } = options;

  const retrieveStart = performance.now();
  const retrieval = await runObserved(
    telemetry,
    {
      name: "retrieve-evidence",
      asType: "retriever",
      input: { query: trimmedQuestion, topK: profile.topK }
    },
    async (obsCtx) => {
      const res = await retrieveEvidence(store, profile, trimmedQuestion, {
        ...(embedder !== undefined ? { embedder } : {}),
        ...(signal !== undefined ? { signal } : {})
      });
      obsCtx.setOutput({
        hitCount: res.hits.length,
        topDocIds: res.hits.slice(0, 10).map((hit) => hit.datasetDocUuid)
      });
      return res;
    }
  );
  onEvent?.({
    type: "pre_search_done",
    query: trimmedQuestion,
    mode: retrieval.mode,
    hitCount: retrieval.hits.length,
    topDocIds: retrieval.hits.slice(0, 10).map((hit) => hit.datasetDocUuid),
    durationMs: Math.max(0, performance.now() - retrieveStart)
  });

  const candidateDocIds = collectTopUniqueDocIds(
    retrieval.hits.map((hit) => hit.datasetDocUuid),
    profile.maxFetchedDocs
  );
  const docs = store.fetchDocs(candidateDocIds);
  const readDocIds = new Set(docs.map((doc) => doc.datasetDocUuid));

  onEvent?.({
    type: "synthesis_start",
    model: profile.synthesisModel,
    fetchedDocCount: docs.length,
    filterSnapshotCount: 0
  });

  const synthesis = await llm.synthesize(
    {
      model: profile.synthesisModel,
      system: SYNTHESIS_SYSTEM_PROMPT,
      prompt: buildL0SynthesisPrompt(trimmedQuestion, docs)
    },
    signal
  );

  if (synthesis.reasoningText !== undefined && synthesis.reasoningText.length > 0) {
    onEvent?.({
      type: "synthesis_thinking",
      text: synthesis.reasoningText
    });
  }

  return finalizeAnswer({
    synthesis: synthesis.result,
    observedDocIds: readDocIds,
    store,
    usage: synthesis.usage,
    embeddingInputTokens: retrieval.embeddingInputTokens,
    model: synthesis.model,
    agentTurns: 0,
    deadlineHit: false,
    ...(onEvent !== undefined ? { onEvent } : {})
  });
}

async function answerWithBoundedAgent(
  store: KnowledgeStore,
  llm: LlmProvider,
  profile: RunProfile,
  trimmedQuestion: string,
  options: AnswerQuestionOptions
): Promise<AnswerResult> {
  const { embedder, signal, onEvent, telemetry } = options;

  if (llm.runAgentStep === undefined) {
    throw new ProviderError(
      "Configured LlmProvider does not support agent tool steps"
    );
  }

  const tools = buildToolDefinitions(
    profile.enabledTools,
    profile.enableEmbeddings && embedder !== undefined
  );
  const enabledToolsSet = new Set<string>(tools.map((item) => item.name));
  const schemaCatalogText = formatSchemaCatalog(store.getSchemaCatalog());
  const agentSystemPrompt = buildAgentSystemPrompt(schemaCatalogText, profile);

  const fetchedDocsById = new Map<string, StoredDocument>();
  const observedDocIds = new Set<string>();
  const filterSnapshots: FilterSnapshot[] = [];
  const totalUsage: LlmTokenUsage = { inputTokens: 0, outputTokens: 0 };

  let totalEmbeddingInputTokens = 0;
  let turns = 0;
  let deadlineHit = false;
  let deadlineEventEmitted = false;
  let lastAgentText = "";

  const markDeadlineHit = (): void => {
    deadlineHit = true;
    if (!deadlineEventEmitted) {
      deadlineEventEmitted = true;
      onEvent?.({
        type: "deadline_hit",
        turnsCompleted: turns,
        softDeadlineMs: profile.softDeadlineMs
      });
    }
  };

  const deadlineController = new AbortController();
  const deadlineTimer = setTimeout(() => {
    deadlineHit = true;
    deadlineController.abort(new Error("Soft deadline exceeded"));
  }, profile.softDeadlineMs);
  deadlineTimer.unref();

  const agentSignal =
    signal !== undefined
      ? AbortSignal.any([signal, deadlineController.signal])
      : deadlineController.signal;

  try {
    let initialUserContent = `Question:\n${trimmedQuestion}`;

    if (profile.preSearchEnabled && !deadlineHit && !agentSignal.aborted) {
      try {
        const preSearchStart = performance.now();
        const preSearch = await runObserved(
          telemetry,
          {
            name: "retrieve-evidence",
            asType: "retriever",
            input: { query: trimmedQuestion, topK: profile.topK }
          },
          async (obsCtx) => {
            const res = await retrieveEvidence(
              store,
              profile,
              trimmedQuestion,
              {
                ...(embedder !== undefined ? { embedder } : {}),
                signal: agentSignal
              }
            );
            obsCtx.setOutput({
              hitCount: res.hits.length,
              topDocIds: res.hits.slice(0, 10).map((hit) => hit.datasetDocUuid)
            });
            return res;
          }
        );
        totalEmbeddingInputTokens += preSearch.embeddingInputTokens;
        onEvent?.({
          type: "pre_search_done",
          query: trimmedQuestion,
          mode: preSearch.mode,
          hitCount: preSearch.hits.length,
          topDocIds: preSearch.hits.slice(0, 10).map((hit) => hit.datasetDocUuid),
          durationMs: Math.max(0, performance.now() - preSearchStart)
        });
        initialUserContent = [
          `Question:\n${trimmedQuestion}`,
          `Pre-search candidate snippets (${preSearch.hits.length} hits):`,
          formatPreSearchHits(preSearch.hits),
          "Use fetch_docs to read promising documents in full, or refine with search / filter_metadata."
        ].join("\n\n");
      } catch (error) {
        if (signal?.aborted) {
          throw error;
        }
        if (deadlineHit || deadlineController.signal.aborted) {
          markDeadlineHit();
        } else {
          throw error;
        }
      }
    }

    const messages: LlmAgentMessage[] = [
      { role: "user", content: initialUserContent }
    ];

    while (turns < profile.maxAgentTurns) {
      if (signal?.aborted) {
        throw new ProviderError("Agent loop aborted", { cause: signal.reason });
      }
      if (deadlineHit || deadlineController.signal.aborted) {
        markDeadlineHit();
        break;
      }

      const currentTurn = turns + 1;
      onEvent?.({
        type: "agent_turn_start",
        turn: currentTurn,
        maxTurns: profile.maxAgentTurns,
        model: profile.routingModel
      });

      let stepOutput;
      try {
        stepOutput = await llm.runAgentStep(
          {
            model: profile.routingModel,
            system: agentSystemPrompt,
            messages,
            tools
          },
          agentSignal
        );
      } catch (error) {
        if (signal?.aborted) {
          throw error;
        }
        if (deadlineHit || deadlineController.signal.aborted) {
          markDeadlineHit();
          break;
        }
        throw error;
      }

      addUsage(totalUsage, stepOutput.usage);
      if (stepOutput.reasoningText !== undefined && stepOutput.reasoningText.length > 0) {
        onEvent?.({
          type: "agent_thinking",
          turn: currentTurn,
          text: stepOutput.reasoningText
        });
      }
      if (stepOutput.text.trim().length > 0) {
        lastAgentText = stepOutput.text.trim();
      }

      if (deadlineHit || deadlineController.signal.aborted) {
        markDeadlineHit();
        break;
      }

      if (stepOutput.toolCalls.length === 0) {
        break;
      }

      turns += 1;
      messages.push({
        role: "assistant",
        text: stepOutput.text,
        toolCalls: stepOutput.toolCalls,
        ...(stepOutput.rawProviderMessage !== undefined
          ? { rawProviderMessage: stepOutput.rawProviderMessage }
          : {})
      });

      const toolResults: LlmToolResult[] = [];
      for (const call of stepOutput.toolCalls) {
        if (deadlineHit || deadlineController.signal.aborted) {
          markDeadlineHit();
          break;
        }
        onEvent?.({
          type: "tool_call",
          turn: turns,
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: call.input
        });
        const toolStart = performance.now();
        try {
          const executed = await executeAgentToolCall({
            call,
            enabledToolsSet,
            store,
            profile,
            ...(embedder !== undefined ? { embedder } : {}),
            signal: agentSignal,
            ...(telemetry !== undefined ? { telemetry } : {}),
            fetchedDocsById,
            observedDocIds,
            filterSnapshots
          });
          totalEmbeddingInputTokens += executed.embeddingInputTokens;
          toolResults.push(executed.result);
          onEvent?.({
            type: "tool_result",
            turn: turns,
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            isError: executed.result.isError === true,
            durationMs: Math.max(0, performance.now() - toolStart),
            summary: executed.summary,
            output: executed.result.output
          });
        } catch (error) {
          if (signal?.aborted) {
            throw error;
          }
          if (deadlineHit || deadlineController.signal.aborted) {
            markDeadlineHit();
            break;
          }
          throw error;
        }
      }

      if (deadlineHit || deadlineController.signal.aborted) {
        markDeadlineHit();
        break;
      }

      messages.push({
        role: "tool",
        results: toolResults
      });
    }
  } finally {
    clearTimeout(deadlineTimer);
  }

  const synthesisPrompt = buildAgentSynthesisPrompt({
    question: trimmedQuestion,
    fetchedDocs: [...fetchedDocsById.values()],
    filterSnapshots,
    agentNotes: lastAgentText,
    deadlineHit,
    hasObservedEvidence: observedDocIds.size > 0
  });

  onEvent?.({
    type: "synthesis_start",
    model: profile.synthesisModel,
    fetchedDocCount: fetchedDocsById.size,
    filterSnapshotCount: filterSnapshots.length
  });

  const synthesis = await llm.synthesize(
    {
      model: profile.synthesisModel,
      system: SYNTHESIS_SYSTEM_PROMPT,
      prompt: synthesisPrompt
    },
    signal
  );

  addUsage(totalUsage, synthesis.usage);
  if (synthesis.reasoningText !== undefined && synthesis.reasoningText.length > 0) {
    onEvent?.({
      type: "synthesis_thinking",
      text: synthesis.reasoningText
    });
  }

  return finalizeAnswer({
    synthesis: synthesis.result,
    observedDocIds,
    store,
    usage: totalUsage,
    embeddingInputTokens: totalEmbeddingInputTokens,
    model: synthesis.model,
    agentTurns: turns,
    deadlineHit,
    ...(onEvent !== undefined ? { onEvent } : {})
  });
}

function isAbortSignal(value: unknown): value is AbortSignal {
  return (
    typeof value === "object" &&
    value !== null &&
    "aborted" in value &&
    typeof (value as { aborted: unknown }).aborted === "boolean"
  );
}

function normalizeAnswerOptions(
  signalOrOptions?: AbortSignal | AnswerQuestionOptions
): AnswerQuestionOptions {
  if (signalOrOptions === undefined) {
    return {};
  }
  if (isAbortSignal(signalOrOptions)) {
    return { signal: signalOrOptions };
  }
  return signalOrOptions;
}

export async function answerQuestion(
  store: KnowledgeStore,
  llm: LlmProvider,
  profile: RunProfile,
  question: string,
  signalOrOptions?: AbortSignal | AnswerQuestionOptions
): Promise<AnswerResult> {
  const trimmedQuestion = question.trim();
  if (trimmedQuestion.length === 0) {
    throw new ValidationError("Question must not be empty");
  }

  const options = normalizeAnswerOptions(signalOrOptions);
  const startedAt = performance.now();
  const indexConfigHash = computeIndexConfigHash(profile);

  return runObserved(
    options.telemetry,
    {
      name: "answer-question",
      asType: "agent",
      input: { question: trimmedQuestion },
      metadata: {
        profile: profile.name,
        indexConfigHash,
        enabledTools: [...profile.enabledTools]
      }
    },
    async (obsCtx) => {
      const rawResult =
        profile.enabledTools.length === 0
          ? await answerWithSinglePassRetrieval(
              store,
              llm,
              profile,
              trimmedQuestion,
              options
            )
          : await answerWithBoundedAgent(
              store,
              llm,
              profile,
              trimmedQuestion,
              options
            );

      const traceId = obsCtx.traceId ?? options.telemetry?.getActiveTraceId();
      const result: AnswerResult =
        traceId !== undefined ? { ...rawResult, traceId } : rawResult;

      obsCtx.setOutput({
        answer: result.response.answer,
        abstained: result.abstained,
        document_ids: result.response.document_ids,
        agentTurns: result.agentTurns,
        deadlineHit: result.deadlineHit
      });

      options.onEvent?.({
        type: "answer_complete",
        abstained: result.abstained,
        documentIds: result.response.document_ids,
        strippedDocIds: result.strippedDocIds,
        agentTurns: result.agentTurns,
        deadlineHit: result.deadlineHit,
        usage: result.usage,
        embeddingInputTokens: result.embeddingInputTokens,
        durationMs: Math.max(0, performance.now() - startedAt),
        ...(traceId !== undefined ? { traceId } : {})
      });

      return result;
    }
  );
}
