import { join, resolve } from "node:path";
import type { CorpusReader } from "../adapters/corpus-reader.js";
import type { Embedder } from "../adapters/embedder.js";
import type { KnowledgeStore } from "../adapters/knowledge-store.js";
import {
  IngestionError,
  type ChunkEmbeddingRecord,
  type ChunkForEmbedding,
  type ChunkRecord,
  type NormalizedDocument
} from "../models/corpus.js";
import { ConfigError } from "../models/errors.js";
import {
  computeIndexConfigHash,
  type IndexConfig,
  type RunProfile
} from "../models/profile.js";
import { extractGraphArtifacts } from "../utils/graph.js";
import { sha256Hex } from "../utils/hash.js";
import { chunkDocumentText, renderCanonicalDocument } from "../utils/text.js";

export interface EmbeddingUsageEvent {
  readonly model: string;
  readonly dims: number;
  readonly chunkCount: number;
  readonly inputTokens: number;
}

export interface IngestCorpusOptions {
  readonly signal?: AbortSignal;
  readonly embedder?: Embedder;
  readonly embeddingBatchSize?: number;
  readonly embeddingConcurrency?: number;
  readonly maxEmbeddingChars?: number;
  readonly onEmbeddingUsage?: (event: EmbeddingUsageEvent) => void;
}

export interface IngestSummary {
  total: number;
  inserted: number;
  updated: number;
  unchanged: number;
  deleted: number;
  sourceCount: number;
  embeddedChunks: number;
  embeddingInputTokens: number;
}

const DEFAULT_EMBEDDING_BATCH_SIZE = 100;
const DEFAULT_EMBEDDING_CONCURRENCY = 8;
const DEFAULT_MAX_EMBEDDING_CHARS = 1200;

export function resolveCachedDbPath(
  cacheDir: string,
  corpusDir: string,
  profile: IndexConfig
): string {
  const normalizedCacheDir = resolve(cacheDir);
  const normalizedCorpusDir = resolve(corpusDir);
  const corpusHash = sha256Hex(normalizedCorpusDir).slice(0, 12);
  const indexHash = computeIndexConfigHash(profile).slice(0, 12);
  return join(normalizedCacheDir, `brain-${corpusHash}-${indexHash}.sqlite`);
}

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new IngestionError("Corpus ingestion aborted", {
      cause: signal.reason
    });
  }
}

function isAbortSignal(value: unknown): value is AbortSignal {
  return (
    typeof value === "object" &&
    value !== null &&
    "aborted" in value &&
    typeof (value as { aborted: unknown }).aborted === "boolean"
  );
}

function normalizeIngestOptions(
  signalOrOptions?: AbortSignal | IngestCorpusOptions
): IngestCorpusOptions {
  if (signalOrOptions === undefined) {
    return {};
  }
  if (isAbortSignal(signalOrOptions)) {
    return { signal: signalOrOptions };
  }
  return signalOrOptions;
}

export function capEmbeddingText(text: string, maxChars: number): string {
  if (maxChars <= 0 || text.length <= maxChars) {
    return text;
  }
  const separator = "\n...\n";
  if (maxChars <= separator.length + 20) {
    return text.slice(0, maxChars);
  }
  const available = maxChars - separator.length;
  const headChars = Math.ceil(available * 0.625);
  const tailChars = available - headChars;
  return `${text.slice(0, headChars).trimEnd()}${separator}${text.slice(text.length - tailChars).trimStart()}`;
}

function formatChunkForEmbedding(
  chunk: ChunkForEmbedding,
  includeHeaderCard: boolean,
  maxEmbeddingChars: number
): string {
  const body = chunk.body.trim();
  const raw = includeHeaderCard
    ? body.length > 0
      ? body
      : chunk.title
    : body.length === 0
      ? chunk.title
      : `${chunk.title}\n\n${body}`;
  return capEmbeddingText(raw, maxEmbeddingChars);
}

export async function ingestCorpus(
  reader: CorpusReader,
  store: KnowledgeStore,
  profile: RunProfile,
  signalOrOptions?: AbortSignal | IngestCorpusOptions
): Promise<IngestSummary> {
  const options = normalizeIngestOptions(signalOrOptions);
  const signal = options.signal;
  assertNotAborted(signal);

  if (profile.enableEmbeddings) {
    if (options.embedder === undefined) {
      throw new ConfigError(
        "Embedder is required when profile.enableEmbeddings is true"
      );
    }
    if (
      options.embedder.model !== profile.embeddingModel ||
      options.embedder.dims !== profile.embeddingDims
    ) {
      throw new ConfigError(
        `Embedder (${options.embedder.model}/${options.embedder.dims}) does not match profile (${profile.embeddingModel}/${profile.embeddingDims})`
      );
    }
  }

  const embeddingBatchSize =
    options.embeddingBatchSize ?? DEFAULT_EMBEDDING_BATCH_SIZE;
  if (!Number.isInteger(embeddingBatchSize) || embeddingBatchSize <= 0) {
    throw new ConfigError(
      `embeddingBatchSize must be a positive integer, got ${embeddingBatchSize}`
    );
  }
  const embeddingConcurrency =
    options.embeddingConcurrency ?? DEFAULT_EMBEDDING_CONCURRENCY;
  if (!Number.isInteger(embeddingConcurrency) || embeddingConcurrency <= 0) {
    throw new ConfigError(
      `embeddingConcurrency must be a positive integer, got ${embeddingConcurrency}`
    );
  }
  const maxEmbeddingChars =
    options.maxEmbeddingChars ?? DEFAULT_MAX_EMBEDDING_CHARS;
  if (!Number.isInteger(maxEmbeddingChars) || maxEmbeddingChars <= 0) {
    throw new ConfigError(
      `maxEmbeddingChars must be a positive integer, got ${maxEmbeddingChars}`
    );
  }

  const existingIds = new Set(store.listDocumentIds());
  const seenIds = new Set<string>();

  let inserted = 0;
  let updated = 0;
  let unchanged = 0;
  let deleted = 0;

  for await (const record of reader.readCorpus(signal)) {
    assertNotAborted(signal);

    const uuid = record.raw.dataset_doc_uuid;
    if (seenIds.has(uuid)) {
      throw new IngestionError(`Duplicate dataset_doc_uuid in corpus: ${uuid}`);
    }
    seenIds.add(uuid);

    const existingHash = store.getDocumentHash(uuid);
    if (existingHash === record.contentHash) {
      if (!store.hasDocumentGraph(uuid, record.contentHash)) {
        const rendered = renderCanonicalDocument(record.raw, record.source);
        const graphArtifacts = extractGraphArtifacts({
          datasetDocUuid: uuid,
          contentHash: record.contentHash,
          source: record.source,
          relativePath: record.relativePath,
          raw: record.raw,
          title: rendered.title,
          bodyText: rendered.bodyText
        });
        store.upsertDocumentGraph(uuid, record.contentHash, graphArtifacts);
      }
      unchanged += 1;
      continue;
    }

    const rendered = renderCanonicalDocument(record.raw, record.source);
    const chunkBodies = chunkDocumentText(
      {
        headerCard: rendered.headerCard,
        bodyText: rendered.bodyText
      },
      profile.chunkStrategy
    );

    const normalizedDoc: NormalizedDocument = {
      datasetDocUuid: uuid,
      contentHash: record.contentHash,
      source: record.source,
      title: rendered.title,
      author: rendered.author,
      status: rendered.status,
      createdAt: rendered.createdAt,
      updatedAt: rendered.updatedAt,
      metadataJson: JSON.stringify(rendered.metadataRecord),
      metadataText: rendered.metadataText,
      fullText: rendered.fullText
    };

    const chunks: ChunkRecord[] = chunkBodies.map((body, chunkIndex) => ({
      datasetDocUuid: uuid,
      contentHash: record.contentHash,
      chunkIndex,
      title: rendered.title,
      metadataText: rendered.metadataText,
      body
    }));

    const graphArtifacts = extractGraphArtifacts({
      datasetDocUuid: uuid,
      contentHash: record.contentHash,
      source: record.source,
      relativePath: record.relativePath,
      raw: record.raw,
      title: rendered.title,
      bodyText: rendered.bodyText
    });

    store.upsertDocument(normalizedDoc, chunks, graphArtifacts);
    if (existingHash === null) {
      inserted += 1;
    } else {
      updated += 1;
    }
  }

  for (const existingId of existingIds) {
    assertNotAborted(signal);
    if (!seenIds.has(existingId)) {
      if (store.deleteDocument(existingId)) {
        deleted += 1;
      }
    }
  }

  assertNotAborted(signal);
  store.resolveGraphEdges();
  const catalog = store.rebuildSchemaCatalog();

  let embeddedChunks = 0;
  let embeddingInputTokens = 0;

  if (profile.enableEmbeddings && options.embedder !== undefined) {
    const embedder = options.embedder;
    const missingChunks = store.getMissingEmbeddingChunks(
      embedder.model,
      embedder.dims
    );

    const batches: ChunkForEmbedding[][] = [];
    for (
      let offset = 0;
      offset < missingChunks.length;
      offset += embeddingBatchSize
    ) {
      batches.push(missingChunks.slice(offset, offset + embeddingBatchSize));
    }

    let nextBatchIndex = 0;
    let firstError: unknown = null;

    const runEmbeddingWorker = async (): Promise<void> => {
      for (;;) {
        if (firstError !== null) {
          return;
        }
        assertNotAborted(signal);
        const batchIndex = nextBatchIndex;
        if (batchIndex >= batches.length) {
          return;
        }
        nextBatchIndex += 1;
        const batch = batches[batchIndex];
        if (batch === undefined) {
          return;
        }

        try {
          const texts = batch.map((chunk) =>
            formatChunkForEmbedding(
              chunk,
              profile.chunkStrategy.includeHeaderCard,
              maxEmbeddingChars
            )
          );
          const batchResult = await embedder.embedBatch(texts, signal);

          if (batchResult.vectors.length !== batch.length) {
            throw new IngestionError(
              `Embedder returned ${batchResult.vectors.length} vectors for batch of ${batch.length}`
            );
          }

          const records: ChunkEmbeddingRecord[] = [];
          for (let i = 0; i < batch.length; i += 1) {
            const chunk = batch[i];
            const vector = batchResult.vectors[i];
            if (chunk === undefined || vector === undefined) {
              throw new IngestionError("Missing chunk or vector in embedding batch");
            }
            records.push({
              chunkId: chunk.chunkId,
              datasetDocUuid: chunk.datasetDocUuid,
              contentHash: chunk.contentHash,
              model: embedder.model,
              dims: embedder.dims,
              vector
            });
          }

          store.upsertChunkEmbeddings(records);
          embeddedChunks += batch.length;
          embeddingInputTokens += batchResult.inputTokens;

          if (options.onEmbeddingUsage !== undefined) {
            options.onEmbeddingUsage({
              model: embedder.model,
              dims: embedder.dims,
              chunkCount: batch.length,
              inputTokens: batchResult.inputTokens
            });
          }
        } catch (error) {
          if (firstError === null) {
            firstError = error;
          }
          return;
        }
      }
    };

    const workerCount = Math.min(embeddingConcurrency, batches.length);
    await Promise.all(Array.from({ length: workerCount }, () => runEmbeddingWorker()));
    if (firstError !== null) {
      throw firstError;
    }

    assertNotAborted(signal);
    store.loadVectorCache(embedder.model, embedder.dims);
  }

  return {
    total: seenIds.size,
    inserted,
    updated,
    unchanged,
    deleted,
    sourceCount: catalog.sources.length,
    embeddedChunks,
    embeddingInputTokens
  };
}
