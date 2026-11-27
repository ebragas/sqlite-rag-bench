import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import Database from "better-sqlite3";
import {
  ExpandGraphQuerySchema,
  IngestionError,
  METADATA_KEY_PATH_PATTERN,
  MetadataFilterQuerySchema,
  RetrievalError,
  type ChunkEmbeddingRecord,
  type ChunkForEmbedding,
  type ChunkRecord,
  type DenseSearchHit,
  type DocAliasRecord,
  type EntityMentionRecord,
  type EntityType,
  type ExpandGraphNeighbor,
  type ExpandGraphQueryInput,
  type ExtractedGraphArtifacts,
  type FtsSearchHit,
  type GraphStats,
  type MetadataFilterOrderBy,
  type MetadataFilterQueryInput,
  type MetadataFilterResult,
  type MetadataFilterRow,
  type NormalizedDocument,
  type RawGraphRef,
  type SchemaCatalog,
  type SchemaCatalogEntry,
  type StoredDocument,
  type VectorCacheStats
} from "../models/corpus.js";
import { ValidationError } from "../models/errors.js";
import {
  buildEntityKey,
  ENTITY_TYPES,
  normalizeEntityName
} from "../utils/graph.js";
import { buildFtsQueryCandidates } from "../utils/text.js";
import {
  bufferToFloat32,
  dotProduct,
  float32ToBuffer,
  l2Normalize
} from "../utils/vector.js";

interface HashRow {
  content_hash: string;
}

interface IdRow {
  dataset_doc_uuid: string;
}

interface DocumentRow {
  dataset_doc_uuid: string;
  content_hash: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  created_at: string | null;
  updated_at: string | null;
  metadata_json: string;
  full_text: string;
}

interface CatalogSourceRow {
  source: string;
  metadata_json: string;
}

interface StoredCatalogRow {
  source: string;
  doc_count: number;
  metadata_keys_json: string;
  sample_values_json: string;
}

interface FtsRow {
  chunkId: number;
  datasetDocUuid: string;
  chunkIndex: number;
  source: string;
  title: string;
  score: number;
  excerpt: string;
}

interface CountRow {
  count: number;
}

interface MetadataSummaryRow {
  dataset_doc_uuid: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  created_at: string | null;
  updated_at: string | null;
}

interface MissingEmbeddingChunkRow {
  chunkId: number;
  datasetDocUuid: string;
  contentHash: string;
  chunkIndex: number;
  title: string;
  metadataText: string;
  body: string;
}

interface StoredEmbeddingRow {
  chunkId: number;
  datasetDocUuid: string;
  chunkIndex: number;
  source: string;
  title: string;
  snippet: string;
  embedding: Buffer;
}

interface SerializedEmbeddingRow {
  chunkId: number;
  datasetDocUuid: string;
  contentHash: string;
  model: string;
  dims: number;
  embedding: Buffer;
}

interface LoadedVectorCache {
  model: string;
  dims: number;
  count: number;
  vectors: Float32Array;
  chunkIds: number[];
  docUuids: string[];
  chunkIndices: number[];
  sources: string[];
  titles: string[];
  excerpts: string[];
}

interface RankedDenseCandidate {
  index: number;
  score: number;
  docUuid: string;
  chunkId: number;
}

interface GraphNeighborEdgeRow {
  neighborUuid: string;
  weight: number;
  reason: string;
}

interface DocEntityWithFreqRow {
  entityKey: string;
  entityType: EntityType;
  entityName: string;
  docFreq: number;
}

interface EntityPeerRow {
  neighborUuid: string;
}

interface EntityMatchRow {
  datasetDocUuid: string;
  entityType: EntityType;
  entityName: string;
  structuralRole: string;
}

interface GraphDocMetaRow {
  dataset_doc_uuid: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  updated_at: string | null;
}

interface GraphCandidateAccumulator {
  bestHop: number;
  score: number;
  reasonWeights: Map<string, number>;
}

export interface KnowledgeStoreStats {
  documentCount: number;
  chunkCount: number;
  sourceCount: number;
}

const MAX_SAMPLE_VALUE_CHARS = 120;
const FETCH_BATCH_SIZE = 400;
const DENSE_EXCERPT_MAX_CHARS = 240;
const MAX_SHARED_ENTITY_PEERS = 200;
const MAX_ENTITY_LOOKUP_ROWS = 200;
const MAX_REASONS_PER_NEIGHBOR = 5;
const HOP2_FRONTIER_MIN = 25;
const HOP2_DECAY = 0.5;

const ENTITY_TYPE_WEIGHTS: Readonly<Record<EntityType, number>> = {
  deal: 1.2,
  account: 1.1,
  competitor: 1.0,
  product_or_model: 0.85,
  person: 0.75,
  label: 0.55
};

const PRIMARY_STRUCTURAL_ROLES = new Set([
  "company",
  "company_name",
  "account",
  "account_name",
  "customer",
  "customer_company",
  "customer_name",
  "deal",
  "deal_name",
  "opportunity",
  "author",
  "owner",
  "assignee",
  "reporter",
  "creator",
  "redwood_owner",
  "mailbox_owner",
  "product",
  "model",
  "repo",
  "repository",
  "competitor"
]);

function getStructuralRoleBonus(structuralRole: string): number {
  return PRIMARY_STRUCTURAL_ROLES.has(structuralRole) ? 0.25 : 0.1;
}

function compareExpandNeighbors(
  a: ExpandGraphNeighbor,
  b: ExpandGraphNeighbor
): number {
  if (Math.abs(b.score - a.score) > 1e-12) {
    return b.score - a.score;
  }
  if (a.hop !== b.hop) {
    return a.hop - b.hop;
  }
  if (a.updatedAt !== b.updatedAt) {
    if (a.updatedAt === null) {
      return 1;
    }
    if (b.updatedAt === null) {
      return -1;
    }
    return b.updatedAt.localeCompare(a.updatedAt);
  }
  return a.datasetDocUuid.localeCompare(b.datasetDocUuid);
}

function formatDenseExcerpt(rawSnippet: string): string {
  const collapsed = rawSnippet.replace(/\s+/g, " ").trim();
  if (collapsed.length <= DENSE_EXCERPT_MAX_CHARS) {
    return collapsed;
  }
  return `${collapsed.slice(0, DENSE_EXCERPT_MAX_CHARS - 3)}...`;
}

function compareDenseCandidates(
  a: RankedDenseCandidate,
  b: RankedDenseCandidate
): number {
  if (b.score !== a.score) {
    return b.score - a.score;
  }
  if (a.docUuid !== b.docUuid) {
    return a.docUuid < b.docUuid ? -1 : 1;
  }
  return a.chunkId - b.chunkId;
}

const ORDER_BY_SQL: Readonly<Record<MetadataFilterOrderBy, string>> = {
  updated_at_desc: "updated_at IS NULL ASC, updated_at DESC, dataset_doc_uuid ASC",
  updated_at_asc: "updated_at IS NULL ASC, updated_at ASC, dataset_doc_uuid ASC",
  created_at_desc: "created_at IS NULL ASC, created_at DESC, dataset_doc_uuid ASC",
  created_at_asc: "created_at IS NULL ASC, created_at ASC, dataset_doc_uuid ASC",
  title_asc: "LOWER(title) ASC, dataset_doc_uuid ASC"
};

function escapeLikePattern(value: string): string {
  return value.replace(/[\\%_]/g, "\\$&");
}

function buildSafeJsonExtractPath(keyPath: string): string {
  if (
    !METADATA_KEY_PATH_PATTERN.test(keyPath) ||
    keyPath.startsWith(".") ||
    keyPath.endsWith(".") ||
    keyPath.includes("..")
  ) {
    throw new ValidationError(`Invalid metadata key path: "${keyPath}"`);
  }
  const segments = keyPath.split(".");
  return `$${segments.map((segment) => `."${segment}"`).join("")}`;
}

function toMetadataFilterRow(row: MetadataSummaryRow): MetadataFilterRow {
  return {
    datasetDocUuid: row.dataset_doc_uuid,
    source: row.source,
    title: row.title,
    author: row.author,
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function toStoredDocument(row: DocumentRow): StoredDocument {
  return {
    datasetDocUuid: row.dataset_doc_uuid,
    contentHash: row.content_hash,
    source: row.source,
    title: row.title,
    author: row.author,
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    metadataJson: row.metadata_json,
    fullText: row.full_text
  };
}

function formatSampleScalar(value: unknown): string | null {
  let raw: string | null = null;
  if (typeof value === "string") {
    raw = value.trim();
  } else if (typeof value === "number" && Number.isFinite(value)) {
    raw = String(value);
  } else if (typeof value === "boolean") {
    raw = String(value);
  } else if (Array.isArray(value)) {
    const parts = value
      .map((item) => formatSampleScalar(item))
      .filter((item): item is string => item !== null);
    if (parts.length > 0) {
      raw = parts.join(", ");
    }
  }

  if (raw === null || raw.length === 0) {
    return null;
  }
  return raw.length > MAX_SAMPLE_VALUE_CHARS
    ? `${raw.slice(0, MAX_SAMPLE_VALUE_CHARS - 3)}...`
    : raw;
}

export class KnowledgeStore {
  private readonly db: Database.Database;
  private closed = false;
  private vectorCache: LoadedVectorCache | null = null;

  private readonly stmtGetHash: Database.Statement<[string], HashRow>;
  private readonly stmtListIds: Database.Statement<[], IdRow>;
  private readonly stmtHasId: Database.Statement<[string], IdRow>;
  private readonly stmtHasDocumentGraph: Database.Statement<
    [string, string],
    { present: number }
  >;
  private readonly stmtDeleteChunksByDoc: Database.Statement<[string]>;
  private readonly stmtDeleteAliasesByDoc: Database.Statement<[string]>;
  private readonly stmtDeleteEdgesBySource: Database.Statement<[string]>;
  private readonly stmtDeleteEntityMentionsByDoc: Database.Statement<[string]>;
  private readonly stmtUnresolveIncomingEdges: Database.Statement<[string]>;
  private readonly stmtInsertDocAlias: Database.Statement<DocAliasRecord>;
  private readonly stmtResolveIncomingByAlias: Database.Statement<
    [string, string, string]
  >;
  private readonly stmtInsertGraphEdge: Database.Statement<RawGraphRef>;
  private readonly stmtInsertEntityMention: Database.Statement<EntityMentionRecord>;
  private readonly stmtResolveGraphEdges: Database.Statement<[]>;
  private readonly stmtGetOutgoingEdges: Database.Statement<
    [string],
    GraphNeighborEdgeRow
  >;
  private readonly stmtGetIncomingEdges: Database.Statement<
    [string],
    GraphNeighborEdgeRow
  >;
  private readonly stmtGetDocEntitiesWithFreq: Database.Statement<
    [string],
    DocEntityWithFreqRow
  >;
  private readonly stmtGetEntityPeers: Database.Statement<
    [string, string, number],
    EntityPeerRow
  >;
  private readonly stmtCountAliases: Database.Statement<[], CountRow>;
  private readonly stmtCountEdges: Database.Statement<[], CountRow>;
  private readonly stmtCountResolvedEdges: Database.Statement<[], CountRow>;
  private readonly stmtCountEntityMentions: Database.Statement<[], CountRow>;
  private readonly stmtDeleteDoc: Database.Statement<[string]>;
  private readonly stmtUpsertDoc: Database.Statement<{
    datasetDocUuid: string;
    contentHash: string;
    source: string;
    title: string;
    author: string | null;
    status: string | null;
    createdAt: string | null;
    updatedAt: string | null;
    metadataJson: string;
    fullText: string;
  }>;
  private readonly stmtInsertChunk: Database.Statement<{
    datasetDocUuid: string;
    contentHash: string;
    chunkIndex: number;
    title: string;
    metadataText: string;
    body: string;
  }>;
  private readonly stmtUpsertChunkEmbedding: Database.Statement<SerializedEmbeddingRow>;
  private readonly stmtGetMissingEmbeddingChunks: Database.Statement<
    [string, number],
    MissingEmbeddingChunkRow
  >;
  private readonly stmtCountEmbeddingsForModel: Database.Statement<
    [string, number],
    CountRow
  >;
  private readonly stmtIterateEmbeddingsForModel: Database.Statement<
    [string, number],
    StoredEmbeddingRow
  >;
  private readonly stmtSearchFts: Database.Statement<[string, number], FtsRow>;
  private readonly stmtCountDocs: Database.Statement<[], CountRow>;
  private readonly stmtCountChunks: Database.Statement<[], CountRow>;
  private readonly stmtCountSources: Database.Statement<[], CountRow>;
  private readonly stmtReadAllDocMetadata: Database.Statement<[], CatalogSourceRow>;
  private readonly stmtClearCatalog: Database.Statement<[]>;
  private readonly stmtInsertCatalog: Database.Statement<{
    source: string;
    docCount: number;
    metadataKeysJson: string;
    sampleValuesJson: string;
  }>;
  private readonly stmtGetCatalog: Database.Statement<[], StoredCatalogRow>;

  private readonly txUpsertDocument: (
    doc: NormalizedDocument,
    chunks: readonly ChunkRecord[],
    graphArtifacts?: ExtractedGraphArtifacts
  ) => void;
  private readonly txUpsertDocumentGraph: (
    datasetDocUuid: string,
    contentHash: string,
    graphArtifacts: ExtractedGraphArtifacts
  ) => void;
  private readonly txDeleteDocument: (uuid: string) => boolean;
  private readonly txReplaceCatalog: (entries: readonly SchemaCatalogEntry[]) => void;
  private readonly txUpsertChunkEmbeddings: (
    entries: readonly SerializedEmbeddingRow[]
  ) => void;

  public constructor(dbPath: string) {
    if (dbPath !== ":memory:") {
      mkdirSync(dirname(dbPath), { recursive: true });
    }

    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("synchronous = NORMAL");
    this.db.pragma("cache_size = -131072");
    this.db.pragma("temp_store = MEMORY");
    this.db.pragma("mmap_size = 536870912");
    this.db.pragma("foreign_keys = ON");

    this.initializeSchema();

    this.stmtGetHash = this.db.prepare(
      "SELECT content_hash FROM documents WHERE dataset_doc_uuid = ?"
    );
    this.stmtListIds = this.db.prepare(
      "SELECT dataset_doc_uuid FROM documents ORDER BY dataset_doc_uuid ASC"
    );
    this.stmtHasId = this.db.prepare(
      "SELECT dataset_doc_uuid FROM documents WHERE dataset_doc_uuid = ? LIMIT 1"
    );
    this.stmtHasDocumentGraph = this.db.prepare(
      "SELECT 1 AS present FROM doc_aliases WHERE dataset_doc_uuid = ? AND content_hash = ? LIMIT 1"
    );
    this.stmtDeleteChunksByDoc = this.db.prepare(
      "DELETE FROM chunks WHERE dataset_doc_uuid = ?"
    );
    this.stmtDeleteAliasesByDoc = this.db.prepare(
      "DELETE FROM doc_aliases WHERE dataset_doc_uuid = ?"
    );
    this.stmtDeleteEdgesBySource = this.db.prepare(
      "DELETE FROM graph_edges WHERE source_doc_uuid = ?"
    );
    this.stmtDeleteEntityMentionsByDoc = this.db.prepare(
      "DELETE FROM entity_mentions WHERE dataset_doc_uuid = ?"
    );
    this.stmtUnresolveIncomingEdges = this.db.prepare(
      "UPDATE graph_edges SET target_doc_uuid = NULL WHERE target_doc_uuid = ?"
    );
    this.stmtInsertDocAlias = this.db.prepare(`
      INSERT OR REPLACE INTO doc_aliases (
        alias_key,
        dataset_doc_uuid,
        content_hash,
        alias_kind
      ) VALUES (
        @aliasKey,
        @datasetDocUuid,
        @contentHash,
        @aliasKind
      )
    `);
    this.stmtResolveIncomingByAlias = this.db.prepare(`
      UPDATE graph_edges
      SET target_doc_uuid = ?
      WHERE target_doc_uuid IS NULL
        AND source_doc_uuid <> ?
        AND target_alias = ?
    `);
    this.stmtInsertGraphEdge = this.db.prepare(`
      INSERT OR REPLACE INTO graph_edges (
        source_doc_uuid,
        target_doc_uuid,
        target_alias,
        edge_type,
        weight,
        reason,
        content_hash
      ) VALUES (
        @sourceDocUuid,
        (
          SELECT a.dataset_doc_uuid
          FROM doc_aliases a
          WHERE a.alias_key = @targetAlias
            AND a.dataset_doc_uuid <> @sourceDocUuid
          ORDER BY a.dataset_doc_uuid ASC
          LIMIT 1
        ),
        @targetAlias,
        @edgeType,
        @weight,
        @reason,
        @contentHash
      )
    `);
    this.stmtInsertEntityMention = this.db.prepare(`
      INSERT OR REPLACE INTO entity_mentions (
        entity_key,
        entity_type,
        entity_name,
        dataset_doc_uuid,
        structural_role,
        content_hash
      ) VALUES (
        @entityKey,
        @entityType,
        @entityName,
        @datasetDocUuid,
        @structuralRole,
        @contentHash
      )
    `);
    this.stmtResolveGraphEdges = this.db.prepare(`
      UPDATE graph_edges
      SET target_doc_uuid = (
        SELECT a.dataset_doc_uuid
        FROM doc_aliases a
        WHERE a.alias_key = graph_edges.target_alias
          AND a.dataset_doc_uuid <> graph_edges.source_doc_uuid
        ORDER BY a.dataset_doc_uuid ASC
        LIMIT 1
      )
      WHERE target_doc_uuid IS NULL
        AND EXISTS (
          SELECT 1
          FROM doc_aliases a
          WHERE a.alias_key = graph_edges.target_alias
            AND a.dataset_doc_uuid <> graph_edges.source_doc_uuid
        )
    `);
    this.stmtGetOutgoingEdges = this.db.prepare(`
      SELECT
        target_doc_uuid AS neighborUuid,
        weight,
        reason
      FROM graph_edges
      WHERE source_doc_uuid = ?
        AND target_doc_uuid IS NOT NULL
    `);
    this.stmtGetIncomingEdges = this.db.prepare(`
      SELECT
        source_doc_uuid AS neighborUuid,
        weight,
        reason
      FROM graph_edges
      WHERE target_doc_uuid = ?
    `);
    this.stmtGetDocEntitiesWithFreq = this.db.prepare(`
      SELECT
        em.entity_key AS entityKey,
        em.entity_type AS entityType,
        MIN(em.entity_name) AS entityName,
        (
          SELECT COUNT(DISTINCT em2.dataset_doc_uuid)
          FROM entity_mentions em2
          WHERE em2.entity_key = em.entity_key
        ) AS docFreq
      FROM entity_mentions em
      WHERE em.dataset_doc_uuid = ?
      GROUP BY em.entity_key, em.entity_type
    `);
    this.stmtGetEntityPeers = this.db.prepare(`
      SELECT DISTINCT dataset_doc_uuid AS neighborUuid
      FROM entity_mentions
      WHERE entity_key = ?
        AND dataset_doc_uuid <> ?
      ORDER BY dataset_doc_uuid ASC
      LIMIT ?
    `);
    this.stmtCountAliases = this.db.prepare(
      "SELECT COUNT(*) AS count FROM doc_aliases"
    );
    this.stmtCountEdges = this.db.prepare(
      "SELECT COUNT(*) AS count FROM graph_edges"
    );
    this.stmtCountResolvedEdges = this.db.prepare(
      "SELECT COUNT(*) AS count FROM graph_edges WHERE target_doc_uuid IS NOT NULL"
    );
    this.stmtCountEntityMentions = this.db.prepare(
      "SELECT COUNT(*) AS count FROM entity_mentions"
    );
    this.stmtDeleteDoc = this.db.prepare(
      "DELETE FROM documents WHERE dataset_doc_uuid = ?"
    );
    this.stmtUpsertDoc = this.db.prepare(`
      INSERT INTO documents (
        dataset_doc_uuid,
        content_hash,
        source,
        title,
        author,
        status,
        created_at,
        updated_at,
        metadata_json,
        full_text
      ) VALUES (
        @datasetDocUuid,
        @contentHash,
        @source,
        @title,
        @author,
        @status,
        @createdAt,
        @updatedAt,
        @metadataJson,
        @fullText
      )
      ON CONFLICT(dataset_doc_uuid) DO UPDATE SET
        content_hash = excluded.content_hash,
        source = excluded.source,
        title = excluded.title,
        author = excluded.author,
        status = excluded.status,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at,
        metadata_json = excluded.metadata_json,
        full_text = excluded.full_text
    `);
    this.stmtInsertChunk = this.db.prepare(`
      INSERT INTO chunks (
        dataset_doc_uuid,
        content_hash,
        chunk_index,
        title,
        metadata_text,
        body
      ) VALUES (
        @datasetDocUuid,
        @contentHash,
        @chunkIndex,
        @title,
        @metadataText,
        @body
      )
    `);
    this.stmtUpsertChunkEmbedding = this.db.prepare(`
      INSERT INTO chunk_embeddings (
        chunk_id,
        dataset_doc_uuid,
        content_hash,
        model,
        dims,
        embedding
      ) VALUES (
        @chunkId,
        @datasetDocUuid,
        @contentHash,
        @model,
        @dims,
        @embedding
      )
      ON CONFLICT(chunk_id, model, dims) DO UPDATE SET
        dataset_doc_uuid = excluded.dataset_doc_uuid,
        content_hash = excluded.content_hash,
        embedding = excluded.embedding
    `);
    this.stmtGetMissingEmbeddingChunks = this.db.prepare(`
      SELECT
        c.chunk_id AS chunkId,
        c.dataset_doc_uuid AS datasetDocUuid,
        c.content_hash AS contentHash,
        c.chunk_index AS chunkIndex,
        c.title AS title,
        c.metadata_text AS metadataText,
        c.body AS body
      FROM chunks c
      LEFT JOIN chunk_embeddings ce
        ON ce.chunk_id = c.chunk_id
       AND ce.model = ?
       AND ce.dims = ?
       AND ce.content_hash = c.content_hash
      WHERE ce.chunk_id IS NULL
      ORDER BY c.chunk_id ASC
    `);
    this.stmtCountEmbeddingsForModel = this.db.prepare(`
      SELECT COUNT(*) AS count
      FROM chunk_embeddings ce
      JOIN chunks c ON c.chunk_id = ce.chunk_id
      JOIN documents d ON d.dataset_doc_uuid = c.dataset_doc_uuid
      WHERE ce.model = ? AND ce.dims = ?
    `);
    this.stmtIterateEmbeddingsForModel = this.db.prepare(`
      SELECT
        ce.chunk_id AS chunkId,
        ce.dataset_doc_uuid AS datasetDocUuid,
        c.chunk_index AS chunkIndex,
        d.source AS source,
        c.title AS title,
        substr(c.body, 1, 320) AS snippet,
        ce.embedding AS embedding
      FROM chunk_embeddings ce
      JOIN chunks c ON c.chunk_id = ce.chunk_id
      JOIN documents d ON d.dataset_doc_uuid = c.dataset_doc_uuid
      WHERE ce.model = ? AND ce.dims = ?
      ORDER BY ce.chunk_id ASC
    `);
    this.stmtSearchFts = this.db.prepare(`
      SELECT
        c.chunk_id AS chunkId,
        c.dataset_doc_uuid AS datasetDocUuid,
        c.chunk_index AS chunkIndex,
        d.source AS source,
        c.title AS title,
        bm25(chunks_fts, 4.0, 2.0, 1.0) AS score,
        snippet(chunks_fts, 2, '[', ']', '...', 16) AS excerpt
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.dataset_doc_uuid = c.dataset_doc_uuid
      WHERE chunks_fts MATCH ?
      ORDER BY score ASC
      LIMIT ?
    `);
    this.stmtCountDocs = this.db.prepare("SELECT COUNT(*) AS count FROM documents");
    this.stmtCountChunks = this.db.prepare("SELECT COUNT(*) AS count FROM chunks");
    this.stmtCountSources = this.db.prepare(
      "SELECT COUNT(DISTINCT source) AS count FROM documents"
    );
    this.stmtReadAllDocMetadata = this.db.prepare(
      "SELECT source, metadata_json FROM documents"
    );
    this.stmtClearCatalog = this.db.prepare("DELETE FROM source_schema_catalog");
    this.stmtInsertCatalog = this.db.prepare(`
      INSERT INTO source_schema_catalog (
        source,
        doc_count,
        metadata_keys_json,
        sample_values_json
      ) VALUES (
        @source,
        @docCount,
        @metadataKeysJson,
        @sampleValuesJson
      )
    `);
    this.stmtGetCatalog = this.db.prepare(`
      SELECT source, doc_count, metadata_keys_json, sample_values_json
      FROM source_schema_catalog
      ORDER BY source ASC
    `);

    this.txUpsertDocument = this.db.transaction(
      (
        doc: NormalizedDocument,
        chunks: readonly ChunkRecord[],
        graphArtifacts?: ExtractedGraphArtifacts
      ) => {
        this.stmtDeleteChunksByDoc.run(doc.datasetDocUuid);
        this.stmtDeleteAliasesByDoc.run(doc.datasetDocUuid);
        this.stmtDeleteEdgesBySource.run(doc.datasetDocUuid);
        this.stmtDeleteEntityMentionsByDoc.run(doc.datasetDocUuid);
        this.stmtUnresolveIncomingEdges.run(doc.datasetDocUuid);
        this.stmtUpsertDoc.run({
          datasetDocUuid: doc.datasetDocUuid,
          contentHash: doc.contentHash,
          source: doc.source,
          title: doc.title,
          author: doc.author,
          status: doc.status,
          createdAt: doc.createdAt,
          updatedAt: doc.updatedAt,
          metadataJson: doc.metadataJson,
          fullText: doc.fullText
        });
        for (const chunk of chunks) {
          this.stmtInsertChunk.run({
            datasetDocUuid: chunk.datasetDocUuid,
            contentHash: chunk.contentHash,
            chunkIndex: chunk.chunkIndex,
            title: chunk.title,
            metadataText: chunk.metadataText,
            body: chunk.body
          });
        }
        if (graphArtifacts !== undefined) {
          this.insertDocumentGraphRows(
            doc.datasetDocUuid,
            doc.contentHash,
            graphArtifacts
          );
        }
      }
    );

    this.txUpsertDocumentGraph = this.db.transaction(
      (
        datasetDocUuid: string,
        contentHash: string,
        graphArtifacts: ExtractedGraphArtifacts
      ) => {
        this.stmtDeleteAliasesByDoc.run(datasetDocUuid);
        this.stmtDeleteEdgesBySource.run(datasetDocUuid);
        this.stmtDeleteEntityMentionsByDoc.run(datasetDocUuid);
        this.stmtUnresolveIncomingEdges.run(datasetDocUuid);
        this.insertDocumentGraphRows(
          datasetDocUuid,
          contentHash,
          graphArtifacts
        );
      }
    );

    this.txDeleteDocument = this.db.transaction((uuid: string) => {
      this.stmtDeleteChunksByDoc.run(uuid);
      const info = this.stmtDeleteDoc.run(uuid);
      return info.changes > 0;
    });

    this.txReplaceCatalog = this.db.transaction(
      (entries: readonly SchemaCatalogEntry[]) => {
        this.stmtClearCatalog.run();
        for (const entry of entries) {
          this.stmtInsertCatalog.run({
            source: entry.source,
            docCount: entry.docCount,
            metadataKeysJson: JSON.stringify(entry.metadataKeys),
            sampleValuesJson: JSON.stringify(entry.sampleValues)
          });
        }
      }
    );

    this.txUpsertChunkEmbeddings = this.db.transaction(
      (entries: readonly SerializedEmbeddingRow[]) => {
        for (const entry of entries) {
          this.stmtUpsertChunkEmbedding.run(entry);
        }
      }
    );
  }

  private initializeSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS documents (
        dataset_doc_uuid TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        author TEXT,
        status TEXT,
        created_at TEXT,
        updated_at TEXT,
        metadata_json TEXT NOT NULL,
        full_text TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_doc_uuid TEXT NOT NULL REFERENCES documents(dataset_doc_uuid) ON DELETE CASCADE,
        content_hash TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        title TEXT NOT NULL,
        metadata_text TEXT NOT NULL,
        body TEXT NOT NULL,
        UNIQUE(dataset_doc_uuid, chunk_index)
      );

      CREATE INDEX IF NOT EXISTS idx_chunks_doc_uuid ON chunks(dataset_doc_uuid);
      CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

      CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        dataset_doc_uuid TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        model TEXT NOT NULL,
        dims INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (chunk_id, model, dims)
      );

      CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_dims
        ON chunk_embeddings(model, dims);
      CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc_uuid
        ON chunk_embeddings(dataset_doc_uuid);

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        title,
        metadata_text,
        body,
        content='chunks',
        content_rowid='chunk_id',
        tokenize='porter unicode61'
      );

      CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, title, metadata_text, body)
        VALUES (new.chunk_id, new.title, new.metadata_text, new.body);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, metadata_text, body)
        VALUES ('delete', old.chunk_id, old.title, old.metadata_text, old.body);
      END;

      CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, metadata_text, body)
        VALUES ('delete', old.chunk_id, old.title, old.metadata_text, old.body);
        INSERT INTO chunks_fts(rowid, title, metadata_text, body)
        VALUES (new.chunk_id, new.title, new.metadata_text, new.body);
      END;

      CREATE TABLE IF NOT EXISTS source_schema_catalog (
        source TEXT PRIMARY KEY,
        doc_count INTEGER NOT NULL,
        metadata_keys_json TEXT NOT NULL,
        sample_values_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS doc_aliases (
        alias_key TEXT NOT NULL,
        dataset_doc_uuid TEXT NOT NULL REFERENCES documents(dataset_doc_uuid) ON DELETE CASCADE,
        content_hash TEXT NOT NULL,
        alias_kind TEXT NOT NULL,
        PRIMARY KEY (alias_key, dataset_doc_uuid)
      );

      CREATE INDEX IF NOT EXISTS idx_doc_aliases_alias_key
        ON doc_aliases(alias_key);
      CREATE INDEX IF NOT EXISTS idx_doc_aliases_doc_uuid
        ON doc_aliases(dataset_doc_uuid);

      CREATE TABLE IF NOT EXISTS graph_edges (
        source_doc_uuid TEXT NOT NULL REFERENCES documents(dataset_doc_uuid) ON DELETE CASCADE,
        target_doc_uuid TEXT REFERENCES documents(dataset_doc_uuid) ON DELETE SET NULL,
        target_alias TEXT NOT NULL,
        edge_type TEXT NOT NULL,
        weight REAL NOT NULL,
        reason TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        PRIMARY KEY (source_doc_uuid, target_alias, edge_type)
      );

      CREATE INDEX IF NOT EXISTS idx_graph_edges_source
        ON graph_edges(source_doc_uuid);
      CREATE INDEX IF NOT EXISTS idx_graph_edges_target
        ON graph_edges(target_doc_uuid);
      CREATE INDEX IF NOT EXISTS idx_graph_edges_target_alias
        ON graph_edges(target_alias);

      CREATE TABLE IF NOT EXISTS entity_mentions (
        entity_key TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        dataset_doc_uuid TEXT NOT NULL REFERENCES documents(dataset_doc_uuid) ON DELETE CASCADE,
        structural_role TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        PRIMARY KEY (entity_key, dataset_doc_uuid, structural_role)
      );

      CREATE INDEX IF NOT EXISTS idx_entity_mentions_key
        ON entity_mentions(entity_key);
      CREATE INDEX IF NOT EXISTS idx_entity_mentions_type
        ON entity_mentions(entity_type);
      CREATE INDEX IF NOT EXISTS idx_entity_mentions_name
        ON entity_mentions(entity_name);
      CREATE INDEX IF NOT EXISTS idx_entity_mentions_doc_uuid
        ON entity_mentions(dataset_doc_uuid);
    `);
  }

  public getDocumentHash(uuid: string): string | null {
    const row = this.stmtGetHash.get(uuid);
    return row?.content_hash ?? null;
  }

  public listDocumentIds(): string[] {
    return this.stmtListIds.all().map((row) => row.dataset_doc_uuid);
  }

  private removeDocFromVectorCache(uuid: string): void {
    const cache = this.vectorCache;
    if (cache === null || cache.count === 0) {
      return;
    }

    const dims = cache.dims;
    let i = 0;
    while (i < cache.count) {
      if (cache.docUuids[i] !== uuid) {
        i += 1;
        continue;
      }

      const last = cache.count - 1;
      if (i !== last) {
        cache.vectors.copyWithin(i * dims, last * dims, (last + 1) * dims);
        cache.chunkIds[i] = cache.chunkIds[last] ?? 0;
        cache.docUuids[i] = cache.docUuids[last] ?? "";
        cache.chunkIndices[i] = cache.chunkIndices[last] ?? 0;
        cache.sources[i] = cache.sources[last] ?? "";
        cache.titles[i] = cache.titles[last] ?? "";
        cache.excerpts[i] = cache.excerpts[last] ?? "";
      }

      cache.chunkIds.pop();
      cache.docUuids.pop();
      cache.chunkIndices.pop();
      cache.sources.pop();
      cache.titles.pop();
      cache.excerpts.pop();
      cache.count -= 1;
    }
  }

  public upsertDocument(
    doc: NormalizedDocument,
    chunks: readonly ChunkRecord[],
    graphArtifacts?: ExtractedGraphArtifacts
  ): void {
    try {
      this.txUpsertDocument(doc, chunks, graphArtifacts);
      this.removeDocFromVectorCache(doc.datasetDocUuid);
    } catch (error) {
      throw new IngestionError(
        `Failed to upsert document ${doc.datasetDocUuid}`,
        { cause: error }
      );
    }
  }

  private insertDocumentGraphRows(
    datasetDocUuid: string,
    contentHash: string,
    graphArtifacts: ExtractedGraphArtifacts
  ): void {
    for (const alias of graphArtifacts.aliases) {
      this.stmtInsertDocAlias.run({
        ...alias,
        datasetDocUuid,
        contentHash
      });
    }
    for (const alias of graphArtifacts.aliases) {
      this.stmtResolveIncomingByAlias.run(
        datasetDocUuid,
        datasetDocUuid,
        alias.aliasKey
      );
    }
    for (const ref of graphArtifacts.refs) {
      this.stmtInsertGraphEdge.run({
        ...ref,
        sourceDocUuid: datasetDocUuid,
        contentHash
      });
    }
    for (const mention of graphArtifacts.entityMentions) {
      this.stmtInsertEntityMention.run({
        ...mention,
        datasetDocUuid,
        contentHash
      });
    }
  }

  public hasDocumentGraph(datasetDocUuid: string, contentHash: string): boolean {
    return this.stmtHasDocumentGraph.get(datasetDocUuid, contentHash) !== undefined;
  }

  public upsertDocumentGraph(
    datasetDocUuid: string,
    contentHash: string,
    graphArtifacts: ExtractedGraphArtifacts
  ): void {
    try {
      this.txUpsertDocumentGraph(datasetDocUuid, contentHash, graphArtifacts);
    } catch (error) {
      throw new IngestionError(
        `Failed to upsert graph for document ${datasetDocUuid}`,
        { cause: error }
      );
    }
  }

  public resolveGraphEdges(): number {
    try {
      const info = this.stmtResolveGraphEdges.run();
      return info.changes;
    } catch (error) {
      throw new IngestionError("Failed to resolve graph edges", {
        cause: error
      });
    }
  }

  public deleteDocument(uuid: string): boolean {
    try {
      const deleted = this.txDeleteDocument(uuid);
      if (deleted) {
        this.removeDocFromVectorCache(uuid);
      }
      return deleted;
    } catch (error) {
      throw new IngestionError(`Failed to delete document ${uuid}`, {
        cause: error
      });
    }
  }

  public upsertChunkEmbeddings(entries: readonly ChunkEmbeddingRecord[]): void {
    if (entries.length === 0) {
      return;
    }

    const serialized: SerializedEmbeddingRow[] = [];
    for (const entry of entries) {
      if (entry.model.trim().length === 0) {
        throw new ValidationError("Chunk embedding model must not be empty");
      }
      if (!Number.isInteger(entry.dims) || entry.dims <= 0) {
        throw new ValidationError(
          `Chunk embedding dims must be a positive integer, got ${entry.dims}`
        );
      }
      if (entry.vector.length !== entry.dims) {
        throw new ValidationError(
          `Chunk embedding vector length ${entry.vector.length} does not match dims ${entry.dims}`
        );
      }

      serialized.push({
        chunkId: entry.chunkId,
        datasetDocUuid: entry.datasetDocUuid,
        contentHash: entry.contentHash,
        model: entry.model,
        dims: entry.dims,
        embedding: float32ToBuffer(l2Normalize(entry.vector))
      });
    }

    try {
      this.txUpsertChunkEmbeddings(serialized);
    } catch (error) {
      throw new IngestionError("Failed to upsert chunk embeddings", {
        cause: error
      });
    }

    if (this.vectorCache !== null) {
      const loadedModel = this.vectorCache.model;
      const loadedDims = this.vectorCache.dims;
      if (
        entries.some(
          (entry) => entry.model === loadedModel && entry.dims === loadedDims
        )
      ) {
        this.vectorCache = null;
      }
    }
  }

  public getMissingEmbeddingChunks(
    model: string,
    dims: number
  ): ChunkForEmbedding[] {
    if (model.trim().length === 0) {
      throw new ValidationError("Embedding model must not be empty");
    }
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new ValidationError(
        `Embedding dims must be a positive integer, got ${dims}`
      );
    }

    return this.stmtGetMissingEmbeddingChunks.all(model, dims);
  }

  public loadVectorCache(model: string, dims: number): VectorCacheStats {
    if (model.trim().length === 0) {
      throw new ValidationError("Embedding model must not be empty");
    }
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new ValidationError(
        `Embedding dims must be a positive integer, got ${dims}`
      );
    }

    try {
      const totalCount =
        this.stmtCountEmbeddingsForModel.get(model, dims)?.count ?? 0;
      const vectors = new Float32Array(totalCount * dims);
      const chunkIds: number[] = [];
      const docUuids: string[] = [];
      const chunkIndices: number[] = [];
      const sources: string[] = [];
      const titles: string[] = [];
      const excerpts: string[] = [];

      let index = 0;
      for (const row of this.stmtIterateEmbeddingsForModel.iterate(model, dims)) {
        if (index >= totalCount) {
          break;
        }
        const decoded = bufferToFloat32(row.embedding, dims);
        const normalized = l2Normalize(decoded);
        vectors.set(normalized, index * dims);
        chunkIds.push(row.chunkId);
        docUuids.push(row.datasetDocUuid);
        chunkIndices.push(row.chunkIndex);
        sources.push(row.source);
        titles.push(row.title);
        excerpts.push(formatDenseExcerpt(row.snippet));
        index += 1;
      }

      this.vectorCache = {
        model,
        dims,
        count: index,
        vectors,
        chunkIds,
        docUuids,
        chunkIndices,
        sources,
        titles,
        excerpts
      };

      return this.getVectorCacheStats();
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new RetrievalError(
        `Failed to load vector cache for ${model} (${dims}d)`,
        { cause: error }
      );
    }
  }

  public searchDense(
    queryVector: Float32Array,
    topK: number
  ): DenseSearchHit[] {
    if (topK <= 0) {
      return [];
    }

    const cache = this.vectorCache;
    if (cache === null) {
      throw new RetrievalError(
        "Vector cache is not loaded; call loadVectorCache before searchDense"
      );
    }
    if (queryVector.length !== cache.dims) {
      throw new ValidationError(
        `Query vector length ${queryVector.length} does not match loaded cache dims ${cache.dims}`
      );
    }
    if (cache.count === 0) {
      return [];
    }

    const normalizedQuery = l2Normalize(queryVector);
    const dims = cache.dims;
    const limit = Math.min(topK, cache.count);
    const topCandidates: RankedDenseCandidate[] = [];

    for (let i = 0; i < cache.count; i += 1) {
      const score = dotProduct(cache.vectors, i * dims, normalizedQuery, dims);
      const candidate: RankedDenseCandidate = {
        index: i,
        score,
        docUuid: cache.docUuids[i] ?? "",
        chunkId: cache.chunkIds[i] ?? 0
      };

      if (topCandidates.length < limit) {
        topCandidates.push(candidate);
        if (topCandidates.length === limit) {
          topCandidates.sort(compareDenseCandidates);
        }
        continue;
      }

      const worst = topCandidates[limit - 1];
      if (worst !== undefined && compareDenseCandidates(candidate, worst) < 0) {
        topCandidates[limit - 1] = candidate;
        let pos = limit - 1;
        while (pos > 0) {
          const prev = topCandidates[pos - 1];
          const curr = topCandidates[pos];
          if (
            prev !== undefined &&
            curr !== undefined &&
            compareDenseCandidates(curr, prev) < 0
          ) {
            topCandidates[pos - 1] = curr;
            topCandidates[pos] = prev;
            pos -= 1;
          } else {
            break;
          }
        }
      }
    }

    if (topCandidates.length < limit) {
      topCandidates.sort(compareDenseCandidates);
    }

    return topCandidates.map((item) => ({
      chunkId: cache.chunkIds[item.index] ?? 0,
      datasetDocUuid: cache.docUuids[item.index] ?? "",
      chunkIndex: cache.chunkIndices[item.index] ?? 0,
      source: cache.sources[item.index] ?? "",
      title: cache.titles[item.index] ?? "",
      score: item.score,
      excerpt: cache.excerpts[item.index] ?? ""
    }));
  }

  public getVectorCacheStats(): VectorCacheStats {
    const cache = this.vectorCache;
    if (cache === null) {
      return {
        loaded: false,
        model: null,
        dims: 0,
        vectorCount: 0,
        byteLength: 0
      };
    }
    return {
      loaded: true,
      model: cache.model,
      dims: cache.dims,
      vectorCount: cache.count,
      byteLength: cache.count * cache.dims * Float32Array.BYTES_PER_ELEMENT
    };
  }

  public rebuildSchemaCatalog(): SchemaCatalog {
    const MAX_CATALOG_KEYS_PER_SOURCE = 24;
    const MAX_ENUM_DISTINCT_VALUES = 12;
    const MAX_ENUM_VALUE_CHARS = 36;
    const MIN_KEY_DOC_FRACTION = 0.1;

    const grouped = new Map<
      string,
      {
        docCount: number;
        keyDocCounts: Map<string, number>;
        valueCounts: Map<string, Map<string, number>>;
        overflowedKeys: Set<string>;
      }
    >();

    for (const row of this.stmtReadAllDocMetadata.iterate()) {
      let bucket = grouped.get(row.source);
      if (bucket === undefined) {
        bucket = {
          docCount: 0,
          keyDocCounts: new Map<string, number>(),
          valueCounts: new Map<string, Map<string, number>>(),
          overflowedKeys: new Set<string>()
        };
        grouped.set(row.source, bucket);
      }
      bucket.docCount += 1;

      let parsedMetadata: unknown;
      try {
        parsedMetadata = JSON.parse(row.metadata_json);
      } catch (error) {
        throw new IngestionError(
          `Corrupt metadata_json stored for source ${row.source}`,
          { cause: error }
        );
      }

      if (
        typeof parsedMetadata !== "object" ||
        parsedMetadata === null ||
        Array.isArray(parsedMetadata)
      ) {
        continue;
      }

      for (const [key, value] of Object.entries(parsedMetadata)) {
        bucket.keyDocCounts.set(key, (bucket.keyDocCounts.get(key) ?? 0) + 1);
        const formatted = formatSampleScalar(value);
        if (formatted === null) {
          continue;
        }
        let counts = bucket.valueCounts.get(key);
        if (counts === undefined) {
          counts = new Map<string, number>();
          bucket.valueCounts.set(key, counts);
        }
        const existingCount = counts.get(formatted);
        if (existingCount !== undefined) {
          counts.set(formatted, existingCount + 1);
        } else if (counts.size < MAX_ENUM_DISTINCT_VALUES) {
          counts.set(formatted, 1);
        } else {
          bucket.overflowedKeys.add(key);
        }
      }
    }

    const entries: SchemaCatalogEntry[] = [];
    const sortedSources = [...grouped.keys()].sort((a, b) => a.localeCompare(b));
    for (const source of sortedSources) {
      const bucket = grouped.get(source);
      if (bucket === undefined) {
        continue;
      }

      const allKeys = [...bucket.keyDocCounts.entries()].sort(
        (a, b) => b[1] - a[1] || a[0].localeCompare(b[0])
      );
      const minDocThreshold =
        allKeys.length <= 16
          ? 1
          : Math.max(2, Math.ceil(bucket.docCount * MIN_KEY_DOC_FRACTION));
      const frequentKeys = allKeys
        .filter(([, count]) => count >= minDocThreshold)
        .slice(0, MAX_CATALOG_KEYS_PER_SOURCE)
        .map(([key]) => key);
      const metadataKeys =
        frequentKeys.length > 0
          ? frequentKeys
          : allKeys.slice(0, MAX_CATALOG_KEYS_PER_SOURCE).map(([key]) => key);

      const sampleValues: Record<string, string[]> = {};
      for (const key of metadataKeys) {
        const counts = bucket.valueCounts.get(key);
        if (counts === undefined || counts.size === 0) {
          continue;
        }
        const sortedValues = [...counts.entries()].sort(
          (a, b) => b[1] - a[1] || a[0].localeCompare(b[0])
        );
        const isEnumLike =
          !bucket.overflowedKeys.has(key) &&
          sortedValues.every(([val]) => val.length <= MAX_ENUM_VALUE_CHARS);
        if (isEnumLike) {
          sampleValues[key] = sortedValues.map(([val]) => val);
        }
      }

      entries.push({
        source,
        docCount: bucket.docCount,
        metadataKeys,
        sampleValues
      });
    }

    this.txReplaceCatalog(entries);
    return { sources: entries };
  }

  public getSchemaCatalog(): SchemaCatalog {
    const rows = this.stmtGetCatalog.all();
    const sources: SchemaCatalogEntry[] = rows.map((row) => ({
      source: row.source,
      docCount: row.doc_count,
      metadataKeys: JSON.parse(row.metadata_keys_json) as string[],
      sampleValues: JSON.parse(row.sample_values_json) as Record<string, string[]>
    }));
    return { sources };
  }

  public searchFts(query: string, topK: number): FtsSearchHit[] {
    if (topK <= 0) {
      return [];
    }
    const candidates = buildFtsQueryCandidates(query);
    if (candidates.length === 0) {
      return [];
    }

    const combined: FtsSearchHit[] = [];
    const seenChunkIds = new Set<number>();

    for (const candidateQuery of candidates) {
      const remaining = topK - combined.length;
      if (remaining <= 0) {
        break;
      }
      try {
        const rows = this.stmtSearchFts.all(candidateQuery, topK + seenChunkIds.size);
        for (const row of rows) {
          if (seenChunkIds.has(row.chunkId)) {
            continue;
          }
          seenChunkIds.add(row.chunkId);
          combined.push(row);
          if (combined.length >= topK) {
            break;
          }
        }
      } catch (error) {
        throw new RetrievalError(`FTS5 search failed for query "${candidateQuery}"`, {
          cause: error
        });
      }
    }

    return combined;
  }

  public filterMetadata(query: MetadataFilterQueryInput): MetadataFilterResult {
    const parsed = MetadataFilterQuerySchema.safeParse(query);
    if (!parsed.success) {
      throw new ValidationError(
        `Invalid metadata filter query: ${parsed.error.message}`
      );
    }

    const filter = parsed.data;
    const clauses: string[] = [];
    const params: unknown[] = [];

    if (filter.source !== undefined) {
      clauses.push("LOWER(source) = LOWER(?)");
      params.push(filter.source);
    }
    if (filter.author !== undefined) {
      clauses.push("author IS NOT NULL AND LOWER(author) = LOWER(?)");
      params.push(filter.author);
    }
    if (filter.status !== undefined) {
      clauses.push("status IS NOT NULL AND LOWER(status) = LOWER(?)");
      params.push(filter.status);
    }
    if (filter.createdAfter !== undefined) {
      clauses.push("created_at IS NOT NULL AND created_at >= ?");
      params.push(filter.createdAfter);
    }
    if (filter.createdBefore !== undefined) {
      clauses.push("created_at IS NOT NULL AND created_at <= ?");
      params.push(filter.createdBefore);
    }
    if (filter.updatedAfter !== undefined) {
      clauses.push("updated_at IS NOT NULL AND updated_at >= ?");
      params.push(filter.updatedAfter);
    }
    if (filter.updatedBefore !== undefined) {
      clauses.push("updated_at IS NOT NULL AND updated_at <= ?");
      params.push(filter.updatedBefore);
    }
    if (filter.titleContains !== undefined) {
      clauses.push("LOWER(title) LIKE LOWER(?) ESCAPE '\\'");
      params.push(`%${escapeLikePattern(filter.titleContains)}%`);
    }

    if (filter.metadataFilters !== undefined) {
      for (const metaFilter of filter.metadataFilters) {
        const jsonPath = buildSafeJsonExtractPath(metaFilter.key);
        if (metaFilter.op === "eq") {
          const expected = metaFilter.value ?? "";
          const lowered = expected.toLowerCase();
          if (lowered === "true" || lowered === "false") {
            clauses.push(
              "(json_extract(metadata_json, ?) IS NOT NULL AND (LOWER(CAST(json_extract(metadata_json, ?) AS TEXT)) = LOWER(?) OR json_type(metadata_json, ?) = ?))"
            );
            params.push(jsonPath, jsonPath, expected, jsonPath, lowered);
          } else {
            clauses.push(
              "(json_extract(metadata_json, ?) IS NOT NULL AND LOWER(CAST(json_extract(metadata_json, ?) AS TEXT)) = LOWER(?))"
            );
            params.push(jsonPath, jsonPath, expected);
          }
        } else if (metaFilter.op === "contains") {
          const needle = metaFilter.value ?? "";
          clauses.push(
            "(json_extract(metadata_json, ?) IS NOT NULL AND LOWER(CAST(json_extract(metadata_json, ?) AS TEXT)) LIKE LOWER(?) ESCAPE '\\')"
          );
          params.push(jsonPath, jsonPath, `%${escapeLikePattern(needle)}%`);
        } else {
          const values = metaFilter.values ?? [];
          const loweredValues = values.map((item) => item.toLowerCase());
          const placeholders = loweredValues.map(() => "?").join(", ");
          const boolTypes = loweredValues.filter(
            (item) => item === "true" || item === "false"
          );
          if (boolTypes.length > 0) {
            const boolPlaceholders = boolTypes.map(() => "?").join(", ");
            clauses.push(
              `(json_extract(metadata_json, ?) IS NOT NULL AND (LOWER(CAST(json_extract(metadata_json, ?) AS TEXT)) IN (${placeholders}) OR json_type(metadata_json, ?) IN (${boolPlaceholders})))`
            );
            params.push(jsonPath, jsonPath, ...loweredValues, jsonPath, ...boolTypes);
          } else {
            clauses.push(
              `(json_extract(metadata_json, ?) IS NOT NULL AND LOWER(CAST(json_extract(metadata_json, ?) AS TEXT)) IN (${placeholders}))`
            );
            params.push(jsonPath, jsonPath, ...loweredValues);
          }
        }
      }
    }

    const whereSql = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
    const orderSql = ORDER_BY_SQL[filter.orderBy];

    try {
      const countRow = this.db
        .prepare<unknown[], CountRow>(
          `SELECT COUNT(*) AS count FROM documents ${whereSql}`
        )
        .get(...params);

      const rows = this.db
        .prepare<unknown[], MetadataSummaryRow>(
          `
          SELECT
            dataset_doc_uuid,
            source,
            title,
            author,
            status,
            created_at,
            updated_at
          FROM documents
          ${whereSql}
          ORDER BY ${orderSql}
          LIMIT ?
          `
        )
        .all(...params, filter.limit);

      return {
        totalCount: countRow?.count ?? 0,
        documents: rows.map(toMetadataFilterRow)
      };
    } catch (error) {
      throw new RetrievalError("Metadata filter query failed", {
        cause: error
      });
    }
  }

  public fetchDocs(uuids: readonly string[]): StoredDocument[] {
    if (uuids.length === 0) {
      return [];
    }

    const uniqueOrdered: string[] = [];
    const seen = new Set<string>();
    for (const uuid of uuids) {
      if (!seen.has(uuid)) {
        seen.add(uuid);
        uniqueOrdered.push(uuid);
      }
    }

    const byId = new Map<string, StoredDocument>();
    for (let offset = 0; offset < uniqueOrdered.length; offset += FETCH_BATCH_SIZE) {
      const batch = uniqueOrdered.slice(offset, offset + FETCH_BATCH_SIZE);
      const placeholders = batch.map(() => "?").join(", ");
      const rows = this.db
        .prepare<string[], DocumentRow>(
          `
          SELECT
            dataset_doc_uuid,
            content_hash,
            source,
            title,
            author,
            status,
            created_at,
            updated_at,
            metadata_json,
            full_text
          FROM documents
          WHERE dataset_doc_uuid IN (${placeholders})
          `
        )
        .all(...batch);
      for (const row of rows) {
        byId.set(row.dataset_doc_uuid, toStoredDocument(row));
      }
    }

    const ordered: StoredDocument[] = [];
    for (const uuid of uniqueOrdered) {
      const doc = byId.get(uuid);
      if (doc !== undefined) {
        ordered.push(doc);
      }
    }
    return ordered;
  }

  public hasDocumentId(uuid: string): boolean {
    return this.stmtHasId.get(uuid) !== undefined;
  }

  private recordGraphCandidate(
    candidates: Map<string, GraphCandidateAccumulator>,
    excludedDocId: string | undefined,
    neighborUuid: string,
    hop: number,
    deltaScore: number,
    reason: string
  ): void {
    if (
      (excludedDocId !== undefined && neighborUuid === excludedDocId) ||
      deltaScore <= 0
    ) {
      return;
    }

    let entry = candidates.get(neighborUuid);
    if (entry === undefined) {
      entry = {
        bestHop: hop,
        score: 0,
        reasonWeights: new Map<string, number>()
      };
      candidates.set(neighborUuid, entry);
    }

    if (hop < entry.bestHop) {
      entry.bestHop = hop;
    }

    const prevWeight = entry.reasonWeights.get(reason);
    if (prevWeight === undefined) {
      entry.reasonWeights.set(reason, deltaScore);
      entry.score += deltaScore;
    } else if (deltaScore > prevWeight) {
      entry.reasonWeights.set(reason, deltaScore);
      entry.score += deltaScore - prevWeight;
    }
  }

  private expandOneHopFromDoc(
    seedDocUuid: string,
    hop: number,
    decay: number,
    excludedDocId: string | undefined,
    candidates: Map<string, GraphCandidateAccumulator>
  ): void {
    const outgoing = this.stmtGetOutgoingEdges.all(seedDocUuid);
    for (const edge of outgoing) {
      this.recordGraphCandidate(
        candidates,
        excludedDocId,
        edge.neighborUuid,
        hop,
        edge.weight * decay,
        edge.reason
      );
    }

    const incoming = this.stmtGetIncomingEdges.all(seedDocUuid);
    for (const edge of incoming) {
      this.recordGraphCandidate(
        candidates,
        excludedDocId,
        edge.neighborUuid,
        hop,
        edge.weight * decay,
        `referenced_by: ${edge.reason}`
      );
    }

    const entities = this.stmtGetDocEntitiesWithFreq.all(seedDocUuid);
    for (const entity of entities) {
      if (entity.docFreq < 2) {
        continue;
      }
      const typeWeight = ENTITY_TYPE_WEIGHTS[entity.entityType];
      const specificityScore =
        (typeWeight / Math.log2(2 + entity.docFreq)) * decay;
      const reason = `shared ${entity.entityType}: ${entity.entityName}`;
      const peers = this.stmtGetEntityPeers.all(
        entity.entityKey,
        seedDocUuid,
        MAX_SHARED_ENTITY_PEERS
      );
      for (const peer of peers) {
        this.recordGraphCandidate(
          candidates,
          excludedDocId,
          peer.neighborUuid,
          hop,
          specificityScore,
          reason
        );
      }
    }
  }

  private lookupEntityMentions(
    entityName: string,
    entityType?: EntityType
  ): { rows: EntityMatchRow[]; exact: boolean } {
    const normalized = normalizeEntityName(entityName);
    if (normalized.length === 0) {
      return { rows: [], exact: true };
    }

    const exactKeys =
      entityType !== undefined
        ? [buildEntityKey(entityType, normalized)]
        : ENTITY_TYPES.map((type) => buildEntityKey(type, normalized));
    const exactPlaceholders = exactKeys.map(() => "?").join(", ");

    const exactRows = this.db
      .prepare<unknown[], EntityMatchRow>(
        `
        SELECT
          dataset_doc_uuid AS datasetDocUuid,
          entity_type AS entityType,
          entity_name AS entityName,
          structural_role AS structuralRole
        FROM entity_mentions
        WHERE entity_key IN (${exactPlaceholders})
        ORDER BY dataset_doc_uuid ASC, structural_role ASC
        LIMIT ?
        `
      )
      .all(...exactKeys, MAX_ENTITY_LOOKUP_ROWS);

    if (exactRows.length > 0) {
      return { rows: exactRows, exact: true };
    }

    const pattern = `%${escapeLikePattern(normalized)}%`;
    const typeClause = entityType !== undefined ? "AND entity_type = ?" : "";
    const params: unknown[] =
      entityType !== undefined
        ? [pattern, entityType, MAX_ENTITY_LOOKUP_ROWS]
        : [pattern, MAX_ENTITY_LOOKUP_ROWS];

    const fuzzyRows = this.db
      .prepare<unknown[], EntityMatchRow>(
        `
        SELECT
          dataset_doc_uuid AS datasetDocUuid,
          entity_type AS entityType,
          entity_name AS entityName,
          structural_role AS structuralRole
        FROM entity_mentions
        WHERE LOWER(entity_name) LIKE LOWER(?) ESCAPE '\\'
        ${typeClause}
        ORDER BY dataset_doc_uuid ASC, structural_role ASC
        LIMIT ?
        `
      )
      .all(...params);

    return { rows: fuzzyRows, exact: false };
  }

  private fetchGraphDocMetadata(
    uuids: readonly string[]
  ): Map<string, GraphDocMetaRow> {
    const byId = new Map<string, GraphDocMetaRow>();
    for (let offset = 0; offset < uuids.length; offset += FETCH_BATCH_SIZE) {
      const batch = uuids.slice(offset, offset + FETCH_BATCH_SIZE);
      const placeholders = batch.map(() => "?").join(", ");
      const rows = this.db
        .prepare<string[], GraphDocMetaRow>(
          `
          SELECT
            dataset_doc_uuid,
            source,
            title,
            author,
            status,
            updated_at
          FROM documents
          WHERE dataset_doc_uuid IN (${placeholders})
          `
        )
        .all(...batch);
      for (const row of rows) {
        byId.set(row.dataset_doc_uuid, row);
      }
    }
    return byId;
  }

  public expandGraph(query: ExpandGraphQueryInput): ExpandGraphNeighbor[] {
    const parsed = ExpandGraphQuerySchema.safeParse(query);
    if (!parsed.success) {
      throw new ValidationError(
        `Invalid expandGraph query: ${parsed.error.message}`
      );
    }

    const { docId, entityName, entityType, maxHops, limit } = parsed.data;
    const candidates = new Map<string, GraphCandidateAccumulator>();

    try {
      if (docId !== undefined && this.hasDocumentId(docId)) {
        this.expandOneHopFromDoc(docId, 1, 1.0, docId, candidates);
      }

      if (entityName !== undefined) {
        const lookup = this.lookupEntityMentions(entityName, entityType);
        const matchBase = lookup.exact ? 1.0 : 0.65;
        for (const row of lookup.rows) {
          const score =
            matchBase * ENTITY_TYPE_WEIGHTS[row.entityType] +
            getStructuralRoleBonus(row.structuralRole);
          this.recordGraphCandidate(
            candidates,
            docId,
            row.datasetDocUuid,
            1,
            score,
            `entity match (${row.entityType}, ${row.structuralRole}): ${row.entityName}`
          );
        }
      }

      if (maxHops === 2 && candidates.size > 0) {
        const frontierLimit = Math.max(limit, HOP2_FRONTIER_MIN);
        const hop1Frontier = [...candidates.entries()]
          .filter(([, acc]) => acc.bestHop === 1)
          .sort((a, b) => {
            if (Math.abs(b[1].score - a[1].score) > 1e-12) {
              return b[1].score - a[1].score;
            }
            return a[0].localeCompare(b[0]);
          })
          .slice(0, frontierLimit);

        for (const [frontierUuid] of hop1Frontier) {
          this.expandOneHopFromDoc(
            frontierUuid,
            2,
            HOP2_DECAY,
            docId,
            candidates
          );
        }
      }

      if (candidates.size === 0) {
        return [];
      }

      const candidateIds = [...candidates.keys()];
      const metaById = this.fetchGraphDocMetadata(candidateIds);
      const neighbors: ExpandGraphNeighbor[] = [];

      for (const [uuid, acc] of candidates.entries()) {
        const meta = metaById.get(uuid);
        if (meta === undefined) {
          continue;
        }
        const sortedReasons = [...acc.reasonWeights.entries()]
          .sort((a, b) => {
            if (Math.abs(b[1] - a[1]) > 1e-12) {
              return b[1] - a[1];
            }
            return a[0].localeCompare(b[0]);
          })
          .slice(0, MAX_REASONS_PER_NEIGHBOR)
          .map(([reason]) => reason);

        neighbors.push({
          datasetDocUuid: meta.dataset_doc_uuid,
          source: meta.source,
          title: meta.title,
          author: meta.author,
          status: meta.status,
          updatedAt: meta.updated_at,
          hop: acc.bestHop,
          score: acc.score,
          reasons: sortedReasons
        });
      }

      neighbors.sort(compareExpandNeighbors);
      return neighbors.slice(0, limit);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new RetrievalError("Link Graph expansion failed", {
        cause: error
      });
    }
  }

  public getGraphStats(): GraphStats {
    const aliasCount = this.stmtCountAliases.get()?.count ?? 0;
    const edgeCount = this.stmtCountEdges.get()?.count ?? 0;
    const resolvedEdgeCount = this.stmtCountResolvedEdges.get()?.count ?? 0;
    const entityMentionCount = this.stmtCountEntityMentions.get()?.count ?? 0;
    return {
      aliasCount,
      edgeCount,
      resolvedEdgeCount,
      entityMentionCount
    };
  }

  public getStats(): KnowledgeStoreStats {
    const documentCount = this.stmtCountDocs.get()?.count ?? 0;
    const chunkCount = this.stmtCountChunks.get()?.count ?? 0;
    const sourceCount = this.stmtCountSources.get()?.count ?? 0;
    return { documentCount, chunkCount, sourceCount };
  }

  public close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    this.vectorCache = null;
    this.db.close();
  }
}
