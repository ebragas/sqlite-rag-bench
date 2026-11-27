import { z } from "zod";
import {
  ENTITY_TYPES,
  GRAPH_EDGE_TYPES,
  type DocAliasRecord,
  type EntityMentionRecord,
  type EntityType,
  type ExtractedGraphArtifacts,
  type GraphEdgeType,
  type RawGraphRef
} from "../utils/graph.js";
import { AppError } from "./errors.js";

export type {
  DocAliasRecord,
  EntityMentionRecord,
  EntityType,
  ExtractedGraphArtifacts,
  GraphEdgeType,
  RawGraphRef
};

export class IngestionError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("INGESTION_ERROR", message, options);
  }
}

export class RetrievalError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("RETRIEVAL_ERROR", message, options);
  }
}

export class ProviderError extends AppError {
  public constructor(message: string, options?: ErrorOptions) {
    super("PROVIDER_ERROR", message, options);
  }
}

export const RawCorpusDocumentSchema = z
  .object({
    dataset_doc_uuid: z.string().min(1),
    title_field_name: z.string().min(1).optional(),
    content_field_names: z.array(z.string().min(1)).optional()
  })
  .passthrough();

export type RawCorpusDocument = z.infer<typeof RawCorpusDocumentSchema>;

export interface NormalizedDocument {
  datasetDocUuid: string;
  contentHash: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  metadataJson: string;
  metadataText: string;
  fullText: string;
}

export interface StoredDocument {
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
}

export interface ChunkRecord {
  datasetDocUuid: string;
  contentHash: string;
  chunkIndex: number;
  title: string;
  metadataText: string;
  body: string;
}

export interface SchemaCatalogEntry {
  source: string;
  docCount: number;
  metadataKeys: string[];
  sampleValues: Record<string, string[]>;
}

export interface SchemaCatalog {
  sources: SchemaCatalogEntry[];
}

export interface FtsSearchHit {
  chunkId: number;
  datasetDocUuid: string;
  chunkIndex: number;
  source: string;
  title: string;
  score: number;
  excerpt: string;
}

export interface ChunkForEmbedding {
  chunkId: number;
  datasetDocUuid: string;
  contentHash: string;
  chunkIndex: number;
  title: string;
  metadataText: string;
  body: string;
}

export interface ChunkEmbeddingRecord {
  chunkId: number;
  datasetDocUuid: string;
  contentHash: string;
  model: string;
  dims: number;
  vector: Float32Array;
}

export interface DenseSearchHit {
  chunkId: number;
  datasetDocUuid: string;
  chunkIndex: number;
  source: string;
  title: string;
  score: number;
  excerpt: string;
}

export interface VectorCacheStats {
  loaded: boolean;
  model: string | null;
  dims: number;
  vectorCount: number;
  byteLength: number;
}

export const METADATA_KEY_PATH_PATTERN = /^[A-Za-z0-9_.-]+$/;

export const MetadataJsonFilterSchema = z
  .object({
    key: z
      .string()
      .min(1)
      .regex(
        METADATA_KEY_PATH_PATTERN,
        "Metadata key path must match /^[A-Za-z0-9_.-]+$/"
      )
      .refine(
        (value) =>
          !value.startsWith(".") &&
          !value.endsWith(".") &&
          !value.includes(".."),
        "Metadata key path must not contain empty segments"
      ),
    op: z.enum(["eq", "contains", "in"]).default("eq"),
    value: z.string().optional(),
    values: z.array(z.string().min(1)).min(1).optional()
  })
  .refine(
    (filter) =>
      filter.op === "in"
        ? filter.values !== undefined && filter.values.length > 0
        : filter.value !== undefined,
    {
      message:
        "Provide values for 'in' filters and value for 'eq' or 'contains' filters"
    }
  );

export type MetadataJsonFilter = z.infer<typeof MetadataJsonFilterSchema>;

export const MetadataFilterOrderBySchema = z.enum([
  "updated_at_desc",
  "updated_at_asc",
  "created_at_desc",
  "created_at_asc",
  "title_asc"
]);

export type MetadataFilterOrderBy = z.infer<typeof MetadataFilterOrderBySchema>;

export const MetadataFilterQuerySchema = z.object({
  source: z.string().min(1).optional(),
  author: z.string().min(1).optional(),
  status: z.string().min(1).optional(),
  createdAfter: z.string().min(1).optional(),
  createdBefore: z.string().min(1).optional(),
  updatedAfter: z.string().min(1).optional(),
  updatedBefore: z.string().min(1).optional(),
  titleContains: z.string().min(1).optional(),
  metadataFilters: z.array(MetadataJsonFilterSchema).optional(),
  limit: z.number().int().min(1).max(100).default(25),
  orderBy: MetadataFilterOrderBySchema.default("updated_at_desc")
});

export type MetadataFilterQuery = z.infer<typeof MetadataFilterQuerySchema>;
export type MetadataFilterQueryInput = z.input<typeof MetadataFilterQuerySchema>;

export interface MetadataFilterRow {
  datasetDocUuid: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface MetadataFilterResult {
  totalCount: number;
  documents: readonly MetadataFilterRow[];
}

export const EntityTypeSchema = z.enum(ENTITY_TYPES);

export const GraphEdgeTypeSchema = z.enum(GRAPH_EDGE_TYPES);

export const ExpandGraphQuerySchema = z
  .object({
    docId: z
      .string()
      .trim()
      .min(1)
      .optional()
      .describe("Seed dataset_doc_uuid to expand neighbors from"),
    entityName: z
      .string()
      .trim()
      .min(1)
      .optional()
      .describe("Canonical entity name to find connected documents for"),
    entityType: EntityTypeSchema.optional().describe(
      "Optional entity type filter when querying by entityName"
    ),
    maxHops: z
      .union([z.literal(1), z.literal(2)])
      .default(1)
      .describe("Maximum graph traversal hops (1 or 2)"),
    limit: z
      .number()
      .int()
      .min(1)
      .max(50)
      .default(15)
      .describe("Maximum number of connected neighbor documents to return")
  })
  .refine(
    (query) => query.docId !== undefined || query.entityName !== undefined,
    {
      message: "Provide at least one of docId or entityName"
    }
  );

export type ExpandGraphQuery = z.infer<typeof ExpandGraphQuerySchema>;
export type ExpandGraphQueryInput = z.input<typeof ExpandGraphQuerySchema>;

export interface ExpandGraphNeighbor {
  datasetDocUuid: string;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  updatedAt: string | null;
  hop: number;
  score: number;
  reasons: string[];
}

export interface GraphStats {
  aliasCount: number;
  edgeCount: number;
  resolvedEdgeCount: number;
  entityMentionCount: number;
}

export const AskRequestSchema = z.object({
  question: z.string().trim().min(1)
});

export type AskRequest = z.infer<typeof AskRequestSchema>;

export const AskResponseSchema = z.object({
  answer: z.string().min(1),
  document_ids: z.array(z.string().min(1))
});

export type AskResponse = z.infer<typeof AskResponseSchema>;

export const AgentSynthesisSchema = z.object({
  answer: z.string().min(1),
  abstained: z.boolean(),
  supporting_doc_ids: z.array(z.string())
});

export type AgentSynthesis = z.infer<typeof AgentSynthesisSchema>;

export type ServerPhase =
  | { readonly kind: "booting" }
  | { readonly kind: "ingesting" }
  | { readonly kind: "ready" }
  | { readonly kind: "draining"; readonly startedAtMs: number }
  | { readonly kind: "stopped" }
  | { readonly kind: "failed"; readonly reason: string };

export type ServerPhaseKind = ServerPhase["kind"];
