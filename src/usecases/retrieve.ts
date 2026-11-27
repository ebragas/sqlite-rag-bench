import type { Embedder } from "../adapters/embedder.js";
import type { KnowledgeStore } from "../adapters/knowledge-store.js";
import { RetrievalError } from "../models/corpus.js";
import type { RunProfile } from "../models/profile.js";
import {
  fuseReciprocalRank,
  type FusedSearchHit
} from "../utils/vector.js";

export interface RetrieveEvidenceOptions {
  readonly embedder?: Embedder;
  readonly topK?: number;
  readonly rrfK?: number;
  readonly signal?: AbortSignal;
}

export interface RetrieveEvidenceResult {
  readonly mode: "fts" | "hybrid";
  readonly hits: readonly FusedSearchHit[];
  readonly ftsHitCount: number;
  readonly denseHitCount: number;
  readonly embeddingInputTokens: number;
}

const DEFAULT_RRF_K = 60;
const MIN_HYBRID_CANDIDATE_K = 30;
const MAX_HYBRID_CANDIDATE_K = 100;

export async function retrieveEvidence(
  store: KnowledgeStore,
  profile: RunProfile,
  query: string,
  options?: RetrieveEvidenceOptions
): Promise<RetrieveEvidenceResult> {
  const topK = options?.topK ?? profile.topK;
  const rrfK = options?.rrfK ?? DEFAULT_RRF_K;
  const trimmedQuery = query.trim();
  const useHybrid = profile.enableEmbeddings && options?.embedder !== undefined;

  if (topK <= 0 || trimmedQuery.length === 0) {
    return {
      mode: useHybrid ? "hybrid" : "fts",
      hits: [],
      ftsHitCount: 0,
      denseHitCount: 0,
      embeddingInputTokens: 0
    };
  }

  if (!useHybrid || options?.embedder === undefined) {
    const ftsHits = store.searchFts(trimmedQuery, topK);
    const hits = fuseReciprocalRank(ftsHits, [], rrfK, topK);
    return {
      mode: "fts",
      hits,
      ftsHitCount: ftsHits.length,
      denseHitCount: 0,
      embeddingInputTokens: 0
    };
  }

  const candidateK = Math.max(
    topK,
    Math.min(Math.max(topK * 2, MIN_HYBRID_CANDIDATE_K), MAX_HYBRID_CANDIDATE_K)
  );
  const ftsHits = store.searchFts(trimmedQuery, candidateK);
  const embedded = await options.embedder.embedBatch(
    [trimmedQuery],
    options.signal
  );
  const queryVector = embedded.vectors[0];
  if (embedded.vectors.length !== 1 || queryVector === undefined) {
    throw new RetrievalError(
      `Embedder returned ${embedded.vectors.length} vectors for single query`
    );
  }

  const denseHits = store.searchDense(queryVector, candidateK);
  const hits = fuseReciprocalRank(ftsHits, denseHits, rrfK, topK);

  return {
    mode: "hybrid",
    hits,
    ftsHitCount: ftsHits.length,
    denseHitCount: denseHits.length,
    embeddingInputTokens: embedded.inputTokens
  };
}
