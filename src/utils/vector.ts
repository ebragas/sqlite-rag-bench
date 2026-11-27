export interface RankedHitInput {
  readonly chunkId: number;
  readonly datasetDocUuid: string;
  readonly chunkIndex: number;
  readonly source: string;
  readonly title: string;
  readonly score: number;
  readonly excerpt: string;
}

export interface FusedSearchHit {
  readonly chunkId: number;
  readonly datasetDocUuid: string;
  readonly chunkIndex: number;
  readonly source: string;
  readonly title: string;
  readonly excerpt: string;
  readonly rrfScore: number;
  readonly ftsRank: number | null;
  readonly ftsScore: number | null;
  readonly denseRank: number | null;
  readonly denseScore: number | null;
}

export function l2Normalize(vec: Float32Array): Float32Array {
  const out = new Float32Array(vec.length);
  let sumSq = 0;

  for (let i = 0; i < vec.length; i += 1) {
    const val = vec[i] ?? 0;
    sumSq += val * val;
  }

  const norm = Math.sqrt(sumSq);
  if (norm === 0 || !Number.isFinite(norm)) {
    return out;
  }

  const invNorm = 1 / norm;
  for (let i = 0; i < vec.length; i += 1) {
    out[i] = (vec[i] ?? 0) * invNorm;
  }
  return out;
}

export function dotProduct(
  a: Float32Array,
  aOffset: number,
  b: Float32Array,
  dims: number
): number {
  if (!Number.isInteger(dims) || dims < 0) {
    throw new RangeError(`Invalid vector dimensions: ${dims}`);
  }
  if (!Number.isInteger(aOffset) || aOffset < 0 || aOffset + dims > a.length) {
    throw new RangeError(
      `Vector slice out of bounds: offset=${aOffset}, dims=${dims}, length=${a.length}`
    );
  }
  if (b.length < dims) {
    throw new RangeError(
      `Query vector length ${b.length} is smaller than dims ${dims}`
    );
  }

  let sum = 0;
  for (let i = 0; i < dims; i += 1) {
    sum += (a[aOffset + i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

export function float32ToBuffer(vec: Float32Array): Buffer {
  const view = new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength);
  return Buffer.from(view);
}

export function bufferToFloat32(
  buf: Buffer,
  expectedDims: number
): Float32Array {
  if (!Number.isInteger(expectedDims) || expectedDims <= 0) {
    throw new RangeError(`Expected positive integer dims, got ${expectedDims}`);
  }

  const expectedBytes = expectedDims * Float32Array.BYTES_PER_ELEMENT;
  if (buf.byteLength !== expectedBytes) {
    throw new RangeError(
      `Embedding buffer byte length mismatch: expected ${expectedBytes} bytes for ${expectedDims} dims, got ${buf.byteLength}`
    );
  }

  const out = new Float32Array(expectedDims);
  const dstBytes = new Uint8Array(out.buffer);
  dstBytes.set(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength));
  return out;
}

interface MutableFusedEntry {
  chunkId: number;
  datasetDocUuid: string;
  chunkIndex: number;
  source: string;
  title: string;
  excerpt: string;
  rrfScore: number;
  ftsRank: number | null;
  ftsScore: number | null;
  denseRank: number | null;
  denseScore: number | null;
}

function compareFusedEntries(a: MutableFusedEntry, b: MutableFusedEntry): number {
  if (b.rrfScore !== a.rrfScore) {
    return b.rrfScore - a.rrfScore;
  }

  const aBestRank = Math.min(
    a.ftsRank ?? Number.POSITIVE_INFINITY,
    a.denseRank ?? Number.POSITIVE_INFINITY
  );
  const bBestRank = Math.min(
    b.ftsRank ?? Number.POSITIVE_INFINITY,
    b.denseRank ?? Number.POSITIVE_INFINITY
  );
  if (aBestRank !== bBestRank) {
    return aBestRank - bBestRank;
  }

  if (a.datasetDocUuid !== b.datasetDocUuid) {
    return a.datasetDocUuid < b.datasetDocUuid ? -1 : 1;
  }

  return a.chunkId - b.chunkId;
}

export function fuseReciprocalRank(
  ftsHits: readonly RankedHitInput[],
  denseHits: readonly RankedHitInput[],
  k = 60,
  topK?: number
): FusedSearchHit[] {
  if (!Number.isFinite(k) || k <= 0) {
    throw new RangeError(`RRF constant k must be a positive number, got ${k}`);
  }
  if (topK !== undefined && topK <= 0) {
    return [];
  }

  const byChunkId = new Map<number, MutableFusedEntry>();

  for (let i = 0; i < ftsHits.length; i += 1) {
    const hit = ftsHits[i];
    if (hit === undefined) {
      continue;
    }
    const rank = i + 1;
    const existing = byChunkId.get(hit.chunkId);
    if (existing !== undefined) {
      if (existing.ftsRank !== null) {
        continue;
      }
      existing.ftsRank = rank;
      existing.ftsScore = hit.score;
      existing.rrfScore += 1 / (k + rank);
      if (hit.excerpt.trim().length > 0) {
        existing.excerpt = hit.excerpt;
      }
      continue;
    }

    byChunkId.set(hit.chunkId, {
      chunkId: hit.chunkId,
      datasetDocUuid: hit.datasetDocUuid,
      chunkIndex: hit.chunkIndex,
      source: hit.source,
      title: hit.title,
      excerpt: hit.excerpt,
      rrfScore: 1 / (k + rank),
      ftsRank: rank,
      ftsScore: hit.score,
      denseRank: null,
      denseScore: null
    });
  }

  for (let i = 0; i < denseHits.length; i += 1) {
    const hit = denseHits[i];
    if (hit === undefined) {
      continue;
    }
    const rank = i + 1;
    const existing = byChunkId.get(hit.chunkId);
    if (existing !== undefined) {
      if (existing.denseRank !== null) {
        continue;
      }
      existing.denseRank = rank;
      existing.denseScore = hit.score;
      existing.rrfScore += 1 / (k + rank);
      if (existing.excerpt.trim().length === 0 && hit.excerpt.trim().length > 0) {
        existing.excerpt = hit.excerpt;
      }
      continue;
    }

    byChunkId.set(hit.chunkId, {
      chunkId: hit.chunkId,
      datasetDocUuid: hit.datasetDocUuid,
      chunkIndex: hit.chunkIndex,
      source: hit.source,
      title: hit.title,
      excerpt: hit.excerpt,
      rrfScore: 1 / (k + rank),
      ftsRank: null,
      ftsScore: null,
      denseRank: rank,
      denseScore: hit.score
    });
  }

  const sorted = [...byChunkId.values()].sort(compareFusedEntries);
  const sliced = topK !== undefined ? sorted.slice(0, topK) : sorted;

  return sliced.map((entry) => ({
    chunkId: entry.chunkId,
    datasetDocUuid: entry.datasetDocUuid,
    chunkIndex: entry.chunkIndex,
    source: entry.source,
    title: entry.title,
    excerpt: entry.excerpt,
    rrfScore: entry.rrfScore,
    ftsRank: entry.ftsRank,
    ftsScore: entry.ftsScore,
    denseRank: entry.denseRank,
    denseScore: entry.denseScore
  }));
}
