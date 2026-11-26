import { describe, expect, it } from "vitest";
import {
  bufferToFloat32,
  dotProduct,
  float32ToBuffer,
  fuseReciprocalRank,
  l2Normalize,
  type RankedHitInput
} from "../../src/utils/vector.js";

function makeHit(
  chunkId: number,
  datasetDocUuid: string,
  score: number,
  excerpt = "snippet"
): RankedHitInput {
  return {
    chunkId,
    datasetDocUuid,
    chunkIndex: 0,
    source: "confluence",
    title: `Doc ${datasetDocUuid}`,
    score,
    excerpt
  };
}

describe("vector math utilities", () => {
  it("normalizes vectors to unit L2 length and handles zero vectors without NaN", () => {
    const normalized = l2Normalize(new Float32Array([3, 4, 0]));
    expect(normalized[0]).toBeCloseTo(0.6, 5);
    expect(normalized[1]).toBeCloseTo(0.8, 5);
    expect(normalized[2]).toBeCloseTo(0, 5);

    const zero = l2Normalize(new Float32Array([0, 0, 0]));
    expect(Array.from(zero)).toEqual([0, 0, 0]);
    expect(zero.every((value) => Number.isFinite(value))).toBe(true);

    const empty = l2Normalize(new Float32Array(0));
    expect(empty.length).toBe(0);
  });

  it("computes exact dot products on contiguous buffer offsets and validates bounds", () => {
    const v0 = l2Normalize(new Float32Array([1, 0, 0]));
    const v1 = l2Normalize(new Float32Array([0, 3, 4]));
    const packed = new Float32Array(6);
    packed.set(v0, 0);
    packed.set(v1, 3);

    const query = l2Normalize(new Float32Array([0, 3, 4]));
    expect(dotProduct(packed, 0, query, 3)).toBeCloseTo(0, 5);
    expect(dotProduct(packed, 3, query, 3)).toBeCloseTo(1, 5);

    expect(() => dotProduct(packed, 4, query, 3)).toThrow(RangeError);
    expect(() => dotProduct(packed, -1, query, 3)).toThrow(RangeError);
    expect(() => dotProduct(packed, 0, new Float32Array([1, 2]), 3)).toThrow(
      RangeError
    );
  });

  it("roundtrips Float32Array through Buffer including unaligned slices and validates byte length", () => {
    const original = new Float32Array([-1.25, 0, 0.5, 42.125]);
    const encoded = float32ToBuffer(original);
    expect(encoded.byteLength).toBe(16);

    const decoded = bufferToFloat32(encoded, 4);
    expect(Array.from(decoded)).toEqual(Array.from(original));

    // Verify unaligned Buffer view (offset 1) decodes cleanly without alignment errors.
    const padded = Buffer.alloc(17);
    encoded.copy(padded, 1);
    const unalignedSlice = padded.subarray(1, 17);
    const decodedUnaligned = bufferToFloat32(unalignedSlice, 4);
    expect(Array.from(decodedUnaligned)).toEqual(Array.from(original));

    expect(() => bufferToFloat32(encoded, 3)).toThrow(RangeError);
    expect(() => bufferToFloat32(encoded, 5)).toThrow(RangeError);
    expect(() => bufferToFloat32(encoded, 0)).toThrow(RangeError);
  });

  it("fuses FTS5 and dense rankings with deterministic RRF order and exact provenance", () => {
    const ftsHits: RankedHitInput[] = [
      makeHit(10, "doc_a", -5.2, "fts [alpha]"),
      makeHit(20, "doc_b", -3.1, "fts [beta]"),
      makeHit(30, "doc_c", -1.4, "fts [gamma]")
    ];
    const denseHits: RankedHitInput[] = [
      makeHit(40, "doc_d", 0.94, "dense delta"),
      makeHit(20, "doc_b", 0.91, "dense beta"),
      makeHit(50, "doc_e", 0.85, "dense epsilon")
    ];

    const fused = fuseReciprocalRank(ftsHits, denseHits, 60, 4);
    expect(fused).toHaveLength(4);

    // doc_b appears at rank 2 in both lists: 2 / 62 = 0.032258 > 1 / 61 = 0.016393.
    expect(fused[0]?.chunkId).toBe(20);
    expect(fused[0]?.datasetDocUuid).toBe("doc_b");
    expect(fused[0]?.ftsRank).toBe(2);
    expect(fused[0]?.ftsScore).toBe(-3.1);
    expect(fused[0]?.denseRank).toBe(2);
    expect(fused[0]?.denseScore).toBe(0.91);
    expect(fused[0]?.rrfScore).toBeCloseTo(2 / 62, 8);
    expect(fused[0]?.excerpt).toBe("fts [beta]");

    // doc_a (ftsRank 1) and doc_d (denseRank 1) tie at 1/61; doc_a sorts before doc_d deterministically.
    expect(fused[1]?.datasetDocUuid).toBe("doc_a");
    expect(fused[1]?.ftsRank).toBe(1);
    expect(fused[1]?.denseRank).toBeNull();
    expect(fused[1]?.denseScore).toBeNull();

    expect(fused[2]?.datasetDocUuid).toBe("doc_d");
    expect(fused[2]?.ftsRank).toBeNull();
    expect(fused[2]?.denseRank).toBe(1);
    expect(fused[2]?.denseScore).toBe(0.94);

    expect(fused[3]?.datasetDocUuid).toBe("doc_c");
    expect(fuseReciprocalRank(ftsHits, denseHits, 60, 0)).toEqual([]);
    expect(() => fuseReciprocalRank(ftsHits, denseHits, 0)).toThrow(RangeError);
  });
});
