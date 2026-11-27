import { describe, expect, it } from "vitest";
import { ConfigError } from "../../src/models/errors.js";
import {
  applyProfileOverrides,
  computeIndexConfigHash,
  parseOverrideAssignments,
  RunProfileSchema
} from "../../src/models/profile.js";
import { sha256Hex } from "../../src/utils/hash.js";

describe("RunProfileSchema", () => {
  it("applies defaults for all index-time and query-time knobs", () => {
    const profile = RunProfileSchema.parse({ name: "baseline" });

    expect(profile).toEqual({
      name: "baseline",
      chunkStrategy: {
        shortDocMaxChars: 6000,
        chunkTargetChars: 4000,
        chunkOverlapChars: 600,
        includeHeaderCard: true
      },
      enableEmbeddings: false,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: 768,
      topK: 15,
      preSearchEnabled: false,
      enabledTools: [],
      maxAgentTurns: 6,
      maxFetchedDocs: 15,
      softDeadlineMs: 90000,
      routingModel: "gemini-3.5-flash-lite",
      synthesisModel: "gemini-3.8-flash",
      enableAbstentionGate: false,
      enableSecondPassPruner: false
    });
  });

  it("rejects invalid bounds, unknown tools, and unknown fields", () => {
    expect(RunProfileSchema.safeParse({ name: "" }).success).toBe(false);
    expect(RunProfileSchema.safeParse({ name: "bad", topK: 0 }).success).toBe(false);
    expect(RunProfileSchema.safeParse({ name: "bad", embeddingDims: 768.5 }).success).toBe(false);
    expect(
      RunProfileSchema.safeParse({
        name: "bad",
        chunkStrategy: {
          shortDocMaxChars: 6000,
          chunkTargetChars: 1000,
          chunkOverlapChars: 1000,
          includeHeaderCard: true
        }
      }).success
    ).toBe(false);
    expect(
      RunProfileSchema.safeParse({ name: "bad", enabledTools: ["raw_sql"] }).success
    ).toBe(false);
    expect(
      RunProfileSchema.safeParse({ name: "bad", extraKnob: true }).success
    ).toBe(false);
  });
});

describe("computeIndexConfigHash", () => {
  it("hashes only index-time knobs deterministically", () => {
    const base = RunProfileSchema.parse({ name: "l0-fts" });
    const baseHash = computeIndexConfigHash(base);

    expect(baseHash).toMatch(/^[a-f0-9]{64}$/);
    expect(sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    const queryVariant = RunProfileSchema.parse({
      name: "l1-tools",
      topK: 30,
      preSearchEnabled: true,
      enabledTools: ["search", "fetch_docs", "filter_metadata", "expand_graph"],
      maxAgentTurns: 10,
      maxFetchedDocs: 25,
      softDeadlineMs: 45000,
      routingModel: "gemini-2.5-flash-lite",
      synthesisModel: "gemini-2.5-pro",
      enableAbstentionGate: true,
      enableSecondPassPruner: true
    });
    expect(computeIndexConfigHash(queryVariant)).toBe(baseHash);

    const withEmbeddings = RunProfileSchema.parse({
      name: "l2-hybrid",
      enableEmbeddings: true
    });
    expect(computeIndexConfigHash(withEmbeddings)).not.toBe(baseHash);

    const withDifferentChunkTarget = RunProfileSchema.parse({
      name: "l0-chunk-3500",
      chunkStrategy: {
        shortDocMaxChars: 6000,
        chunkTargetChars: 3500,
        chunkOverlapChars: 600,
        includeHeaderCard: true
      }
    });
    expect(computeIndexConfigHash(withDifferentChunkTarget)).not.toBe(baseHash);

    const withDifferentDims = RunProfileSchema.parse({
      name: "l2-1536",
      embeddingDims: 1536
    });
    expect(computeIndexConfigHash(withDifferentDims)).not.toBe(baseHash);
  });
});

describe("applyProfileOverrides", () => {
  it("parses and applies dot-paths, booleans, numbers, JSON arrays, and strings", () => {
    const base = RunProfileSchema.parse({ name: "l0-fts" });
    const parsed = parseOverrideAssignments([
      "chunkStrategy.chunkTargetChars=3500",
      "chunkStrategy.includeHeaderCard=false",
      "enableEmbeddings=true",
      "topK=25",
      'enabledTools=["search","fetch_docs"]',
      "synthesisModel=gemini-2.5-pro"
    ]);

    expect(parsed).toEqual({
      chunkStrategy: {
        chunkTargetChars: 3500,
        includeHeaderCard: false
      },
      enableEmbeddings: true,
      topK: 25,
      enabledTools: ["search", "fetch_docs"],
      synthesisModel: "gemini-2.5-pro"
    });

    const updated = applyProfileOverrides(base, [
      "chunkStrategy.chunkTargetChars=3500",
      "chunkStrategy.includeHeaderCard=false",
      "enableEmbeddings=true",
      "topK=25",
      'enabledTools=["search","fetch_docs"]',
      "synthesisModel=gemini-2.5-pro"
    ]);

    expect(updated.chunkStrategy).toEqual({
      shortDocMaxChars: 6000,
      chunkTargetChars: 3500,
      chunkOverlapChars: 600,
      includeHeaderCard: false
    });
    expect(updated.enableEmbeddings).toBe(true);
    expect(updated.topK).toBe(25);
    expect(updated.enabledTools).toEqual(["search", "fetch_docs"]);
    expect(updated.synthesisModel).toBe("gemini-2.5-pro");
  });

  it("throws ConfigError on unknown keys, malformed entries, and schema violations", () => {
    const base = RunProfileSchema.parse({ name: "l0-fts" });

    expect(() => applyProfileOverrides(base, ["unknownKey=1"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["chunkStrategy.unknownField=1"])).toThrow(
      ConfigError
    );
    expect(() => applyProfileOverrides(base, ["topK.nested=1"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["missingEquals"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["=noKey"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["chunkStrategy..chunkTargetChars=3000"])).toThrow(
      ConfigError
    );
    expect(() => applyProfileOverrides(base, ["__proto__.polluted=true"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["enabledTools=[not-json"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["topK=0"])).toThrow(ConfigError);
    expect(() => applyProfileOverrides(base, ["chunkStrategy.chunkOverlapChars=5000"])).toThrow(
      ConfigError
    );
    expect(() => applyProfileOverrides(base, ['enabledTools=["raw_sql"]'])).toThrow(ConfigError);
    expect(() =>
      parseOverrideAssignments(["chunkStrategy=1", "chunkStrategy.chunkTargetChars=3000"])
    ).toThrow(ConfigError);
  });
});
