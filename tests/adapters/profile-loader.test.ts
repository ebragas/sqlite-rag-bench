import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadRunProfile } from "../../src/adapters/profile-loader.js";
import { ConfigError } from "../../src/models/errors.js";
import { computeIndexConfigHash } from "../../src/models/profile.js";

const tempDirs: string[] = [];

function createTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "profile-loader-test-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("loadRunProfile", () => {
  it("loads eval/profiles/l0-fts.json by profile name and explicit path", () => {
    const byName = loadRunProfile("l0-fts");
    const byPath = loadRunProfile("eval/profiles/l0-fts.json");

    expect(byName).toEqual({
      name: "l0-fts",
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
    expect(byPath).toEqual(byName);
  });

  it("loads eval/profiles/l1-filters.json with bounded L1 agent tools", () => {
    const loaded = loadRunProfile("l1-filters");

    expect(loaded.name).toBe("l1-filters");
    expect(loaded.enableEmbeddings).toBe(false);
    expect(loaded.preSearchEnabled).toBe(false);
    expect(loaded.enabledTools).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata"
    ]);
    expect(loaded.topK).toBe(15);
    expect(loaded.maxAgentTurns).toBe(6);
    expect(loaded.maxFetchedDocs).toBe(15);
    expect(loaded.softDeadlineMs).toBe(90000);
    expect(loaded.routingModel).toBe("gemini-3.5-flash-lite");
    expect(loaded.synthesisModel).toBe("gemini-3.8-flash");
  });

  it("loads eval/profiles/l2-hybrid.json and produces a distinct index config hash vs l1-filters", () => {
    const l1 = loadRunProfile("l1-filters");
    const l2 = loadRunProfile("l2-hybrid");

    expect(l2.name).toBe("l2-hybrid");
    expect(l2.enableEmbeddings).toBe(true);
    expect(l2.embeddingModel).toBe("gemini-embedding-001");
    expect(l2.embeddingDims).toBe(768);
    expect(l2.preSearchEnabled).toBe(false);
    expect(l2.enabledTools).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata"
    ]);
    expect(l2.topK).toBe(15);
    expect(l2.maxAgentTurns).toBe(6);
    expect(l2.maxFetchedDocs).toBe(15);
    expect(l2.softDeadlineMs).toBe(90000);
    expect(l2.routingModel).toBe("gemini-3.5-flash-lite");
    expect(l2.synthesisModel).toBe("gemini-3.8-flash");
    expect(computeIndexConfigHash(l2)).not.toBe(computeIndexConfigHash(l1));
  });

  it("loads eval/profiles/l3-graph.json with expand_graph enabled and shares l2 index hash", () => {
    const l2 = loadRunProfile("l2-hybrid");
    const l3 = loadRunProfile("l3-graph");

    expect(l3.name).toBe("l3-graph");
    expect(l3.enableEmbeddings).toBe(true);
    expect(l3.embeddingModel).toBe("gemini-embedding-001");
    expect(l3.embeddingDims).toBe(768);
    expect(l3.preSearchEnabled).toBe(false);
    expect(l3.enabledTools).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata",
      "expand_graph"
    ]);
    expect(l3.topK).toBe(15);
    expect(l3.maxAgentTurns).toBe(6);
    expect(l3.maxFetchedDocs).toBe(15);
    expect(l3.softDeadlineMs).toBe(90000);
    expect(l3.routingModel).toBe("gemini-3.5-flash-lite");
    expect(l3.synthesisModel).toBe("gemini-3.8-flash");
    expect(computeIndexConfigHash(l3)).toBe(computeIndexConfigHash(l2));
  });

  it("loads eval/profiles/prod.json with promoted hybrid + graph + pre-search configuration", () => {
    const prod = loadRunProfile("prod");

    expect(prod.name).toBe("prod");
    expect(prod.enableEmbeddings).toBe(true);
    expect(prod.embeddingModel).toBe("gemini-embedding-2");
    expect(prod.embeddingDims).toBe(768);
    expect(prod.preSearchEnabled).toBe(true);
    expect(prod.topK).toBe(20);
    expect(prod.enabledTools).toEqual([
      "search",
      "fetch_docs",
      "filter_metadata",
      "expand_graph"
    ]);
    expect(prod.maxAgentTurns).toBe(5);
    expect(prod.softDeadlineMs).toBe(42000);
    expect(prod.routingModel).toBe("gemini-3.8-flash");
    expect(prod.synthesisModel).toBe("gemini-3.8-flash");
  });

  it("applies CLI overrides onto a loaded profile", () => {
    const loaded = loadRunProfile("l0-fts", [
      "topK=20",
      "preSearchEnabled=true",
      'enabledTools=["search","fetch_docs"]',
      "chunkStrategy.chunkTargetChars=3200"
    ]);

    expect(loaded.topK).toBe(20);
    expect(loaded.preSearchEnabled).toBe(true);
    expect(loaded.enabledTools).toEqual(["search", "fetch_docs"]);
    expect(loaded.chunkStrategy.chunkTargetChars).toBe(3200);
    expect(loaded.chunkStrategy.chunkOverlapChars).toBe(600);
  });

  it("throws ConfigError on missing files, malformed JSON, or invalid schema", () => {
    const dir = createTempDir();

    expect(() => loadRunProfile("", [], dir)).toThrow(ConfigError);
    expect(() => loadRunProfile("does-not-exist", [], dir)).toThrow(ConfigError);

    const malformedPath = join(dir, "malformed.json");
    writeFileSync(malformedPath, "{ not valid json", "utf8");
    expect(() => loadRunProfile(malformedPath)).toThrow(ConfigError);

    const invalidSchemaPath = join(dir, "invalid-schema.json");
    writeFileSync(invalidSchemaPath, JSON.stringify({ name: "broken", topK: -5 }), "utf8");
    expect(() => loadRunProfile(invalidSchemaPath)).toThrow(ConfigError);
  });
});
