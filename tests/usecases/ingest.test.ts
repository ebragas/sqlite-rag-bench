import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createFilesystemCorpusReader } from "../../src/adapters/corpus-reader.js";
import { KnowledgeStore } from "../../src/adapters/knowledge-store.js";
import { IngestionError } from "../../src/models/corpus.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import {
  ingestCorpus,
  resolveCachedDbPath
} from "../../src/usecases/ingest.js";

const tempDirs: string[] = [];

function createTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function writeJson(filePath: string, payload: unknown): void {
  mkdirSync(join(filePath, ".."), { recursive: true });
  writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf8");
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("ingestCorpus and KnowledgeStore", () => {
  const profile = RunProfileSchema.parse({ name: "l0-fts" });

  it("ingests nested sources incrementally, updates hashes atomically, deletes stale docs, and populates schema catalog", async () => {
    const root = createTempDir("srb-ingest-");
    const corpusDir = join(root, "corpus");
    const dbPath = join(root, "index", "brain.sqlite");

    const confPath = join(corpusDir, "confluence", "infra", "postgres.json");
    const slackPath = join(corpusDir, "slack", "threads", "cutover.json");
    const jiraPath = join(corpusDir, "jira", "ops", "inc-101.json");

    writeJson(confPath, {
      dataset_doc_uuid: "dsid_conf_pg",
      title_field_name: "title",
      content_field_names: ["body"],
      title: "Postgres 16 Migration Runbook",
      author: "dana",
      status: "approved",
      owner_team: "data-platform",
      body: "Billing cluster moved to Postgres 16 with checksum verification."
    });
    writeJson(slackPath, {
      dataset_doc_uuid: "dsid_slack_cutover",
      title_field_name: "channel",
      content_field_names: ["messages"],
      channel: "#db-ops",
      status: "active",
      messages: [
        { user: "dana", text: "Replica lag stayed below 20ms during cutover." }
      ]
    });
    writeJson(jiraPath, {
      dataset_doc_uuid: "dsid_jira_inc101",
      title_field_name: "summary",
      content_field_names: ["description"],
      summary: "Alert noise on legacy MySQL monitor",
      priority: "P2",
      description: "Decommission obsolete MySQL pager alert after migration."
    });
    writeFileSync(join(corpusDir, "README.txt"), "ignored file", "utf8");

    const store = new KnowledgeStore(dbPath);
    try {
      const reader = createFilesystemCorpusReader(corpusDir);

      const first = await ingestCorpus(reader, store, profile);
      expect(first).toEqual({
        total: 3,
        inserted: 3,
        updated: 0,
        unchanged: 0,
        deleted: 0,
        sourceCount: 3,
        embeddedChunks: 0,
        embeddingInputTokens: 0
      });
      expect(store.getStats()).toEqual({
        documentCount: 3,
        chunkCount: 3,
        sourceCount: 3
      });

      const hits = store.searchFts("Postgres checksum verification", 5);
      expect(hits.length).toBeGreaterThan(0);
      expect(hits[0]?.datasetDocUuid).toBe("dsid_conf_pg");
      expect(hits[0]?.source).toBe("confluence");
      expect(hits[0]?.excerpt).toContain("[");

      const catalog = store.getSchemaCatalog();
      expect(catalog.sources.map((entry) => entry.source)).toEqual([
        "confluence",
        "jira",
        "slack"
      ]);
      const confluenceEntry = catalog.sources.find(
        (entry) => entry.source === "confluence"
      );
      expect(confluenceEntry?.docCount).toBe(1);
      expect(confluenceEntry?.metadataKeys).toContain("owner_team");
      expect(confluenceEntry?.sampleValues.owner_team).toEqual(["data-platform"]);

      const second = await ingestCorpus(reader, store, profile);
      expect(second).toEqual({
        total: 3,
        inserted: 0,
        updated: 0,
        unchanged: 3,
        deleted: 0,
        sourceCount: 3,
        embeddedChunks: 0,
        embeddingInputTokens: 0
      });

      writeJson(confPath, {
        dataset_doc_uuid: "dsid_conf_pg",
        title_field_name: "title",
        content_field_names: ["body"],
        title: "Postgres 17 Upgrade Runbook",
        author: "dana",
        status: "approved",
        owner_team: "data-platform",
        body: "Billing cluster upgraded to Postgres 17 using logical replication."
      });
      unlinkSync(jiraPath);

      const third = await ingestCorpus(reader, store, profile);
      expect(third).toEqual({
        total: 2,
        inserted: 0,
        updated: 1,
        unchanged: 1,
        deleted: 1,
        sourceCount: 2,
        embeddedChunks: 0,
        embeddingInputTokens: 0
      });

      expect(store.hasDocumentId("dsid_jira_inc101")).toBe(false);
      expect(store.searchFts("obsolete MySQL pager", 5)).toEqual([]);
      expect(store.searchFts("checksum verification", 5)).toEqual([]);

      const updatedHits = store.searchFts("logical replication", 5);
      expect(updatedHits.map((hit) => hit.datasetDocUuid)).toEqual([
        "dsid_conf_pg"
      ]);

      const fetched = store.fetchDocs([
        "dsid_slack_cutover",
        "dsid_conf_pg",
        "dsid_missing"
      ]);
      expect(fetched.map((doc) => doc.datasetDocUuid)).toEqual([
        "dsid_slack_cutover",
        "dsid_conf_pg"
      ]);
    } finally {
      store.close();
    }
  });

  it("raises IngestionError on corrupt JSON, duplicate UUIDs, and aborted signals", async () => {
    const root = createTempDir("srb-ingest-err-");
    const corruptDir = join(root, "corrupt");
    mkdirSync(join(corruptDir, "jira"), { recursive: true });
    writeFileSync(join(corruptDir, "jira", "broken.json"), "{not-json", "utf8");

    const store = new KnowledgeStore(":memory:");
    try {
      await expect(
        ingestCorpus(createFilesystemCorpusReader(corruptDir), store, profile)
      ).rejects.toThrow(IngestionError);

      const dupDir = join(root, "dup");
      writeJson(join(dupDir, "slack", "a.json"), {
        dataset_doc_uuid: "dsid_dup",
        text: "first"
      });
      writeJson(join(dupDir, "slack", "b.json"), {
        dataset_doc_uuid: "dsid_dup",
        text: "second"
      });

      await expect(
        ingestCorpus(createFilesystemCorpusReader(dupDir), store, profile)
      ).rejects.toThrow(IngestionError);

      const abortController = new AbortController();
      abortController.abort(new Error("stop"));
      await expect(
        ingestCorpus(
          createFilesystemCorpusReader(dupDir),
          store,
          profile,
          abortController.signal
        )
      ).rejects.toThrow(IngestionError);
    } finally {
      store.close();
    }
  });

  it("derives distinct cached DB paths for index config changes while reusing paths for query-only sweeps", () => {
    const base = RunProfileSchema.parse({ name: "base" });
    const querySweep = RunProfileSchema.parse({
      name: "sweep",
      topK: 30,
      preSearchEnabled: true,
      enabledTools: ["search", "fetch_docs"]
    });
    const chunkChange = RunProfileSchema.parse({
      name: "chunk-change",
      chunkStrategy: {
        shortDocMaxChars: 6000,
        chunkTargetChars: 3000,
        chunkOverlapChars: 400,
        includeHeaderCard: true
      }
    });
    const embedChange = RunProfileSchema.parse({
      name: "embed-change",
      enableEmbeddings: true,
      embeddingDims: 768
    });

    const basePath = resolveCachedDbPath("/tmp/cache", "/data/corpus-a", base);
    const sweepPath = resolveCachedDbPath("/tmp/cache", "/data/corpus-a", querySweep);
    const chunkPath = resolveCachedDbPath("/tmp/cache", "/data/corpus-a", chunkChange);
    const embedPath = resolveCachedDbPath("/tmp/cache", "/data/corpus-a", embedChange);
    const otherCorpusPath = resolveCachedDbPath("/tmp/cache", "/data/corpus-b", base);

    expect(sweepPath).toBe(basePath);
    expect(chunkPath).not.toBe(basePath);
    expect(embedPath).not.toBe(basePath);
    expect(otherCorpusPath).not.toBe(basePath);
  });
});
