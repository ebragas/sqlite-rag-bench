import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { afterEach, describe, expect, it } from "vitest";
import { createFilesystemCorpusReader } from "../../src/adapters/corpus-reader.js";
import type { Embedder } from "../../src/adapters/embedder.js";
import { KnowledgeStore } from "../../src/adapters/knowledge-store.js";
import { ValidationError } from "../../src/models/errors.js";
import { RunProfileSchema } from "../../src/models/profile.js";
import { ingestCorpus } from "../../src/usecases/ingest.js";
import { extractGraphArtifacts } from "../../src/utils/graph.js";

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

describe("Link Graph ingestion and multi-hop expandGraph", () => {
  const profile = RunProfileSchema.parse({ name: "l0-fts" });

  it("discovers 1-hop and 2-hop cross-source neighbors, ranks rare entities over noisy labels, and supports entity lookup", async () => {
    const root = createTempDir("srb-graph-");
    const corpusDir = join(root, "corpus");
    const dbPath = join(root, "index", "brain.sqlite");

    // 1. Jira ticket -> explicit PR link to GitHub PR #108, shares noisy label "enterprise"
    writeJson(join(corpusDir, "jira", "eng-42.json"), {
      dataset_doc_uuid: "dsid_jira_eng42",
      title_field_name: "summary",
      content_field_names: ["description"],
      key: "ENG-42",
      summary: "Implement SAML group sync for enterprise customers",
      reporter: "Aisha Khan",
      status: "In Progress",
      updated_at: "2026-03-10",
      labels: ["enterprise"],
      related_github_prs: [
        "https://github.com/redwood-org/runtime/pull/108"
      ],
      description: "Tracks SAML group mapping rollout."
    });

    // 2. GitHub PR #108 -> regex mention of Slack thread dsid_slack_cutover
    writeJson(join(corpusDir, "github", "pr-108.json"), {
      dataset_doc_uuid: "dsid_gh_pr108",
      title_field_name: "title",
      content_field_names: ["description"],
      repo: "redwood-org/runtime",
      pr_number: "108",
      title: "Add SAML attribute mapping hooks",
      author: "Elena Park",
      state: "merged",
      updated_at: "2026-03-12",
      description:
        "Follow-up discussion captured in Slack thread dsid_slack_cutover."
    });

    // 3. Slack thread -> explicit deal_id link to HubSpot deal + shared account Magnetar Edgeworks
    writeJson(join(corpusDir, "slack", "saml-cutover.json"), {
      dataset_doc_uuid: "dsid_slack_cutover",
      title_field_name: "channel",
      content_field_names: ["text"],
      channel: "#eng-identity",
      thread_ts: "1712004829",
      updated_at: "2026-03-13",
      customer_company: "Magnetar Edgeworks",
      deal_id: "HS-DEAL-900",
      text: "Confirmed SAML mapping fixes Magnetar Edgeworks blocker."
    });

    // 4. HubSpot deal/account for Magnetar Edgeworks
    writeJson(join(corpusDir, "hubspot", "magnetar-deal.json"), {
      dataset_doc_uuid: "dsid_hs_magnetar",
      title_field_name: "company_name",
      content_field_names: ["crm_notes"],
      company_id: "HS-ACCT-900",
      deal_id: "HS-DEAL-900",
      company_name: "Magnetar Edgeworks",
      deal_name: "Magnetar Annual Platform",
      owner: "Markus Klein",
      stage: "procurement",
      updated_at: "2026-03-14",
      competitors: ["AetherAI"],
      crm_notes: "Enterprise deployment requires turnkey SAML mapping."
    });

    // 5. Noisy docs sharing only the common "enterprise" label with ENG-42
    for (let i = 1; i <= 5; i += 1) {
      writeJson(join(corpusDir, "confluence", `noise-${i}.json`), {
        dataset_doc_uuid: `dsid_noise_${i}`,
        title_field_name: "title",
        content_field_names: ["content"],
        title: `Generic Enterprise Note ${i}`,
        author: "Ops Bot",
        updated_at: `2026-02-0${i}`,
        labels: ["enterprise"],
        content: "Routine quarterly note."
      });
    }

    const store = new KnowledgeStore(dbPath);
    try {
      const reader = createFilesystemCorpusReader(corpusDir);
      const summary = await ingestCorpus(reader, store, profile);
      expect(summary.total).toBe(9);
      expect(summary.inserted).toBe(9);

      const graphStats = store.getGraphStats();
      expect(graphStats.aliasCount).toBeGreaterThan(0);
      expect(graphStats.resolvedEdgeCount).toBeGreaterThanOrEqual(3);
      expect(graphStats.entityMentionCount).toBeGreaterThan(0);

      // 1-hop from Jira ENG-42 finds GitHub PR #108 above noisy shared-label docs
      const hop1 = store.expandGraph({
        docId: "dsid_jira_eng42",
        maxHops: 1,
        limit: 10
      });
      expect(hop1[0]?.datasetDocUuid).toBe("dsid_gh_pr108");
      expect(hop1[0]?.hop).toBe(1);
      expect(hop1[0]?.reasons.some((r) => r.includes("explicit_field"))).toBe(
        true
      );
      expect(
        hop1.some((neighbor) => neighbor.datasetDocUuid === "dsid_slack_cutover")
      ).toBe(false);

      // Direct explicit edge ranks strictly above noisy shared label neighbors
      const noiseNeighbor = hop1.find((n) => n.datasetDocUuid === "dsid_noise_1");
      expect(noiseNeighbor).toBeDefined();
      expect((hop1[0]?.score ?? 0) > (noiseNeighbor?.score ?? 0)).toBe(true);

      // 2-hop from Jira ENG-42 reaches Slack thread via GitHub PR #108
      const hop2 = store.expandGraph({
        docId: "dsid_jira_eng42",
        maxHops: 2,
        limit: 15
      });
      const slackNeighbor = hop2.find(
        (n) => n.datasetDocUuid === "dsid_slack_cutover"
      );
      expect(slackNeighbor).toBeDefined();
      expect(slackNeighbor?.hop).toBe(2);
      expect(
        slackNeighbor?.reasons.some((r) => r.includes("regex_mention"))
      ).toBe(true);

      // 2-hop from GitHub PR #108 discovers both Slack thread (hop 1) and HubSpot deal (hop 2)
      const fromPr = store.expandGraph({
        docId: "dsid_gh_pr108",
        maxHops: 2,
        limit: 10
      });
      const fromPrIds = new Map(fromPr.map((n) => [n.datasetDocUuid, n]));
      expect(fromPrIds.get("dsid_jira_eng42")?.hop).toBe(1);
      expect(fromPrIds.get("dsid_slack_cutover")?.hop).toBe(1);
      expect(fromPrIds.get("dsid_hs_magnetar")?.hop).toBe(2);

      // Entity lookup by exact account name finds both Slack thread and HubSpot deal
      const byAccount = store.expandGraph({
        entityName: "Magnetar Edgeworks",
        entityType: "account",
        maxHops: 1,
        limit: 10
      });
      expect(
        byAccount.map((n) => n.datasetDocUuid).sort()
      ).toEqual(["dsid_hs_magnetar", "dsid_slack_cutover"]);
      expect(
        byAccount[0]?.reasons.some((r) => r.includes("Magnetar Edgeworks"))
      ).toBe(true);

      // Substring fallback lookup works when partial name is given
      const bySubstring = store.expandGraph({
        entityName: "Magnetar",
        entityType: "account",
        maxHops: 1,
        limit: 10
      });
      expect(
        bySubstring.map((n) => n.datasetDocUuid).sort()
      ).toEqual(["dsid_hs_magnetar", "dsid_slack_cutover"]);

      // Validation error when neither docId nor entityName is supplied
      expect(() => store.expandGraph({ maxHops: 1 })).toThrow(ValidationError);
    } finally {
      store.close();
    }
  });

  it("replaces graph rows atomically on upsertDocument and leaves zero orphaned rows on deleteDocument", () => {
    const root = createTempDir("srb-graph-atomic-");
    const dbPath = join(root, "brain.sqlite");
    const store = new KnowledgeStore(dbPath);

    try {
      const docA = {
        datasetDocUuid: "dsid_doc_a",
        contentHash: "hash-a-1",
        source: "jira",
        title: "Ticket ENG-10",
        author: "alice",
        status: "open",
        createdAt: "2026-01-01",
        updatedAt: "2026-01-02",
        metadataJson: JSON.stringify({ key: "ENG-10", linked_issues: ["ENG-20"] }),
        metadataText: "key: ENG-10",
        fullText: "Depends on ENG-20."
      };
      const artifactsA = extractGraphArtifacts({
        datasetDocUuid: docA.datasetDocUuid,
        contentHash: docA.contentHash,
        source: docA.source,
        raw: {
          dataset_doc_uuid: docA.datasetDocUuid,
          key: "ENG-10",
          company_name: "Northstar Corp",
          linked_issues: ["ENG-20"]
        },
        title: docA.title,
        bodyText: docA.fullText
      });

      const docB = {
        datasetDocUuid: "dsid_doc_b",
        contentHash: "hash-b-1",
        source: "jira",
        title: "Ticket ENG-20",
        author: "bob",
        status: "open",
        createdAt: "2026-01-01",
        updatedAt: "2026-01-03",
        metadataJson: JSON.stringify({ key: "ENG-20" }),
        metadataText: "key: ENG-20",
        fullText: "Target ticket for ENG-10."
      };
      const artifactsB = extractGraphArtifacts({
        datasetDocUuid: docB.datasetDocUuid,
        contentHash: docB.contentHash,
        source: docB.source,
        raw: {
          dataset_doc_uuid: docB.datasetDocUuid,
          key: "ENG-20",
          company_name: "Northstar Corp"
        },
        title: docB.title,
        bodyText: docB.fullText
      });

      store.upsertDocument(
        docA,
        [
          {
            datasetDocUuid: docA.datasetDocUuid,
            contentHash: docA.contentHash,
            chunkIndex: 0,
            title: docA.title,
            metadataText: docA.metadataText,
            body: docA.fullText
          }
        ],
        artifactsA
      );
      store.upsertDocument(
        docB,
        [
          {
            datasetDocUuid: docB.datasetDocUuid,
            contentHash: docB.contentHash,
            chunkIndex: 0,
            title: docB.title,
            metadataText: docB.metadataText,
            body: docB.fullText
          }
        ],
        artifactsB
      );

      // Edge from A -> B is resolved immediately upon B's upsert
      const initialNeighbors = store.expandGraph({
        docId: "dsid_doc_a",
        maxHops: 1
      });
      expect(initialNeighbors.map((n) => n.datasetDocUuid)).toEqual([
        "dsid_doc_b"
      ]);

      // Update docB with a new alias (ENG-30) and a different account (Solstice Labs)
      const docBUpdated = {
        ...docB,
        contentHash: "hash-b-2",
        title: "Ticket ENG-30"
      };
      const artifactsBUpdated = extractGraphArtifacts({
        datasetDocUuid: docBUpdated.datasetDocUuid,
        contentHash: docBUpdated.contentHash,
        source: docBUpdated.source,
        raw: {
          dataset_doc_uuid: docBUpdated.datasetDocUuid,
          key: "ENG-30",
          company_name: "Solstice Labs"
        },
        title: docBUpdated.title,
        bodyText: "Re-homed ticket."
      });
      store.upsertDocument(
        docBUpdated,
        [
          {
            datasetDocUuid: docBUpdated.datasetDocUuid,
            contentHash: docBUpdated.contentHash,
            chunkIndex: 0,
            title: docBUpdated.title,
            metadataText: "key: ENG-30",
            body: "Re-homed ticket."
          }
        ],
        artifactsBUpdated
      );

      // Old edge A -> B (ENG-20) and old shared account (Northstar Corp) no longer connect A to B
      expect(store.expandGraph({ docId: "dsid_doc_a", maxHops: 1 })).toEqual([]);
      expect(
        store
          .expandGraph({ entityName: "Solstice Labs", entityType: "account" })
          .map((n) => n.datasetDocUuid)
      ).toEqual(["dsid_doc_b"]);

      // Delete docB, then docA, and verify zero orphaned rows in SQLite tables
      expect(store.deleteDocument("dsid_doc_b")).toBe(true);

      const inspectDb = new Database(dbPath, { readonly: true });
      try {
        const orphanAliasesForB = inspectDb
          .prepare<[string], { count: number }>(
            "SELECT COUNT(*) AS count FROM doc_aliases WHERE dataset_doc_uuid = ?"
          )
          .get("dsid_doc_b");
        const orphanEdgesForB = inspectDb
          .prepare<[string, string], { count: number }>(
            "SELECT COUNT(*) AS count FROM graph_edges WHERE source_doc_uuid = ? OR target_doc_uuid = ?"
          )
          .get("dsid_doc_b", "dsid_doc_b");
        const orphanMentionsForB = inspectDb
          .prepare<[string], { count: number }>(
            "SELECT COUNT(*) AS count FROM entity_mentions WHERE dataset_doc_uuid = ?"
          )
          .get("dsid_doc_b");

        expect(orphanAliasesForB?.count).toBe(0);
        expect(orphanEdgesForB?.count).toBe(0);
        expect(orphanMentionsForB?.count).toBe(0);

        expect(store.deleteDocument("dsid_doc_a")).toBe(true);

        expect(store.getGraphStats()).toEqual({
          aliasCount: 0,
          edgeCount: 0,
          resolvedEdgeCount: 0,
          entityMentionCount: 0
        });
      } finally {
        inspectDb.close();
      }
    } finally {
      store.close();
    }
  });

  it("backfills the Link Graph on a pre-existing L2 SQLite DB without deleting or re-embedding cached vectors", async () => {
    const root = createTempDir("srb-graph-l2-backfill-");
    const corpusDir = join(root, "corpus");
    const dbPath = join(root, "index", "brain.sqlite");

    writeJson(join(corpusDir, "jira", "eng-501.json"), {
      dataset_doc_uuid: "dsid_jira_eng501",
      title_field_name: "summary",
      content_field_names: ["description"],
      key: "ENG-501",
      summary: "Rotate tenant signing keys for Magnetar Edgeworks",
      company_name: "Magnetar Edgeworks",
      related_github_prs: [
        "https://github.com/redwood-org/runtime/pull/501"
      ],
      description: "Coordinate cutover with PR #501."
    });

    writeJson(join(corpusDir, "github", "pr-501.json"), {
      dataset_doc_uuid: "dsid_gh_pr501",
      title_field_name: "title",
      content_field_names: ["description"],
      repo: "redwood-org/runtime",
      pr_number: "501",
      title: "Support dual signing keys",
      company_name: "Magnetar Edgeworks",
      description: "Implements ENG-501 key rotation."
    });

    const dims = 4;
    let embedBatchCalls = 0;
    const countingEmbedder: Embedder = {
      model: "gemini-embedding-001",
      dims,
      async embedBatch(texts) {
        embedBatchCalls += 1;
        return {
          vectors: texts.map(
            (_text, idx) =>
              new Float32Array([1, (idx + 1) * 0.25, 0.5, 0.1])
          ),
          inputTokens: texts.length * 12
        };
      }
    };

    const l2Profile = RunProfileSchema.parse({
      name: "l2-hybrid",
      enableEmbeddings: true,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: dims,
      enabledTools: ["search", "fetch_docs", "filter_metadata"]
    });
    const l3Profile = RunProfileSchema.parse({
      name: "l3-graph",
      enableEmbeddings: true,
      embeddingModel: "gemini-embedding-001",
      embeddingDims: dims,
      enabledTools: ["search", "fetch_docs", "filter_metadata", "expand_graph"]
    });

    // Step 1: Populate DB with L2 ingest (documents + chunks + chunk_embeddings)
    const initialStore = new KnowledgeStore(dbPath);
    let expectedEmbeddingCount = 0;
    try {
      const reader = createFilesystemCorpusReader(corpusDir);
      const l2Summary = await ingestCorpus(reader, initialStore, l2Profile, {
        embedder: countingEmbedder
      });
      expect(l2Summary.inserted).toBe(2);
      expect(l2Summary.embeddedChunks).toBeGreaterThan(0);
      expectedEmbeddingCount = l2Summary.embeddedChunks;
      expect(embedBatchCalls).toBe(1);
    } finally {
      initialStore.close();
    }

    // Step 2: Simulate a pre-L3 cached SQLite DB (empty graph tables, intact embeddings)
    const rawDb = new Database(dbPath);
    try {
      rawDb.exec(`
        DELETE FROM doc_aliases;
        DELETE FROM graph_edges;
        DELETE FROM entity_mentions;
      `);
    } finally {
      rawDb.close();
    }

    // Step 3: Reopen with l3-graph and run ingestCorpus; embeddings must stay intact
    embedBatchCalls = 0;
    const l3Store = new KnowledgeStore(dbPath);
    try {
      const hashJira = l3Store.getDocumentHash("dsid_jira_eng501");
      const hashPr = l3Store.getDocumentHash("dsid_gh_pr501");
      expect(hashJira).not.toBeNull();
      expect(hashPr).not.toBeNull();
      expect(l3Store.hasDocumentGraph("dsid_jira_eng501", hashJira ?? "")).toBe(
        false
      );
      expect(l3Store.hasDocumentGraph("dsid_gh_pr501", hashPr ?? "")).toBe(
        false
      );
      expect(l3Store.getGraphStats()).toEqual({
        aliasCount: 0,
        edgeCount: 0,
        resolvedEdgeCount: 0,
        entityMentionCount: 0
      });

      const reader = createFilesystemCorpusReader(corpusDir);
      const l3Summary = await ingestCorpus(reader, l3Store, l3Profile, {
        embedder: countingEmbedder
      });

      expect(l3Summary.total).toBe(2);
      expect(l3Summary.inserted).toBe(0);
      expect(l3Summary.updated).toBe(0);
      expect(l3Summary.unchanged).toBe(2);
      expect(l3Summary.embeddedChunks).toBe(0);
      expect(embedBatchCalls).toBe(0);

      // Verify cached chunk_embeddings and RAM vector cache are 100% preserved
      const inspectDb = new Database(dbPath, { readonly: true });
      try {
        const embeddingRow = inspectDb
          .prepare<[], { count: number }>(
            "SELECT COUNT(*) AS count FROM chunk_embeddings"
          )
          .get();
        expect(embeddingRow?.count).toBe(expectedEmbeddingCount);
      } finally {
        inspectDb.close();
      }
      expect(l3Store.getVectorCacheStats().vectorCount).toBe(
        expectedEmbeddingCount
      );

      // Verify Link Graph tables and expandGraph() are fully populated
      expect(l3Store.hasDocumentGraph("dsid_jira_eng501", hashJira ?? "")).toBe(
        true
      );
      expect(l3Store.hasDocumentGraph("dsid_gh_pr501", hashPr ?? "")).toBe(
        true
      );
      const stats = l3Store.getGraphStats();
      expect(stats.aliasCount).toBeGreaterThan(0);
      expect(stats.edgeCount).toBeGreaterThan(0);
      expect(stats.resolvedEdgeCount).toBeGreaterThanOrEqual(2);
      expect(stats.entityMentionCount).toBeGreaterThanOrEqual(2);

      const neighbors = l3Store.expandGraph({
        docId: "dsid_jira_eng501",
        maxHops: 1,
        limit: 5
      });
      expect(neighbors.map((n) => n.datasetDocUuid)).toEqual(["dsid_gh_pr501"]);

      const byAccount = l3Store.expandGraph({
        entityName: "Magnetar Edgeworks",
        entityType: "account",
        maxHops: 1,
        limit: 5
      });
      expect(byAccount.map((n) => n.datasetDocUuid).sort()).toEqual([
        "dsid_gh_pr501",
        "dsid_jira_eng501"
      ]);

      // Subsequent ingest is a clean no-op with zero re-embedding
      const repeatSummary = await ingestCorpus(reader, l3Store, l3Profile, {
        embedder: countingEmbedder
      });
      expect(repeatSummary.unchanged).toBe(2);
      expect(repeatSummary.embeddedChunks).toBe(0);
      expect(embedBatchCalls).toBe(0);
    } finally {
      l3Store.close();
    }
  });
});
