import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";

interface FtsMatchRow {
  doc_id: string;
  score: number;
  excerpt: string;
}

describe("better-sqlite3 FTS5 runtime support", () => {
  it("supports porter unicode61 tokenization, bm25 ranking, and snippet extraction", () => {
    const db = new Database(":memory:");
    try {
      db.exec(`
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          doc_id UNINDEXED,
          title,
          metadata_text,
          body,
          tokenize = 'porter unicode61'
        );
      `);

      const insert = db.prepare(
        "INSERT INTO chunks_fts (doc_id, title, metadata_text, body) VALUES (?, ?, ?, ?)"
      );
      insert.run(
        "doc-1",
        "Postgres Migration Runbook",
        "source:confluence owner:infra",
        "The team migrated billing storage to Postgres 16 after load testing."
      );
      insert.run(
        "doc-2",
        "Quarterly Offsite Notes",
        "source:slack channel:general",
        "We discussed snacks, travel logistics, and quarterly milestones."
      );
      insert.run(
        "doc-3",
        "Incident Followup",
        "source:linear issue:ENG-402",
        "Postgres migration checksum verification completed cleanly."
      );

      const rows = db
        .prepare(
          `
          SELECT
            doc_id,
            bm25(chunks_fts, 0.0, 4.0, 2.0, 1.0) AS score,
            snippet(chunks_fts, 3, '[', ']', '...', 8) AS excerpt
          FROM chunks_fts
          WHERE chunks_fts MATCH ?
          ORDER BY score ASC
          `
        )
        .all("migrating postgres") as FtsMatchRow[];

      expect(rows.length).toBe(2);
      const docIds = rows.map((row) => row.doc_id);
      expect(docIds).toContain("doc-1");
      expect(docIds).toContain("doc-3");

      const first = rows[0];
      const second = rows[1];
      expect(first).toBeDefined();
      expect(second).toBeDefined();
      expect(Number.isFinite(first!.score)).toBe(true);
      expect(first!.score).toBeLessThanOrEqual(second!.score);
      expect(first!.excerpt).toContain("[");
      expect(first!.excerpt).toContain("]");
    } finally {
      db.close();
    }
  });
});
