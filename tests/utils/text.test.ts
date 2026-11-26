import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";
import {
  buildFtsQueryCandidates,
  buildSafeFtsQuery,
  chunkDocumentText,
  renderCanonicalDocument,
  renderValue
} from "../../src/utils/text.js";

describe("renderCanonicalDocument", () => {
  it("renders Slack threads and nested messages without dropping extra fields", () => {
    const raw = {
      dataset_doc_uuid: "dsid_slack_1",
      title_field_name: "channel",
      content_field_names: ["messages"],
      channel: "#eng-migrations",
      author: "alice",
      status: "resolved",
      created_at: "2025-01-10T10:00:00Z",
      updated_at: "2025-01-11T12:00:00Z",
      participants: ["alice", "bob"],
      messages: [
        {
          user: "alice",
          ts: "2025-01-10T10:00:00Z",
          text: "Switch billing reads to Postgres 16.",
          reactions: ["white_check_mark"]
        },
        {
          user: "bob",
          ts: "2025-01-10T10:05:00Z",
          text: "Checksum verification passed on staging."
        }
      ],
      thread: {
        summary: "Cutover completed with zero downtime.",
        ticket_ref: "ENG-402"
      },
      extra_note: "Retain rollback snapshot for 14 days."
    };

    const rendered = renderCanonicalDocument(raw, "slack");

    expect(rendered.title).toBe("#eng-migrations");
    expect(rendered.author).toBe("alice");
    expect(rendered.status).toBe("resolved");
    expect(rendered.createdAt).toBe("2025-01-10T10:00:00Z");
    expect(rendered.updatedAt).toBe("2025-01-11T12:00:00Z");
    expect(rendered.headerCard).toContain("source: slack");
    expect(rendered.headerCard).toContain("title: #eng-migrations");
    expect(rendered.headerCard).toContain("author: alice");
    expect(rendered.headerCard).toContain("status: resolved");
    expect(rendered.bodyText).toContain("Switch billing reads to Postgres 16.");
    expect(rendered.bodyText).toContain("white_check_mark");
    expect(rendered.bodyText).toContain("Cutover completed with zero downtime.");
    expect(rendered.bodyText).toContain("ENG-402");
    expect(rendered.bodyText).toContain("Retain rollback snapshot for 14 days.");
    expect(rendered.metadataRecord.participants).toEqual(["alice", "bob"]);
  });

  it("renders Confluence body and falls back to title when title_field_name is omitted", () => {
    const raw = {
      dataset_doc_uuid: "dsid_conf_1",
      title: "Incident Response Handbook",
      owner: "sre-team",
      last_updated: "2025-03-01",
      body: {
        overview: "Escalation policy for SEV-1 outages.",
        steps: ["Page on-call", "Open incident channel", "Publish postmortem"]
      }
    };

    const rendered = renderCanonicalDocument(raw, "confluence");

    expect(rendered.title).toBe("Incident Response Handbook");
    expect(rendered.author).toBe("sre-team");
    expect(rendered.updatedAt).toBe("2025-03-01");
    expect(rendered.fullText).toContain("Escalation policy for SEV-1 outages.");
    expect(rendered.fullText).toContain("Publish postmortem");
    expect(renderValue(null)).toBe("");
    expect(renderValue(42)).toBe("42");
    expect(renderValue(true)).toBe("true");
  });
});

describe("chunkDocumentText", () => {
  it("keeps short documents intact as a single chunk", () => {
    const headerCard = "source: confluence\ntitle: Short Policy";
    const bodyText = "All production credentials rotate every 90 days.";

    const chunks = chunkDocumentText(
      { headerCard, bodyText },
      {
        shortDocMaxChars: 6000,
        chunkTargetChars: 4000,
        chunkOverlapChars: 600,
        includeHeaderCard: true
      }
    );

    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toContain("source: confluence");
    expect(chunks[0]).toContain("rotate every 90 days");
  });

  it("omits header card when includeHeaderCard is false", () => {
    const chunks = chunkDocumentText(
      {
        headerCard: "source: jira\ntitle: Ticket 1",
        bodyText: "Root cause was an expired TLS certificate."
      },
      {
        shortDocMaxChars: 6000,
        chunkTargetChars: 4000,
        chunkOverlapChars: 600,
        includeHeaderCard: false
      }
    );

    expect(chunks).toEqual(["Root cause was an expired TLS certificate."]);
  });

  it("splits long documents structurally with overlap and header cards", () => {
    const headerCard = "source: confluence\ntitle: Architecture Spec";
    const paragraphs = [
      "## Section One\n" + "Alpha service handles authentication tokens. ".repeat(12),
      "## Section Two\n" + "Beta service indexes customer records cleanly. ".repeat(12),
      "## Section Three\n" + "Gamma service exports audit logs nightly. ".repeat(12)
    ];
    const bodyText = paragraphs.join("\n\n");

    const chunks = chunkDocumentText(
      { headerCard, bodyText },
      {
        shortDocMaxChars: 400,
        chunkTargetChars: 550,
        chunkOverlapChars: 100,
        includeHeaderCard: true
      }
    );

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.startsWith(headerCard)).toBe(true);
    }
    const joined = chunks.join("\n");
    expect(joined).toContain("Alpha service");
    expect(joined).toContain("Beta service");
    expect(joined).toContain("Gamma service");
  });

  it("handles oversized single lines without spaces deterministically", () => {
    const longToken = "x".repeat(500);
    const chunks = chunkDocumentText(
      { headerCard: "source: github\ntitle: Blob", bodyText: longToken },
      {
        shortDocMaxChars: 100,
        chunkTargetChars: 180,
        chunkOverlapChars: 30,
        includeHeaderCard: true
      }
    );

    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks.every((chunk) => chunk.startsWith("source: github"))).toBe(true);
  });
});

describe("buildSafeFtsQuery", () => {
  it("converts natural language questions into quoted OR queries without stop words", () => {
    const query = buildSafeFtsQuery(
      "Which competitor did Magnetar Edgeworks choose over Redwood, and why?"
    );

    expect(query).toBe(
      '"competitor" OR "magnetar" OR "edgeworks" OR "choose" OR "redwood"'
    );
  });

  it("strips FTS5 operators and syntax characters and returns empty string when nothing remains", () => {
    expect(buildSafeFtsQuery('--- ::: () * ^ "" AND OR NOT NEAR')).toBe("");
    expect(buildSafeFtsQuery("   ")).toBe("");
    expect(buildSafeFtsQuery('foo-bar:baz* ("quoted") NOT AND OR')).toBe(
      '"foo" OR "bar" OR "baz" OR "quoted"'
    );
    expect(buildSafeFtsQuery("what is it")).toBe('"what" OR "is" OR "it"');
  });

  it("produces queries that SQLite FTS5 executes without syntax errors", () => {
    const db = new Database(":memory:");
    try {
      db.exec(
        "CREATE VIRTUAL TABLE t USING fts5(body, tokenize='porter unicode61');"
      );
      db.prepare("INSERT INTO t(body) VALUES (?)").run(
        "Magnetar Edgeworks selected AetherAI for SSO-SAML mapping."
      );

      const nastyInputs = [
        'title:Magnetar "unclosed quote',
        "SSO-SAML (AetherAI*) AND NOT OR ^NEAR/3",
        "foo:bar:baz -- + ~ {a} [b] / \\ ? !",
        "competitor? (redwood) - aetherai"
      ];

      for (const rawInput of nastyInputs) {
        const safe = buildSafeFtsQuery(rawInput);
        expect(safe.length).toBeGreaterThan(0);
        const rows = db
          .prepare("SELECT body FROM t WHERE t MATCH ?")
          .all(safe) as Array<{ body: string }>;
        expect(Array.isArray(rows)).toBe(true);
      }
    } finally {
      db.close();
    }
  });

  it("builds AND-first then OR-fallback candidates for multi-term queries", () => {
    expect(buildFtsQueryCandidates("Magnetar")).toEqual(['"magnetar"']);
    expect(buildFtsQueryCandidates("Magnetar Edgeworks Redwood")).toEqual([
      '"magnetar" AND "edgeworks" AND "redwood"',
      '"magnetar" OR "edgeworks" OR "redwood"'
    ]);
    expect(buildFtsQueryCandidates("   ---   ")).toEqual([]);
  });
});
