import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { KnowledgeStore } from "../../src/adapters/knowledge-store.js";
import type { NormalizedDocument } from "../../src/models/corpus.js";
import { ValidationError } from "../../src/models/errors.js";

const tempDirs: string[] = [];
const openStores: KnowledgeStore[] = [];

function createTempStore(): KnowledgeStore {
  const dir = mkdtempSync(join(tmpdir(), "srb-kstore-"));
  tempDirs.push(dir);
  const store = new KnowledgeStore(join(dir, "brain.sqlite"));
  openStores.push(store);
  return store;
}

function seedDocuments(store: KnowledgeStore): void {
  const docs: NormalizedDocument[] = [
    {
      datasetDocUuid: "dsid_crm_1",
      contentHash: "h1",
      source: "hubspot",
      title: "Acme Corp Q1 Deal",
      author: "Alice Chen",
      status: "closed_won",
      createdAt: "2025-01-10T09:00:00Z",
      updatedAt: "2025-02-01T12:00:00Z",
      metadataJson: JSON.stringify({
        account: "Acme Corp",
        "deal-stage": "won",
        priority: "high",
        enterprise: true,
        tags: ["sso", "saml"],
        owner: { team: "west" }
      }),
      metadataText: "source: hubspot",
      fullText: "Acme Corp closed won deal notes."
    },
    {
      datasetDocUuid: "dsid_crm_2",
      contentHash: "h2",
      source: "hubspot",
      title: "Beta Labs Renewal 100%_Check",
      author: "Alice Chen",
      status: "closed_lost",
      createdAt: "2025-01-20T09:00:00Z",
      updatedAt: "2025-03-05T15:00:00Z",
      metadataJson: JSON.stringify({
        account: "Beta Labs",
        "deal-stage": "lost",
        priority: "medium",
        enterprise: false,
        tags: ["billing"],
        owner: { team: "east" }
      }),
      metadataText: "source: hubspot",
      fullText: "Beta Labs renewal notes."
    },
    {
      datasetDocUuid: "dsid_crm_3",
      contentHash: "h3",
      source: "hubspot",
      title: "Gamma Robotics Expansion",
      author: "Bob Diaz",
      status: "open",
      createdAt: "2025-02-15T10:00:00Z",
      updatedAt: "2025-02-20T10:00:00Z",
      metadataJson: JSON.stringify({
        account: "Gamma Robotics",
        "deal-stage": "negotiation",
        priority: "high",
        enterprise: true,
        tags: ["sso", "audit"],
        owner: { team: "west" }
      }),
      metadataText: "source: hubspot",
      fullText: "Gamma Robotics expansion notes."
    },
    {
      datasetDocUuid: "dsid_crm_4",
      contentHash: "h4",
      source: "hubspot",
      title: "Delta Cloud Pilot",
      author: "Alice Chen",
      status: "open",
      createdAt: "2025-03-01T08:00:00Z",
      updatedAt: null,
      metadataJson: JSON.stringify({
        account: "Delta Cloud",
        "deal-stage": "pilot",
        priority: "low",
        enterprise: true,
        tags: ["trial"],
        owner: { team: "west" }
      }),
      metadataText: "source: hubspot",
      fullText: "Delta Cloud pilot notes."
    },
    {
      datasetDocUuid: "dsid_conf_1",
      contentHash: "h5",
      source: "confluence",
      title: "Security SSO Runbook",
      author: "Security Team",
      status: "approved",
      createdAt: "2024-11-01T00:00:00Z",
      updatedAt: "2025-03-10T18:00:00Z",
      metadataJson: JSON.stringify({
        space: "SEC",
        tier: "official",
        tags: ["sso", "security"]
      }),
      metadataText: "source: confluence",
      fullText: "Official SSO runbook."
    }
  ];

  for (const doc of docs) {
    store.upsertDocument(doc, [
      {
        datasetDocUuid: doc.datasetDocUuid,
        contentHash: doc.contentHash,
        chunkIndex: 0,
        title: doc.title,
        metadataText: doc.metadataText,
        body: doc.fullText
      }
    ]);
  }
}

afterEach(() => {
  while (openStores.length > 0) {
    openStores.pop()?.close();
  }
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("KnowledgeStore.filterMetadata", () => {
  it("filters by promoted columns and returns exact totalCount exceeding limit", () => {
    const store = createTempStore();
    seedDocuments(store);

    const hubspotPage = store.filterMetadata({
      source: "HUBSPOT",
      limit: 2
    });
    expect(hubspotPage.totalCount).toBe(4);
    expect(hubspotPage.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_2",
      "dsid_crm_3"
    ]);

    const aliceOpenOrWon = store.filterMetadata({
      author: "alice chen",
      createdAfter: "2025-01-15T00:00:00Z",
      createdBefore: "2025-03-01T23:59:59Z",
      orderBy: "created_at_asc"
    });
    expect(aliceOpenOrWon.totalCount).toBe(2);
    expect(aliceOpenOrWon.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_2",
      "dsid_crm_4"
    ]);

    const updatedWindow = store.filterMetadata({
      updatedAfter: "2025-02-15T00:00:00Z",
      updatedBefore: "2025-03-06T00:00:00Z",
      orderBy: "updated_at_asc"
    });
    expect(updatedWindow.totalCount).toBe(2);
    expect(updatedWindow.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_3",
      "dsid_crm_2"
    ]);

    const literalPercentMatch = store.filterMetadata({
      titleContains: "100%_"
    });
    expect(literalPercentMatch.totalCount).toBe(1);
    expect(literalPercentMatch.documents[0]?.datasetDocUuid).toBe("dsid_crm_2");

    const byTitle = store.filterMetadata({
      source: "hubspot",
      status: "OPEN",
      orderBy: "title_asc"
    });
    expect(byTitle.totalCount).toBe(2);
    expect(byTitle.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_4",
      "dsid_crm_3"
    ]);
  });

  it("filters by metadata_json eq, contains, and in across flat, hyphenated, nested, boolean, and array fields", () => {
    const store = createTempStore();
    seedDocuments(store);

    const hyphenEq = store.filterMetadata({
      metadataFilters: [{ key: "deal-stage", op: "eq", value: "WON" }]
    });
    expect(hyphenEq.totalCount).toBe(1);
    expect(hyphenEq.documents[0]?.datasetDocUuid).toBe("dsid_crm_1");

    const nestedAndBool = store.filterMetadata({
      metadataFilters: [
        { key: "owner.team", op: "eq", value: "west" },
        { key: "enterprise", op: "eq", value: "true" }
      ],
      orderBy: "created_at_desc"
    });
    expect(nestedAndBool.totalCount).toBe(3);
    expect(nestedAndBool.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_4",
      "dsid_crm_3",
      "dsid_crm_1"
    ]);

    const arrayContains = store.filterMetadata({
      metadataFilters: [{ key: "tags", op: "contains", value: "sso" }]
    });
    expect(arrayContains.totalCount).toBe(3);
    expect(arrayContains.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_conf_1",
      "dsid_crm_3",
      "dsid_crm_1"
    ]);

    const priorityIn = store.filterMetadata({
      source: "hubspot",
      metadataFilters: [
        { key: "priority", op: "in", values: ["high", "medium"] }
      ]
    });
    expect(priorityIn.totalCount).toBe(3);
    expect(priorityIn.documents.map((doc) => doc.datasetDocUuid)).toEqual([
      "dsid_crm_2",
      "dsid_crm_3",
      "dsid_crm_1"
    ]);
  });

  it("rejects SQL injection payloads and malformed metadata_json key paths with ValidationError", () => {
    const store = createTempStore();
    seedDocuments(store);

    const maliciousKeys = [
      "account') OR 1=1 --",
      "account'; DROP TABLE documents; --",
      'account" OR "1"="1',
      "owner..team",
      ".owner",
      "owner.",
      "owner[0]",
      "owner team"
    ];

    for (const key of maliciousKeys) {
      expect(() =>
        store.filterMetadata({
          metadataFilters: [{ key, op: "eq", value: "Acme Corp" }]
        })
      ).toThrow(ValidationError);
    }

    expect(store.getStats().documentCount).toBe(5);
  });
});

describe("KnowledgeStore.searchFts and rebuildSchemaCatalog", () => {
  it("ranks strict-AND matches before partial-OR fallback hits", () => {
    const store = createTempStore();
    const docs: NormalizedDocument[] = [
      {
        datasetDocUuid: "dsid_noise_1",
        contentHash: "n1",
        source: "slack",
        title: "Redwood Competitor Overview",
        author: null,
        status: null,
        createdAt: null,
        updatedAt: null,
        metadataJson: "{}",
        metadataText: "source: slack",
        fullText: "Redwood competitor analysis competitor comparison redwood pricing."
      },
      {
        datasetDocUuid: "dsid_exact_1",
        contentHash: "e1",
        source: "fireflies",
        title: "Magnetar Edgeworks Call",
        author: null,
        status: null,
        createdAt: null,
        updatedAt: null,
        metadataJson: "{}",
        metadataText: "source: fireflies",
        fullText: "Magnetar Edgeworks selected AetherAI over Redwood as their competitor."
      }
    ];
    for (const doc of docs) {
      store.upsertDocument(doc, [
        {
          datasetDocUuid: doc.datasetDocUuid,
          contentHash: doc.contentHash,
          chunkIndex: 0,
          title: doc.title,
          metadataText: doc.metadataText,
          body: doc.fullText
        }
      ]);
    }

    const hits = store.searchFts("Magnetar Edgeworks Redwood competitor", 5);
    expect(hits).toHaveLength(2);
    expect(hits[0]?.datasetDocUuid).toBe("dsid_exact_1");
    expect(hits[1]?.datasetDocUuid).toBe("dsid_noise_1");
  });

  it("preserves rare enum values and drops one-off noisy keys in schema catalog", () => {
    const store = createTempStore();
    for (let i = 0; i < 20; i += 1) {
      const status = i === 19 ? "final" : i < 15 ? "draft" : "in_review";
      const metadata: Record<string, unknown> = { status, owner: `Owner ${i}` };
      if (i === 0) {
        for (let k = 0; k < 20; k += 1) {
          metadata[`rare_one_off_key_${k}`] = `value_${k}`;
        }
      }
      store.upsertDocument(
        {
          datasetDocUuid: `dsid_gd_${i}`,
          contentHash: `h_${i}`,
          source: "google_drive",
          title: `Doc ${i}`,
          author: `Owner ${i}`,
          status,
          createdAt: null,
          updatedAt: null,
          metadataJson: JSON.stringify(metadata),
          metadataText: "source: google_drive",
          fullText: `Body ${i}`
        },
        [
          {
            datasetDocUuid: `dsid_gd_${i}`,
            contentHash: `h_${i}`,
            chunkIndex: 0,
            title: `Doc ${i}`,
            metadataText: "source: google_drive",
            body: `Body ${i}`
          }
        ]
      );
    }

    const catalog = store.rebuildSchemaCatalog();
    const gdrive = catalog.sources.find((s) => s.source === "google_drive");
    expect(gdrive).toBeDefined();
    expect(gdrive?.metadataKeys).toContain("status");
    expect(gdrive?.metadataKeys).toContain("owner");
    expect(gdrive?.metadataKeys).not.toContain("rare_one_off_key_0");
    expect(gdrive?.sampleValues.status).toEqual(["draft", "in_review", "final"]);
  });
});
