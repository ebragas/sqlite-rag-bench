import { describe, expect, it } from "vitest";
import {
  buildEntityKey,
  EXPLICIT_EDGE_WEIGHT,
  extractGraphArtifacts,
  normalizeAliasKey,
  normalizeEntityName,
  REGEX_EDGE_WEIGHT
} from "../../src/utils/graph.js";

describe("extractGraphArtifacts", () => {
  it("extracts normalized aliases across issue keys, GitHub PRs, URLs, CRM IDs, Slack threads, and paths", () => {
    const githubDoc = extractGraphArtifacts({
      datasetDocUuid: "dsid_gh_42",
      contentHash: "hash-gh",
      source: "github",
      relativePath: "github/prs/stream-timebox-pr-42.json",
      raw: {
        dataset_doc_uuid: "dsid_gh_42",
        repo: "Redwood-Org/Runtime",
        pr_number: "42",
        html_url: "https://github.com/Redwood-Org/Runtime/pull/42/"
      }
    });

    const ghAliases = new Set(githubDoc.aliases.map((item) => item.aliasKey));
    expect(ghAliases.has("dsid_gh_42")).toBe(true);
    expect(ghAliases.has("redwood-org/runtime#42")).toBe(true);
    expect(ghAliases.has("runtime#42")).toBe(true);
    expect(ghAliases.has("pr#42")).toBe(true);
    expect(ghAliases.has("https://github.com/redwood-org/runtime/pull/42")).toBe(
      true
    );
    expect(ghAliases.has("/github/prs/stream-timebox-pr-42.json")).toBe(true);
    expect(ghAliases.has("stream-timebox-pr-42")).toBe(true);

    const jiraDoc = extractGraphArtifacts({
      datasetDocUuid: "dsid_jira_1",
      contentHash: "hash-jira",
      source: "jira",
      raw: {
        dataset_doc_uuid: "dsid_jira_1",
        key: "ENG-123"
      }
    });
    expect(jiraDoc.aliases.map((item) => item.aliasKey)).toEqual([
      "dsid_jira_1",
      "eng-123"
    ]);

    const hubspotDoc = extractGraphArtifacts({
      datasetDocUuid: "dsid_hs_1",
      contentHash: "hash-hs",
      source: "hubspot",
      raw: {
        dataset_doc_uuid: "dsid_hs_1",
        company_id: "HS-ACCT-77",
        deal_id: "HS-DEAL-88",
        company_name: "Magnetar Edgeworks"
      }
    });
    const hsAliases = new Set(hubspotDoc.aliases.map((item) => item.aliasKey));
    expect(hsAliases.has("hs-acct-77")).toBe(true);
    expect(hsAliases.has("hs-deal-88")).toBe(true);

    const slackDoc = extractGraphArtifacts({
      datasetDocUuid: "dsid_slack_1",
      contentHash: "hash-slack",
      source: "slack",
      raw: {
        dataset_doc_uuid: "dsid_slack_1",
        channel: "#eng-security",
        thread_ts: "1712004829"
      }
    });
    const slackAliases = new Set(slackDoc.aliases.map((item) => item.aliasKey));
    expect(slackAliases.has("1712004829")).toBe(true);
    expect(slackAliases.has("eng-security:1712004829")).toBe(true);

    const confluenceDoc = extractGraphArtifacts({
      datasetDocUuid: "dsid_conf_1",
      contentHash: "hash-conf",
      source: "confluence",
      raw: {
        dataset_doc_uuid: "dsid_conf_1",
        page_id: "CONF-9012"
      }
    });
    expect(confluenceDoc.aliases.map((item) => item.aliasKey)).toEqual([
      "dsid_conf_1",
      "conf-9012"
    ]);
  });

  it("extracts explicit_field (1.0) and regex_mention (0.6) edges, deduplicates per edge type, and skips self-links", () => {
    const result = extractGraphArtifacts({
      datasetDocUuid: "dsid_jira_self",
      contentHash: "hash-1",
      source: "jira",
      raw: {
        dataset_doc_uuid: "dsid_jira_self",
        key: "ENG-100",
        parent_id: "ENG-100",
        linked_issues: ["ENG-200", "ENG-200", "ENG-100"],
        related_github_prs: [
          "https://github.com/acme/platform/pull/42"
        ]
      },
      title: "Fix ENG-100 and coordinate with ENG-300",
      bodyText:
        "Self ref dsid_jira_self and ENG-100 must be skipped. " +
        "See ENG-300 twice (ENG-300), acme/platform#55, PR #77, dsid_deadbeef12, and https://docs.example.com/spec/."
    });

    const explicitTargets = result.refs
      .filter((ref) => ref.edgeType === "explicit_field")
      .map((ref) => ({ targetAlias: ref.targetAlias, weight: ref.weight }));
    const regexTargets = result.refs
      .filter((ref) => ref.edgeType === "regex_mention")
      .map((ref) => ({ targetAlias: ref.targetAlias, weight: ref.weight }));

    expect(explicitTargets).toEqual(
      expect.arrayContaining([
        { targetAlias: "eng-200", weight: EXPLICIT_EDGE_WEIGHT },
        {
          targetAlias: "https://github.com/acme/platform/pull/42",
          weight: EXPLICIT_EDGE_WEIGHT
        },
        { targetAlias: "acme/platform#42", weight: EXPLICIT_EDGE_WEIGHT },
        { targetAlias: "pr#42", weight: EXPLICIT_EDGE_WEIGHT }
      ])
    );

    expect(regexTargets).toEqual(
      expect.arrayContaining([
        { targetAlias: "eng-300", weight: REGEX_EDGE_WEIGHT },
        { targetAlias: "acme/platform#55", weight: REGEX_EDGE_WEIGHT },
        { targetAlias: "pr#55", weight: REGEX_EDGE_WEIGHT },
        { targetAlias: "pr#77", weight: REGEX_EDGE_WEIGHT },
        { targetAlias: "dsid_deadbeef12", weight: REGEX_EDGE_WEIGHT },
        {
          targetAlias: "https://docs.example.com/spec",
          weight: REGEX_EDGE_WEIGHT
        }
      ])
    );

    // Self-links must never be emitted
    expect(
      result.refs.some(
        (ref) =>
          ref.targetAlias === "eng-100" || ref.targetAlias === "dsid_jira_self"
      )
    ).toBe(false);

    // Deduplicated per (source_doc, target_alias, edge_type)
    const eng200Explicit = result.refs.filter(
      (ref) =>
        ref.edgeType === "explicit_field" && ref.targetAlias === "eng-200"
    );
    expect(eng200Explicit).toHaveLength(1);

    const eng300Regex = result.refs.filter(
      (ref) => ref.edgeType === "regex_mention" && ref.targetAlias === "eng-300"
    );
    expect(eng300Regex).toHaveLength(1);
  });

  it("extracts all 6 canonical entity types with structural roles and normalized keys", () => {
    const result = extractGraphArtifacts({
      datasetDocUuid: "dsid_full_entities",
      contentHash: "hash-ent",
      source: "hubspot",
      raw: {
        dataset_doc_uuid: "dsid_full_entities",
        company_name: "Magnetar Edgeworks",
        owner: "Markus Klein",
        participants_external: [
          "Alyssa Chen <alyssa.chen@magnetar.io>",
          "Dana Wright (CISO)"
        ],
        deal_name: "Magnetar Enterprise Renewal",
        competitors: ["AetherAI", "AWS Bedrock"],
        interested_products: ["Redwood Private", "serving-runtime"],
        labels: ["enterprise", "sso"],
        channel: "#deal-desk"
      }
    });

    const byKeyAndRole = new Set(
      result.entityMentions.map(
        (mention) =>
          `${mention.entityKey}|${mention.structuralRole}|${mention.entityName}`
      )
    );

    expect(
      byKeyAndRole.has(
        "account:magnetar edgeworks|company_name|Magnetar Edgeworks"
      )
    ).toBe(true);
    expect(byKeyAndRole.has("person:markus klein|owner|Markus Klein")).toBe(
      true
    );
    expect(
      byKeyAndRole.has("person:alyssa chen|participants_external|Alyssa Chen")
    ).toBe(true);
    expect(
      byKeyAndRole.has("person:dana wright|participants_external|Dana Wright")
    ).toBe(true);
    expect(
      byKeyAndRole.has(
        "deal:magnetar enterprise renewal|deal_name|Magnetar Enterprise Renewal"
      )
    ).toBe(true);
    expect(
      byKeyAndRole.has("competitor:aetherai|competitors|AetherAI")
    ).toBe(true);
    expect(
      byKeyAndRole.has(
        "product_or_model:redwood private|interested_products|Redwood Private"
      )
    ).toBe(true);
    expect(byKeyAndRole.has("label:enterprise|labels|enterprise")).toBe(true);
    expect(byKeyAndRole.has("label:deal-desk|channel|deal-desk")).toBe(true);

    const extractedTypes = new Set(
      result.entityMentions.map((mention) => mention.entityType)
    );
    expect([...extractedTypes].sort()).toEqual([
      "account",
      "competitor",
      "deal",
      "label",
      "person",
      "product_or_model"
    ]);
  });

  it("normalizes alias keys and entity keys deterministically", () => {
    expect(normalizeAliasKey("  ENG-42  ")).toBe("eng-42");
    expect(normalizeAliasKey("https://Example.COM/Docs/Spec///")).toBe(
      "https://example.com/docs/spec"
    );
    expect(normalizeEntityName("  #Eng-Security  ")).toBe("eng-security");
    expect(buildEntityKey("account", "  Magnetar   Edgeworks ")).toBe(
      "account:magnetar edgeworks"
    );
  });

  it("completes cleanly on adversarial ReDoS-style inputs without stalling", () => {
    const dotted = "a.".repeat(50_000);
    const hyphenated = "a-".repeat(50_000);
    const trailingPunctUrl = `https://example.com/${".".repeat(50_000)}a`;
    const trailingSlashUrl = `https://example.com/${"/".repeat(50_000)}a`;
    const trailingDsid = `dsid_${"_".repeat(50_000)}a`;
    const adversarialAnglePerson = `Alice${" ".repeat(50_000)}<x>y`;
    const adversarialParenPerson = `Bob${" ".repeat(50_000)}(x)y`;

    const result = extractGraphArtifacts({
      datasetDocUuid: "dsid_redos_guard",
      contentHash: "hash-redos",
      source: "github",
      raw: {
        dataset_doc_uuid: "dsid_redos_guard",
        owner: adversarialAnglePerson,
        reviewers: [
          adversarialParenPerson,
          "Dana Wright (CISO) <dana@example.com>"
        ],
        related_links: [trailingPunctUrl, trailingSlashUrl]
      },
      title: `Stress check ${dotted.slice(0, 100)}`,
      bodyText: [
        dotted,
        hyphenated,
        trailingPunctUrl,
        trailingSlashUrl,
        trailingDsid,
        "Valid refs: acme/runtime#99, ENG-777, https://docs.example.com/guide/..."
      ].join("\n")
    });

    const regexTargets = new Set(
      result.refs
        .filter((ref) => ref.edgeType === "regex_mention")
        .map((ref) => ref.targetAlias)
    );
    expect(regexTargets.has("acme/runtime#99")).toBe(true);
    expect(regexTargets.has("pr#99")).toBe(true);
    expect(regexTargets.has("eng-777")).toBe(true);
    expect(regexTargets.has("https://docs.example.com/guide")).toBe(true);

    const people = result.entityMentions
      .filter((mention) => mention.entityType === "person")
      .map((mention) => mention.entityKey);
    expect(people).toContain("person:dana wright");
  });
});
