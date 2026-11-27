export const ENTITY_TYPES = [
  "account",
  "person",
  "deal",
  "competitor",
  "product_or_model",
  "label"
] as const;

export type EntityType = (typeof ENTITY_TYPES)[number];

export const GRAPH_EDGE_TYPES = ["explicit_field", "regex_mention"] as const;

export type GraphEdgeType = (typeof GRAPH_EDGE_TYPES)[number];

export interface DocAliasRecord {
  aliasKey: string;
  datasetDocUuid: string;
  contentHash: string;
  aliasKind: string;
}

export interface RawGraphRef {
  sourceDocUuid: string;
  targetAlias: string;
  edgeType: GraphEdgeType;
  weight: number;
  reason: string;
  contentHash: string;
}

export interface EntityMentionRecord {
  entityKey: string;
  entityType: EntityType;
  entityName: string;
  datasetDocUuid: string;
  structuralRole: string;
  contentHash: string;
}

export interface ExtractedGraphArtifacts {
  aliases: DocAliasRecord[];
  refs: RawGraphRef[];
  entityMentions: EntityMentionRecord[];
}

export interface ExtractGraphInput {
  datasetDocUuid: string;
  contentHash: string;
  source: string;
  raw: Record<string, unknown>;
  title?: string;
  bodyText?: string;
  relativePath?: string;
}

export const EXPLICIT_EDGE_WEIGHT = 1.0;
export const REGEX_EDGE_WEIGHT = 0.6;

const MAX_ALIAS_LENGTH = 256;
const MAX_ENTITY_NAME_LENGTH = 120;
const MIN_STEM_ALIAS_LENGTH = 12;

const ISSUE_KEY_PATTERN = /\b([A-Z]{2,10}-\d+)\b/g;
const GITHUB_ORG_REPO_REF_PATTERN =
  /(?<![A-Za-z0-9_./-])([A-Za-z0-9_.-]{1,100}\/[A-Za-z0-9_.-]{1,100})#(\d+)\b/g;
const GITHUB_PR_WORD_PATTERN =
  /\b(?:PR|pull\s+request|MR)\s*#(\d+)\b/gi;
const GITHUB_PR_URL_PATTERN =
  /https?:\/\/github\.com\/([A-Za-z0-9_.-]{1,100}\/[A-Za-z0-9_.-]{1,100})\/(?:pull|issues)\/(\d+)/gi;
const DSID_UUID_PATTERN = /\b(dsid_[a-z0-9][a-z0-9_-]*)\b/gi;
const URL_PATTERN = /https?:\/\/[^\s<>"')\]}]+/gi;

const URL_TRAILING_PUNCT_CHARS = ".,;:!?";
const URL_TRAILING_SLASH_CHARS = "/";
const DSID_TRAILING_CHARS = "_-";

const OWN_URL_FIELDS = new Set([
  "url",
  "permalink",
  "html_url",
  "page_url",
  "web_url",
  "drive_url",
  "doc_url",
  "link"
]);

const EXPLICIT_REF_FIELDS = new Set([
  "parent_id",
  "parent_issue",
  "parent_key",
  "epic_key",
  "epic_id",
  "linked_issues",
  "linked_linear",
  "linked_jira",
  "linked_prs",
  "linked_docs",
  "linked_artifacts",
  "linked_fireflies",
  "linked_gmail_threads",
  "linked_drive_docs",
  "linked_support_tickets",
  "related_docs",
  "related_pages",
  "related_issues",
  "related_links",
  "related_doc_ids",
  "related_github_prs",
  "related_confluence_pages",
  "dependencies",
  "blocked_by",
  "blocks",
  "pr_url",
  "issue_key",
  "issue_id",
  "ticket_key",
  "ticket_id",
  "thread_ts",
  "parent_thread_ts",
  "channel_id",
  "deal_id",
  "company_id",
  "crm_account_id",
  "account_id",
  "references",
  "links",
  "design_links",
  "docs_and_artifacts",
  "source_doc_id"
]);

const ENTITY_KEY_MAP: Readonly<Record<string, EntityType>> = {
  // account
  company: "account",
  company_name: "account",
  account: "account",
  account_name: "account",
  customer: "account",
  customer_company: "account",
  customer_name: "account",
  related_account: "account",
  organization: "account",
  org: "account",
  client: "account",
  // person
  author: "person",
  assignee: "person",
  owner: "person",
  reporter: "person",
  creator: "person",
  participants: "person",
  participants_internal: "person",
  participants_external: "person",
  attendees: "person",
  redwood_attendees: "person",
  customer_attendees: "person",
  reviewers: "person",
  collaborators: "person",
  stakeholders: "person",
  redwood_owner: "person",
  mailbox_owner: "person",
  se_assigned: "person",
  csm_assigned: "person",
  from: "person",
  to: "person",
  cc: "person",
  sender: "person",
  email: "person",
  // deal
  deal: "deal",
  deal_name: "deal",
  opportunity: "deal",
  opportunity_name: "deal",
  // competitor
  competitor: "competitor",
  competitors: "competitor",
  competitors_mentioned: "competitor",
  // product_or_model
  product: "product_or_model",
  products: "product_or_model",
  interested_products: "product_or_model",
  model: "product_or_model",
  models: "product_or_model",
  component: "product_or_model",
  components: "product_or_model",
  service: "product_or_model",
  services: "product_or_model",
  repository: "product_or_model",
  repo: "product_or_model",
  // label
  label: "label",
  labels: "label",
  tag: "label",
  tags: "label",
  topic: "label",
  topics: "label",
  channel: "label",
  project: "label",
  epic: "label",
  space: "label",
  team: "label",
  owner_team: "label"
};

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asTrimmedScalar(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return null;
}

function stripTrailingChars(value: string, chars: string): string {
  let end = value.length;
  while (end > 0) {
    if (!chars.includes(value.charAt(end - 1))) {
      break;
    }
    end -= 1;
  }
  return end === value.length ? value : value.slice(0, end);
}

function stripTrailingBracketedSuffix(
  value: string,
  openChar: string,
  closeChar: string
): string {
  const trimmed = value.trimEnd();
  if (!trimmed.endsWith(closeChar)) {
    return trimmed;
  }
  const openIndex = trimmed.lastIndexOf(openChar);
  if (openIndex < 0 || openIndex >= trimmed.length - 2) {
    return trimmed;
  }
  const inner = trimmed.slice(openIndex + 1, trimmed.length - 1);
  if (inner.includes(closeChar)) {
    return trimmed;
  }
  return trimmed.slice(0, openIndex).trimEnd();
}

export function normalizeUrlAlias(rawUrl: string): string {
  const cleaned = stripTrailingChars(rawUrl.trim(), URL_TRAILING_PUNCT_CHARS);
  const withoutTrailingSlash = stripTrailingChars(
    cleaned,
    URL_TRAILING_SLASH_CHARS
  );
  return withoutTrailingSlash.toLowerCase();
}

export function normalizeAliasKey(rawAlias: string): string {
  const trimmed = rawAlias.trim().replace(/\s+/g, " ");
  if (trimmed.length === 0) {
    return "";
  }
  if (/^https?:\/\//i.test(trimmed)) {
    return normalizeUrlAlias(trimmed);
  }
  return trimmed.toLowerCase();
}

export function normalizeEntityName(rawName: string): string {
  return rawName
    .trim()
    .replace(/^#+/, "")
    .replace(/\s+/g, " ")
    .toLowerCase();
}

export function buildEntityKey(entityType: EntityType, rawName: string): string {
  return `${entityType}:${normalizeEntityName(rawName)}`;
}

function cleanPersonDisplayName(rawValue: string): string | null {
  const trimmed = rawValue.trim();
  if (trimmed.length === 0) {
    return null;
  }

  const angleEmailOnly = /^<([^<>@\s]+@[^<>@\s]+)>$/.exec(trimmed);
  if (angleEmailOnly?.[1] !== undefined) {
    return angleEmailOnly[1].trim();
  }

  const withoutAngle = stripTrailingBracketedSuffix(trimmed, "<", ">");
  const withoutParen = stripTrailingBracketedSuffix(withoutAngle, "(", ")");
  const stripped = withoutParen.replace(/\s+/g, " ").trim();

  if (stripped.length > 0) {
    return stripped;
  }
  return trimmed;
}

function cleanEntityDisplayName(
  entityType: EntityType,
  rawValue: string
): string | null {
  const base =
    entityType === "person"
      ? cleanPersonDisplayName(rawValue)
      : rawValue.trim().replace(/\s+/g, " ");

  if (base === null || base.length === 0 || base.length > MAX_ENTITY_NAME_LENGTH) {
    return null;
  }

  const normalized = normalizeEntityName(base);
  if (normalized.length < 2) {
    return null;
  }
  return entityType === "label" ? base.replace(/^#+/, "") : base;
}

function collectStringsFromValue(value: unknown, depth = 0): string[] {
  if (depth > 3) {
    return [];
  }
  const scalar = asTrimmedScalar(value);
  if (scalar !== null) {
    return [scalar];
  }
  if (Array.isArray(value)) {
    const out: string[] = [];
    for (const item of value) {
      out.push(...collectStringsFromValue(item, depth + 1));
    }
    return out;
  }
  if (isPlainRecord(value)) {
    const preferredKeys = [
      "name",
      "display_name",
      "full_name",
      "email",
      "user",
      "author",
      "id",
      "key",
      "url",
      "title",
      "value"
    ];
    const out: string[] = [];
    for (const key of preferredKeys) {
      const candidate = asTrimmedScalar(value[key]);
      if (candidate !== null) {
        out.push(candidate);
      }
    }
    return out;
  }
  return [];
}

function extractGithubAliasesFromUrl(rawUrl: string): string[] {
  const results: string[] = [];
  GITHUB_PR_URL_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = GITHUB_PR_URL_PATTERN.exec(rawUrl)) !== null) {
    const orgRepo = match[1]?.toLowerCase();
    const num = match[2];
    if (orgRepo !== undefined && num !== undefined) {
      results.push(`${orgRepo}#${num}`);
      const slashIndex = orgRepo.lastIndexOf("/");
      if (slashIndex >= 0 && slashIndex < orgRepo.length - 1) {
        results.push(`${orgRepo.slice(slashIndex + 1)}#${num}`);
      }
      results.push(`pr#${num}`);
    }
  }
  return results;
}

function isHubspotOrCrmDoc(source: string, raw: Record<string, unknown>): boolean {
  const lowerSource = source.toLowerCase();
  if (
    lowerSource.includes("hubspot") ||
    lowerSource.includes("crm") ||
    lowerSource.includes("salesforce")
  ) {
    return true;
  }
  return (
    asTrimmedScalar(raw.company_name) !== null ||
    asTrimmedScalar(raw.deal_name) !== null
  );
}

function isSlackOrChatDoc(source: string, raw: Record<string, unknown>): boolean {
  const lowerSource = source.toLowerCase();
  if (lowerSource.includes("slack") || lowerSource.includes("chat")) {
    return true;
  }
  return (
    asTrimmedScalar(raw.channel) !== null &&
    asTrimmedScalar(raw.thread_ts) !== null
  );
}

function extractDocAliases(input: ExtractGraphInput): DocAliasRecord[] {
  const { datasetDocUuid, contentHash, source, raw, relativePath } = input;
  const aliasesByKey = new Map<string, DocAliasRecord>();

  const addAlias = (rawAlias: string | null, aliasKind: string): void => {
    if (rawAlias === null) {
      return;
    }
    const aliasKey = normalizeAliasKey(rawAlias);
    if (aliasKey.length === 0 || aliasKey.length > MAX_ALIAS_LENGTH) {
      return;
    }
    if (!aliasesByKey.has(aliasKey)) {
      aliasesByKey.set(aliasKey, {
        aliasKey,
        datasetDocUuid,
        contentHash,
        aliasKind
      });
    }
  };

  addAlias(datasetDocUuid, "uuid");

  const issueKey =
    asTrimmedScalar(raw.key) ??
    (source.toLowerCase().includes("jira") || source.toLowerCase().includes("linear")
      ? asTrimmedScalar(raw.issue_key) ?? asTrimmedScalar(raw.ticket_key)
      : null);
  addAlias(issueKey, "issue_key");

  const prNumber =
    asTrimmedScalar(raw.pr_number) ??
    asTrimmedScalar(raw.pull_request_number) ??
    (source.toLowerCase().includes("github")
      ? asTrimmedScalar(raw.issue_number) ?? asTrimmedScalar(raw.number)
      : null);
  if (prNumber !== null && /^\d+$/.test(prNumber)) {
    const repo =
      asTrimmedScalar(raw.repository) ?? asTrimmedScalar(raw.repo);
    if (repo !== null) {
      const normalizedRepo = repo.toLowerCase();
      addAlias(`${normalizedRepo}#${prNumber}`, "github_pr");
      const slashIndex = normalizedRepo.lastIndexOf("/");
      if (slashIndex >= 0 && slashIndex < normalizedRepo.length - 1) {
        addAlias(
          `${normalizedRepo.slice(slashIndex + 1)}#${prNumber}`,
          "github_pr"
        );
      }
    }
    addAlias(`pr#${prNumber}`, "github_pr");
  }

  const pageId =
    asTrimmedScalar(raw.page_id) ??
    asTrimmedScalar(raw.confluence_page_id) ??
    (source.toLowerCase().includes("confluence")
      ? asTrimmedScalar(raw.id)
      : null);
  addAlias(pageId, "page_id");

  if (isHubspotOrCrmDoc(source, raw)) {
    addAlias(asTrimmedScalar(raw.company_id), "crm_id");
    addAlias(asTrimmedScalar(raw.deal_id), "crm_id");
    addAlias(asTrimmedScalar(raw.account_id), "crm_id");
    addAlias(asTrimmedScalar(raw.hubspot_id), "crm_id");
  }

  const meetingId = asTrimmedScalar(raw.meeting_id);
  if (meetingId !== null) {
    addAlias(meetingId, "meeting_id");
  }

  const threadId = asTrimmedScalar(raw.thread_id);
  if (threadId !== null) {
    addAlias(threadId, "thread_id");
  }

  if (isSlackOrChatDoc(source, raw)) {
    const threadTs = asTrimmedScalar(raw.thread_ts);
    const channel = asTrimmedScalar(raw.channel);
    if (threadTs !== null) {
      addAlias(threadTs, "thread_ts");
      if (channel !== null) {
        addAlias(`${normalizeEntityName(channel)}:${threadTs}`, "thread_ts");
      }
    }
  }

  for (const [key, value] of Object.entries(raw)) {
    if (!OWN_URL_FIELDS.has(key.toLowerCase())) {
      continue;
    }
    const urlValue = asTrimmedScalar(value);
    if (urlValue === null || !/^https?:\/\//i.test(urlValue)) {
      continue;
    }
    addAlias(urlValue, "url");
    for (const ghAlias of extractGithubAliasesFromUrl(urlValue)) {
      addAlias(ghAlias, "github_pr");
    }
  }

  if (relativePath !== undefined && relativePath.trim().length > 0) {
    const normalizedPath = relativePath
      .replace(/\\/g, "/")
      .replace(/^\/+/, "");
    if (normalizedPath.length > 0) {
      addAlias(`/${normalizedPath}`, "path");
      addAlias(normalizedPath, "path");
      const fileName = normalizedPath.slice(normalizedPath.lastIndexOf("/") + 1);
      const stem = fileName.endsWith(".json")
        ? fileName.slice(0, -5)
        : fileName;
      if (stem.length >= MIN_STEM_ALIAS_LENGTH && stem.includes("-")) {
        addAlias(stem, "path_stem");
      }
    }
  }

  return [...aliasesByKey.values()];
}

interface ParsedRefCandidate {
  targetAlias: string;
  displayLabel: string;
}

function extractEmbeddedTargetsFromString(text: string): ParsedRefCandidate[] {
  const candidates: ParsedRefCandidate[] = [];

  ISSUE_KEY_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = ISSUE_KEY_PATTERN.exec(text)) !== null) {
    const key = match[1];
    if (key !== undefined) {
      candidates.push({
        targetAlias: normalizeAliasKey(key),
        displayLabel: key
      });
    }
  }

  GITHUB_ORG_REPO_REF_PATTERN.lastIndex = 0;
  while ((match = GITHUB_ORG_REPO_REF_PATTERN.exec(text)) !== null) {
    const orgRepo = match[1];
    const num = match[2];
    if (orgRepo !== undefined && num !== undefined) {
      candidates.push({
        targetAlias: `${orgRepo.toLowerCase()}#${num}`,
        displayLabel: `${orgRepo}#${num}`
      });
      candidates.push({
        targetAlias: `pr#${num}`,
        displayLabel: `PR #${num}`
      });
    }
  }

  GITHUB_PR_WORD_PATTERN.lastIndex = 0;
  while ((match = GITHUB_PR_WORD_PATTERN.exec(text)) !== null) {
    const num = match[1];
    if (num !== undefined) {
      candidates.push({
        targetAlias: `pr#${num}`,
        displayLabel: `PR #${num}`
      });
    }
  }

  DSID_UUID_PATTERN.lastIndex = 0;
  while ((match = DSID_UUID_PATTERN.exec(text)) !== null) {
    const rawId =
      match[1] !== undefined
        ? stripTrailingChars(match[1], DSID_TRAILING_CHARS)
        : undefined;
    if (rawId !== undefined && rawId.length > 5) {
      candidates.push({
        targetAlias: normalizeAliasKey(rawId),
        displayLabel: rawId
      });
    }
  }

  URL_PATTERN.lastIndex = 0;
  while ((match = URL_PATTERN.exec(text)) !== null) {
    const rawUrl = match[0];
    const normalizedUrl = normalizeUrlAlias(rawUrl);
    if (normalizedUrl.length > 0 && normalizedUrl.length <= MAX_ALIAS_LENGTH) {
      candidates.push({
        targetAlias: normalizedUrl,
        displayLabel: stripTrailingChars(rawUrl, URL_TRAILING_PUNCT_CHARS)
      });
    }
    for (const ghAlias of extractGithubAliasesFromUrl(rawUrl)) {
      candidates.push({
        targetAlias: ghAlias,
        displayLabel: ghAlias
      });
    }
  }

  return candidates;
}

function looksLikeDirectRefIdentifier(value: string): boolean {
  if (value.length === 0 || value.length > MAX_ALIAS_LENGTH) {
    return false;
  }
  if (/\s/.test(value)) {
    return false;
  }
  return value.length >= 2;
}

function extractGraphRefs(
  input: ExtractGraphInput,
  selfAliasKeys: ReadonlySet<string>
): RawGraphRef[] {
  const { datasetDocUuid, contentHash, raw, title, bodyText } = input;
  const refsByKey = new Map<string, RawGraphRef>();

  const addRef = (
    targetAlias: string,
    edgeType: GraphEdgeType,
    weight: number,
    reason: string
  ): void => {
    const normalized = normalizeAliasKey(targetAlias);
    if (
      normalized.length === 0 ||
      normalized.length > MAX_ALIAS_LENGTH ||
      selfAliasKeys.has(normalized)
    ) {
      return;
    }
    const dedupeKey = `${edgeType}:${normalized}`;
    if (!refsByKey.has(dedupeKey)) {
      refsByKey.set(dedupeKey, {
        sourceDocUuid: datasetDocUuid,
        targetAlias: normalized,
        edgeType,
        weight,
        reason,
        contentHash
      });
    }
  };

  for (const [key, value] of Object.entries(raw)) {
    const lowerKey = key.toLowerCase();
    if (!EXPLICIT_REF_FIELDS.has(lowerKey)) {
      continue;
    }

    const rawValues = collectStringsFromValue(value);
    for (const item of rawValues) {
      const trimmed = item.trim();
      if (trimmed.length === 0) {
        continue;
      }

      if (looksLikeDirectRefIdentifier(trimmed)) {
        addRef(
          trimmed,
          "explicit_field",
          EXPLICIT_EDGE_WEIGHT,
          `explicit_field: ${key} -> ${trimmed}`
        );
      }

      const embedded = extractEmbeddedTargetsFromString(trimmed);
      for (const candidate of embedded) {
        addRef(
          candidate.targetAlias,
          "explicit_field",
          EXPLICIT_EDGE_WEIGHT,
          `explicit_field: ${key} -> ${candidate.displayLabel}`
        );
      }
    }
  }

  const proseParts: string[] = [];
  if (title !== undefined && title.trim().length > 0) {
    proseParts.push(title);
  }
  if (bodyText !== undefined && bodyText.trim().length > 0) {
    proseParts.push(bodyText);
  }
  const combinedText = proseParts.join("\n");

  if (combinedText.length > 0) {
    const regexCandidates = extractEmbeddedTargetsFromString(combinedText);
    for (const candidate of regexCandidates) {
      addRef(
        candidate.targetAlias,
        "regex_mention",
        REGEX_EDGE_WEIGHT,
        `regex_mention: ${candidate.displayLabel}`
      );
    }
  }

  return [...refsByKey.values()];
}

function extractEntityMentions(input: ExtractGraphInput): EntityMentionRecord[] {
  const { datasetDocUuid, contentHash, raw } = input;
  const mentionsByKey = new Map<string, EntityMentionRecord>();

  for (const [fieldKey, value] of Object.entries(raw)) {
    const normalizedFieldKey = fieldKey.toLowerCase();
    const entityType = ENTITY_KEY_MAP[normalizedFieldKey];
    if (entityType === undefined) {
      continue;
    }

    const rawItems = collectStringsFromValue(value);
    for (const rawItem of rawItems) {
      const displayName = cleanEntityDisplayName(entityType, rawItem);
      if (displayName === null) {
        continue;
      }
      const entityKey = buildEntityKey(entityType, displayName);
      const dedupeKey = `${entityKey}::${normalizedFieldKey}`;
      if (!mentionsByKey.has(dedupeKey)) {
        mentionsByKey.set(dedupeKey, {
          entityKey,
          entityType,
          entityName: displayName,
          datasetDocUuid,
          structuralRole: normalizedFieldKey,
          contentHash
        });
      }
    }
  }

  return [...mentionsByKey.values()];
}

export function extractGraphArtifacts(
  input: ExtractGraphInput
): ExtractedGraphArtifacts {
  const aliases = extractDocAliases(input);
  const selfAliasKeys = new Set(aliases.map((alias) => alias.aliasKey));
  const refs = extractGraphRefs(input, selfAliasKeys);
  const entityMentions = extractEntityMentions(input);

  return {
    aliases,
    refs,
    entityMentions
  };
}
