export interface ChunkOptions {
  shortDocMaxChars: number;
  chunkTargetChars: number;
  chunkOverlapChars: number;
  includeHeaderCard: boolean;
}

export interface RenderedCorpusDocument {
  title: string;
  author: string | null;
  status: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  headerCard: string;
  metadataText: string;
  bodyText: string;
  fullText: string;
  metadataRecord: Record<string, unknown>;
}

const CONTROL_KEYS = new Set([
  "dataset_doc_uuid",
  "title_field_name",
  "content_field_names"
]);

const AUTHOR_KEYS = [
  "author",
  "owner",
  "creator",
  "reporter",
  "redwood_owner",
  "mailbox_owner",
  "sender"
] as const;

const STATUS_KEYS = ["status", "state", "stage", "priority"] as const;

const CREATED_AT_KEYS = [
  "created_at",
  "recorded_at",
  "first_email_at",
  "first_message_ts",
  "createdAt"
] as const;

const UPDATED_AT_KEYS = [
  "updated_at",
  "last_updated",
  "last_modified",
  "last_email_at",
  "last_message_ts",
  "last_activity_at",
  "merged_at",
  "updatedAt"
] as const;

const TITLE_FALLBACK_KEYS = [
  "title",
  "subject",
  "summary",
  "company_name",
  "channel",
  "name",
  "key"
] as const;

const FTS_OPERATOR_WORDS = new Set(["and", "or", "not", "near"]);

const STOP_WORDS = new Set([
  "a",
  "about",
  "an",
  "are",
  "as",
  "at",
  "be",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "had",
  "has",
  "have",
  "how",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "our",
  "over",
  "should",
  "that",
  "the",
  "their",
  "there",
  "they",
  "this",
  "to",
  "us",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "with",
  "would",
  "you",
  "your"
]);

const MAX_FTS_TERMS = 24;
const MAX_FTS_AND_TERMS = 8;

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNonEmptyScalar(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "boolean") {
    return String(value);
  }
  return null;
}

function pickFirstScalar(
  raw: Record<string, unknown>,
  keys: readonly string[]
): string | null {
  for (const key of keys) {
    const scalar = asNonEmptyScalar(raw[key]);
    if (scalar !== null) {
      return scalar;
    }
  }
  return null;
}

function formatMessageObject(record: Record<string, unknown>): string | null {
  const speaker =
    asNonEmptyScalar(record.author) ??
    asNonEmptyScalar(record.user) ??
    asNonEmptyScalar(record.sender) ??
    asNonEmptyScalar(record.speaker);
  const timestamp =
    asNonEmptyScalar(record.ts) ??
    asNonEmptyScalar(record.timestamp) ??
    asNonEmptyScalar(record.created_at);
  const body =
    asNonEmptyScalar(record.text) ??
    asNonEmptyScalar(record.body) ??
    asNonEmptyScalar(record.message) ??
    asNonEmptyScalar(record.content);

  if (body === null || (speaker === null && timestamp === null)) {
    return null;
  }

  const consumed = new Set([
    "author",
    "user",
    "sender",
    "speaker",
    "ts",
    "timestamp",
    "created_at",
    "text",
    "body",
    "message",
    "content"
  ]);
  const prefixParts: string[] = [];
  if (timestamp !== null) {
    prefixParts.push(`[${timestamp}]`);
  }
  if (speaker !== null) {
    prefixParts.push(`${speaker}:`);
  }

  const extraLines: string[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (consumed.has(key)) {
      continue;
    }
    const rendered = renderValue(value);
    if (rendered.length > 0) {
      extraLines.push(`${key}: ${rendered}`);
    }
  }

  const primary = `${prefixParts.join(" ")} ${body}`.trim();
  if (extraLines.length === 0) {
    return primary;
  }
  return `${primary}\n${extraLines.join("\n")}`;
}

export function renderValue(value: unknown, depth = 0): string {
  const scalar = asNonEmptyScalar(value);
  if (scalar !== null) {
    return scalar;
  }

  if (Array.isArray(value)) {
    const renderedItems: string[] = [];
    for (const item of value) {
      const rendered = renderValue(item, depth + 1);
      if (rendered.length > 0) {
        renderedItems.push(rendered);
      }
    }
    if (renderedItems.length === 0) {
      return "";
    }
    const hasMultiline = renderedItems.some(
      (item) => item.includes("\n") || item.length > 80
    );
    return hasMultiline ? renderedItems.join("\n") : renderedItems.join(", ");
  }

  if (isPlainRecord(value)) {
    const messageLine = formatMessageObject(value);
    if (messageLine !== null) {
      return messageLine;
    }

    const lines: string[] = [];
    for (const [key, entryValue] of Object.entries(value)) {
      const rendered = renderValue(entryValue, depth + 1);
      if (rendered.length === 0) {
        continue;
      }
      if (rendered.includes("\n")) {
        lines.push(`${key}:\n${rendered}`);
      } else {
        lines.push(`${key}: ${rendered}`);
      }
    }
    return lines.join("\n");
  }

  return "";
}

function buildHeaderCard(fields: {
  datasetDocUuid: string | null;
  source: string;
  title: string;
  author: string | null;
  status: string | null;
  createdAt: string | null;
  updatedAt: string | null;
}): string {
  const lines: string[] = [`source: ${fields.source}`, `title: ${fields.title}`];
  if (fields.datasetDocUuid !== null) {
    lines.push(`doc_id: ${fields.datasetDocUuid}`);
  }
  if (fields.author !== null) {
    lines.push(`author: ${fields.author}`);
  }
  if (fields.status !== null) {
    lines.push(`status: ${fields.status}`);
  }
  if (fields.createdAt !== null) {
    lines.push(`created_at: ${fields.createdAt}`);
  }
  if (fields.updatedAt !== null) {
    lines.push(`updated_at: ${fields.updatedAt}`);
  }
  return lines.join("\n");
}

function resolveTitle(raw: Record<string, unknown>): string {
  const titleFieldName = asNonEmptyScalar(raw.title_field_name);
  if (titleFieldName !== null) {
    const dynamicTitle = asNonEmptyScalar(raw[titleFieldName]);
    if (dynamicTitle !== null) {
      return dynamicTitle;
    }
  }
  const fallback = pickFirstScalar(raw, TITLE_FALLBACK_KEYS);
  if (fallback !== null) {
    return fallback;
  }
  return asNonEmptyScalar(raw.dataset_doc_uuid) ?? "Untitled";
}

function resolveContentFieldNames(raw: Record<string, unknown>): string[] {
  if (!Array.isArray(raw.content_field_names)) {
    return [];
  }
  const names: string[] = [];
  const seen = new Set<string>();
  for (const item of raw.content_field_names) {
    if (typeof item === "string" && item.trim().length > 0 && !seen.has(item)) {
      seen.add(item);
      names.push(item);
    }
  }
  return names;
}

export function renderCanonicalDocument(
  raw: Record<string, unknown>,
  source: string
): RenderedCorpusDocument {
  const datasetDocUuid = asNonEmptyScalar(raw.dataset_doc_uuid);
  const title = resolveTitle(raw);
  const author = pickFirstScalar(raw, AUTHOR_KEYS);
  const status = pickFirstScalar(raw, STATUS_KEYS);
  const createdAt = pickFirstScalar(raw, CREATED_AT_KEYS);
  const updatedAt = pickFirstScalar(raw, UPDATED_AT_KEYS);

  const headerCard = buildHeaderCard({
    datasetDocUuid,
    source,
    title,
    author,
    status,
    createdAt,
    updatedAt
  });

  const contentFields = resolveContentFieldNames(raw);
  const contentFieldSet = new Set(contentFields);
  const primarySections: string[] = [];
  const renderedKeys = new Set<string>(CONTROL_KEYS);

  for (const fieldName of contentFields) {
    renderedKeys.add(fieldName);
    const rendered = renderValue(raw[fieldName]);
    if (rendered.length > 0) {
      primarySections.push(`## ${fieldName}\n${rendered}`);
    }
  }

  const metadataPairs: string[] = [`source: ${source}`];
  const metadataRecord: Record<string, unknown> = {};
  const extraSections: string[] = [];

  for (const [key, value] of Object.entries(raw)) {
    if (CONTROL_KEYS.has(key) || contentFieldSet.has(key)) {
      continue;
    }
    const rendered = renderValue(value);
    if (rendered.length === 0) {
      continue;
    }

    const isCompactMetadata =
      typeof value === "string"
        ? value.trim().length <= 240 && !value.includes("\n")
        : typeof value === "number" ||
          typeof value === "boolean" ||
          (Array.isArray(value) &&
            value.every(
              (item) =>
                typeof item === "string" ||
                typeof item === "number" ||
                typeof item === "boolean"
            ) &&
            rendered.length <= 240);

    if (isCompactMetadata) {
      metadataRecord[key] = value;
      metadataPairs.push(`${key}: ${rendered}`);
    }

    if (!renderedKeys.has(key)) {
      renderedKeys.add(key);
      if (isCompactMetadata) {
        extraSections.push(`${key}: ${rendered}`);
      } else {
        extraSections.push(`## ${key}\n${rendered}`);
      }
    }
  }

  const bodyParts: string[] = [...primarySections];
  if (extraSections.length > 0) {
    bodyParts.push(extraSections.join("\n"));
  }

  const bodyText = bodyParts.join("\n\n").trim();
  const metadataText = metadataPairs.join("\n").trim();
  const fullText = bodyText.length > 0 ? `${headerCard}\n\n${bodyText}` : headerCard;

  return {
    title,
    author,
    status,
    createdAt,
    updatedAt,
    headerCard,
    metadataText,
    bodyText,
    fullText,
    metadataRecord
  };
}

function splitLongLine(line: string, maxChars: number): string[] {
  const parts: string[] = [];
  let cursor = 0;

  while (cursor < line.length) {
    const remaining = line.length - cursor;
    if (remaining <= maxChars) {
      parts.push(line.slice(cursor).trim());
      break;
    }

    let splitAt = line.lastIndexOf(" ", cursor + maxChars);
    if (splitAt <= cursor) {
      splitAt = cursor + maxChars;
    }
    const piece = line.slice(cursor, splitAt).trim();
    if (piece.length > 0) {
      parts.push(piece);
    }
    cursor = splitAt;
    while (cursor < line.length && line[cursor] === " ") {
      cursor += 1;
    }
  }

  return parts.filter((part) => part.length > 0);
}

function splitIntoStructuralUnits(text: string, maxUnitChars: number): string[] {
  const rawBlocks = text
    .split(/\n{2,}|(?=\n#{1,6}\s)/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);

  const units: string[] = [];
  for (const block of rawBlocks) {
    if (block.length <= maxUnitChars) {
      units.push(block);
      continue;
    }

    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    for (const line of lines) {
      if (line.length <= maxUnitChars) {
        units.push(line);
      } else {
        units.push(...splitLongLine(line, maxUnitChars));
      }
    }
  }

  return units;
}

function extractOverlapTail(text: string, overlapChars: number): string {
  if (overlapChars <= 0 || text.length === 0) {
    return "";
  }
  if (text.length <= overlapChars) {
    return text.trim();
  }
  const start = text.length - overlapChars;
  const newlineIndex = text.indexOf("\n", start);
  if (newlineIndex >= start && newlineIndex < text.length - 1) {
    return text.slice(newlineIndex + 1).trim();
  }
  const spaceIndex = text.indexOf(" ", start);
  if (spaceIndex >= start && spaceIndex < text.length - 1) {
    return text.slice(spaceIndex + 1).trim();
  }
  return text.slice(start).trim();
}

export function chunkDocumentText(
  input: { headerCard: string; bodyText: string },
  options: ChunkOptions
): string[] {
  const header = input.headerCard.trim();
  const body = input.bodyText.trim();
  const prefix =
    options.includeHeaderCard && header.length > 0 ? `${header}\n\n` : "";

  const wholeText = body.length > 0 ? `${prefix}${body}` : header;
  if (wholeText.length === 0) {
    return [""];
  }
  if (wholeText.length <= options.shortDocMaxChars || body.length === 0) {
    return [wholeText];
  }

  const bodyBudget =
    prefix.length < options.chunkTargetChars
      ? Math.max(1, options.chunkTargetChars - prefix.length)
      : Math.max(1, options.chunkTargetChars);
  const overlapBudget = Math.min(
    options.chunkOverlapChars,
    Math.floor(bodyBudget / 2)
  );

  const units = splitIntoStructuralUnits(body, bodyBudget);
  const chunks: string[] = [];
  let current = "";

  for (const unit of units) {
    if (current.length === 0) {
      current = unit;
      continue;
    }

    const candidate = `${current}\n\n${unit}`;
    if (candidate.length <= bodyBudget) {
      current = candidate;
      continue;
    }

    chunks.push(`${prefix}${current}`.trim());

    const overlap = extractOverlapTail(current, overlapBudget);
    const withOverlap = overlap.length > 0 ? `${overlap}\n\n${unit}` : unit;
    current = withOverlap.length <= bodyBudget ? withOverlap : unit;
  }

  if (current.length > 0) {
    chunks.push(`${prefix}${current}`.trim());
  }

  return chunks;
}

function selectFtsTokens(rawQuestion: string): string[] {
  const sanitized = rawQuestion.replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  if (sanitized.length === 0) {
    return [];
  }

  const rawTokens = sanitized.split(/\s+/);
  const nonOperatorTokens: string[] = [];
  const seenAll = new Set<string>();

  for (const token of rawTokens) {
    const lower = token.toLowerCase();
    if (lower.length === 0 || FTS_OPERATOR_WORDS.has(lower)) {
      continue;
    }
    if (!seenAll.has(lower)) {
      seenAll.add(lower);
      nonOperatorTokens.push(lower);
    }
  }

  if (nonOperatorTokens.length === 0) {
    return [];
  }

  const contentTokens = nonOperatorTokens.filter(
    (token) => !STOP_WORDS.has(token) && token.length >= 2
  );
  return (contentTokens.length > 0 ? contentTokens : nonOperatorTokens).slice(
    0,
    MAX_FTS_TERMS
  );
}

export function buildSafeFtsQuery(rawQuestion: string): string {
  const selected = selectFtsTokens(rawQuestion);
  if (selected.length === 0) {
    return "";
  }
  return selected.map((token) => `"${token}"`).join(" OR ");
}

export function buildFtsQueryCandidates(rawQuestion: string): string[] {
  const selected = selectFtsTokens(rawQuestion);
  if (selected.length === 0) {
    return [];
  }
  if (selected.length === 1) {
    return [`"${selected[0]}"`];
  }

  const andTokens =
    selected.length <= MAX_FTS_AND_TERMS
      ? selected
      : selected
          .map((token, index) => ({ token, index }))
          .sort((a, b) => b.token.length - a.token.length || a.index - b.index)
          .slice(0, MAX_FTS_AND_TERMS)
          .sort((a, b) => a.index - b.index)
          .map((entry) => entry.token);

  const strictAnd = andTokens.map((token) => `"${token}"`).join(" AND ");
  const relaxedOr = selected.map((token) => `"${token}"`).join(" OR ");
  return [strictAnd, relaxedOr];
}
