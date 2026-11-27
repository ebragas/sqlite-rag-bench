import { z } from "zod";
import { sha256Hex } from "../utils/hash.js";
import { ConfigError } from "./errors.js";

const DEFAULT_CHUNK_STRATEGY = {
  shortDocMaxChars: 6000,
  chunkTargetChars: 4000,
  chunkOverlapChars: 600,
  includeHeaderCard: true
} as const;

export const ChunkStrategySchema = z
  .object({
    shortDocMaxChars: z.number().int().positive().default(DEFAULT_CHUNK_STRATEGY.shortDocMaxChars),
    chunkTargetChars: z.number().int().positive().default(DEFAULT_CHUNK_STRATEGY.chunkTargetChars),
    chunkOverlapChars: z
      .number()
      .int()
      .nonnegative()
      .default(DEFAULT_CHUNK_STRATEGY.chunkOverlapChars),
    includeHeaderCard: z.boolean().default(DEFAULT_CHUNK_STRATEGY.includeHeaderCard)
  })
  .strict()
  .refine((value) => value.chunkOverlapChars < value.chunkTargetChars, {
    message: "chunkOverlapChars must be smaller than chunkTargetChars",
    path: ["chunkOverlapChars"]
  });

export type ChunkStrategy = z.infer<typeof ChunkStrategySchema>;

export const AgentToolSchema = z.enum([
  "search",
  "fetch_docs",
  "filter_metadata",
  "expand_graph"
]);

export type AgentTool = z.infer<typeof AgentToolSchema>;

export const RunProfileSchema = z
  .object({
    name: z.string().min(1),
    chunkStrategy: ChunkStrategySchema.default({ ...DEFAULT_CHUNK_STRATEGY }),
    enableEmbeddings: z.boolean().default(false),
    embeddingModel: z.string().min(1).default("gemini-embedding-001"),
    embeddingDims: z.number().int().positive().default(768),
    topK: z.number().int().positive().default(15),
    preSearchEnabled: z.boolean().default(false),
    enabledTools: z.array(AgentToolSchema).default([]),
    maxAgentTurns: z.number().int().positive().default(6),
    maxFetchedDocs: z.number().int().positive().default(15),
    softDeadlineMs: z.number().int().positive().default(90000),
    routingModel: z.string().min(1).default("gemini-3.5-flash-lite"),
    synthesisModel: z.string().min(1).default("gemini-3.8-flash"),
    enableAbstentionGate: z.boolean().default(false),
    enableSecondPassPruner: z.boolean().default(false)
  })
  .strict();

export type RunProfile = z.infer<typeof RunProfileSchema>;
export type RunProfileInput = z.input<typeof RunProfileSchema>;

export type IndexConfig = Pick<
  RunProfile,
  "chunkStrategy" | "enableEmbeddings" | "embeddingModel" | "embeddingDims"
>;

export function computeIndexConfigHash(profile: IndexConfig): string {
  const canonical = JSON.stringify({
    chunkStrategy: {
      shortDocMaxChars: profile.chunkStrategy.shortDocMaxChars,
      chunkTargetChars: profile.chunkStrategy.chunkTargetChars,
      chunkOverlapChars: profile.chunkStrategy.chunkOverlapChars,
      includeHeaderCard: profile.chunkStrategy.includeHeaderCard
    },
    enableEmbeddings: profile.enableEmbeddings,
    embeddingModel: profile.embeddingModel,
    embeddingDims: profile.embeddingDims
  });
  return sha256Hex(canonical);
}

const FORBIDDEN_PATH_SEGMENTS = new Set(["__proto__", "constructor", "prototype"]);
const NUMERIC_LITERAL_PATTERN = /^-?\d+(?:\.\d+)?$/;

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseOverrideValue(rawValue: string, keyPath: string): unknown {
  const trimmed = rawValue.trim();
  if (trimmed === "true") {
    return true;
  }
  if (trimmed === "false") {
    return false;
  }
  if (NUMERIC_LITERAL_PATTERN.test(trimmed)) {
    return Number(trimmed);
  }
  if (
    trimmed.startsWith("[") ||
    trimmed.endsWith("]") ||
    trimmed.startsWith("{") ||
    trimmed.endsWith("}") ||
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
  ) {
    try {
      return JSON.parse(trimmed);
    } catch (error) {
      throw new ConfigError(`Invalid JSON override value for "${keyPath}": ${rawValue}`, {
        cause: error
      });
    }
  }
  return rawValue;
}

function splitOverridePath(keyPath: string): { parents: string[]; leaf: string } {
  const segments = keyPath.split(".");
  for (const segment of segments) {
    if (segment.length === 0 || FORBIDDEN_PATH_SEGMENTS.has(segment)) {
      throw new ConfigError(`Invalid override key segment in "${keyPath}"`);
    }
  }
  const leaf = segments.pop();
  if (leaf === undefined) {
    throw new ConfigError(`Invalid override key segment in "${keyPath}"`);
  }
  return { parents: segments, leaf };
}

export function parseOverrideAssignments(entries: readonly string[]): Record<string, unknown> {
  const root: Record<string, unknown> = {};

  for (const entry of entries) {
    const equalsIndex = entry.indexOf("=");
    if (equalsIndex <= 0) {
      throw new ConfigError(`Override must use key=value format: "${entry}"`);
    }

    const keyPath = entry.slice(0, equalsIndex).trim();
    const rawValue = entry.slice(equalsIndex + 1);
    const { parents, leaf } = splitOverridePath(keyPath);
    const parsedValue = parseOverrideValue(rawValue, keyPath);

    let cursor: Record<string, unknown> = root;
    for (const segment of parents) {
      const existing = cursor[segment];
      if (existing === undefined) {
        const next: Record<string, unknown> = {};
        cursor[segment] = next;
        cursor = next;
      } else if (isPlainRecord(existing)) {
        cursor = existing;
      } else {
        throw new ConfigError(`Conflicting override path at "${keyPath}"`);
      }
    }

    if (isPlainRecord(cursor[leaf]) && !isPlainRecord(parsedValue)) {
      throw new ConfigError(`Conflicting override path at "${keyPath}"`);
    }
    cursor[leaf] = parsedValue;
  }

  return root;
}

function mergeOverrides(
  base: Record<string, unknown>,
  patch: Record<string, unknown>,
  prefix = ""
): Record<string, unknown> {
  const result: Record<string, unknown> = { ...base };

  for (const [key, patchValue] of Object.entries(patch)) {
    const pathLabel = prefix.length > 0 ? `${prefix}.${key}` : key;
    if (!(key in base)) {
      throw new ConfigError(`Unknown run profile override key: "${pathLabel}"`);
    }

    const baseValue = base[key];
    if (isPlainRecord(patchValue)) {
      if (!isPlainRecord(baseValue)) {
        throw new ConfigError(`Cannot assign nested keys under scalar "${pathLabel}"`);
      }
      result[key] = mergeOverrides(baseValue, patchValue, pathLabel);
    } else {
      result[key] = patchValue;
    }
  }

  return result;
}

export function applyProfileOverrides(
  profile: RunProfile,
  overrides: readonly string[]
): RunProfile {
  if (overrides.length === 0) {
    return profile;
  }

  const parsedPatch = parseOverrideAssignments(overrides);
  const baseRecord: Record<string, unknown> = {
    ...profile,
    chunkStrategy: { ...profile.chunkStrategy }
  };
  const merged = mergeOverrides(baseRecord, parsedPatch);
  const validated = RunProfileSchema.safeParse(merged);

  if (!validated.success) {
    throw new ConfigError(`Invalid run profile override: ${validated.error.message}`);
  }
  return validated.data;
}
