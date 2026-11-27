import { readFileSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { ConfigError } from "../models/errors.js";
import {
  applyProfileOverrides,
  RunProfileSchema,
  type RunProfile
} from "../models/profile.js";

const DEFAULT_PROFILES_DIR = "eval/profiles";

function resolveProfilePath(nameOrPath: string, profilesDir?: string): string {
  const trimmed = nameOrPath.trim();
  if (trimmed.length === 0) {
    throw new ConfigError("Run profile name or path must not be empty");
  }

  const looksLikePath =
    isAbsolute(trimmed) ||
    trimmed.endsWith(".json") ||
    trimmed.includes("/") ||
    trimmed.includes("\\");

  if (looksLikePath) {
    return resolve(trimmed);
  }

  const baseDir = profilesDir ?? resolve(DEFAULT_PROFILES_DIR);
  return resolve(baseDir, `${trimmed}.json`);
}

export function loadRunProfile(
  nameOrPath: string,
  overrides: readonly string[] = [],
  profilesDir?: string
): RunProfile {
  const filePath = resolveProfilePath(nameOrPath, profilesDir);

  let raw: string;
  try {
    raw = readFileSync(filePath, "utf8");
  } catch (error) {
    throw new ConfigError(`Unable to read run profile at ${filePath}`, { cause: error });
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(raw);
  } catch (error) {
    throw new ConfigError(`Invalid JSON in run profile at ${filePath}`, { cause: error });
  }

  const validated = RunProfileSchema.safeParse(parsedJson);
  if (!validated.success) {
    throw new ConfigError(`Invalid run profile at ${filePath}: ${validated.error.message}`);
  }

  return applyProfileOverrides(validated.data, overrides);
}
