import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

export interface LoadDotEnvResult {
  readonly loaded: boolean;
  readonly path: string;
  readonly appliedKeys: readonly string[];
}

const ENV_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

function unescapeDoubleQuoted(raw: string): string {
  let result = "";
  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    if (ch === "\\" && i + 1 < raw.length) {
      const next = raw[i + 1];
      if (next === "n") {
        result += "\n";
        i += 1;
        continue;
      }
      if (next === "r") {
        result += "\r";
        i += 1;
        continue;
      }
      if (next === "t") {
        result += "\t";
        i += 1;
        continue;
      }
      if (next === '"' || next === "\\") {
        result += next;
        i += 1;
        continue;
      }
    }
    if (ch !== undefined) {
      result += ch;
    }
  }
  return result;
}

function parseDotEnvValue(rawValue: string): string {
  const trimmed = rawValue.trim();
  if (trimmed.length === 0) {
    return "";
  }

  if (trimmed.startsWith('"') && trimmed.endsWith('"') && trimmed.length >= 2) {
    return unescapeDoubleQuoted(trimmed.slice(1, -1));
  }

  if (trimmed.startsWith("'") && trimmed.endsWith("'") && trimmed.length >= 2) {
    return trimmed.slice(1, -1);
  }

  const commentIndex = trimmed.search(/\s+#/);
  if (commentIndex >= 0) {
    return trimmed.slice(0, commentIndex).trim();
  }
  return trimmed;
}

export function parseDotEnvContent(content: string): Record<string, string> {
  const parsed: Record<string, string> = {};
  const lines = content.split(/\r?\n/);

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) {
      continue;
    }

    const line = trimmed.startsWith("export ")
      ? trimmed.slice("export ".length).trimStart()
      : trimmed;
    const eqIndex = line.indexOf("=");
    if (eqIndex <= 0) {
      continue;
    }

    const key = line.slice(0, eqIndex).trim();
    if (!ENV_KEY_PATTERN.test(key)) {
      continue;
    }

    const value = parseDotEnvValue(line.slice(eqIndex + 1));
    parsed[key] = value;
  }

  return parsed;
}

export function loadDotEnvIfPresent(
  envPath = resolve(process.cwd(), ".env"),
  targetEnv: NodeJS.ProcessEnv = process.env
): LoadDotEnvResult {
  const resolvedPath = resolve(envPath);
  if (!existsSync(resolvedPath)) {
    return {
      loaded: false,
      path: resolvedPath,
      appliedKeys: []
    };
  }

  const content = readFileSync(resolvedPath, "utf8");
  const parsed = parseDotEnvContent(content);
  const appliedKeys: string[] = [];

  for (const [key, value] of Object.entries(parsed)) {
    if (targetEnv[key] === undefined) {
      targetEnv[key] = value;
      appliedKeys.push(key);
    }
  }

  return {
    loaded: true,
    path: resolvedPath,
    appliedKeys
  };
}
