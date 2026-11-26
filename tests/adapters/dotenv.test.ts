import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  loadDotEnvIfPresent,
  parseDotEnvContent
} from "../../src/adapters/dotenv.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("parseDotEnvContent", () => {
  it("parses KEY=value pairs, comments, blank lines, quotes, and export prefix", () => {
    const raw = [
      "# Top comment",
      "",
      "GEMINI_API_KEY=plain-key-123",
      "export LANGFUSE_BASE_URL=https://us.cloud.langfuse.com # inline note",
      'QUOTED_DOUBLE="hello\\nworld\\t\\""',
      "QUOTED_SINGLE='literal # not a comment'",
      "EMPTY_VALUE=",
      "INVALID-KEY=ignored",
      "NO_EQUALS_LINE"
    ].join("\n");

    expect(parseDotEnvContent(raw)).toEqual({
      GEMINI_API_KEY: "plain-key-123",
      LANGFUSE_BASE_URL: "https://us.cloud.langfuse.com",
      QUOTED_DOUBLE: 'hello\nworld\t"',
      QUOTED_SINGLE: "literal # not a comment",
      EMPTY_VALUE: ""
    });
  });
});

describe("loadDotEnvIfPresent", () => {
  it("loads missing variables without overwriting already-set env keys", () => {
    const root = mkdtempSync(join(tmpdir(), "srb-dotenv-"));
    tempDirs.push(root);
    const envPath = join(root, ".env");
    writeFileSync(
      envPath,
      [
        "GEMINI_API_KEY=from-dotenv",
        "LANGFUSE_PUBLIC_KEY=pk-from-dotenv",
        "LANGFUSE_SECRET_KEY=sk-from-dotenv"
      ].join("\n"),
      "utf8"
    );

    const targetEnv: NodeJS.ProcessEnv = {
      GEMINI_API_KEY: "already-set-in-shell"
    };

    const result = loadDotEnvIfPresent(envPath, targetEnv);
    expect(result).toEqual({
      loaded: true,
      path: envPath,
      appliedKeys: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    });
    expect(targetEnv.GEMINI_API_KEY).toBe("already-set-in-shell");
    expect(targetEnv.LANGFUSE_PUBLIC_KEY).toBe("pk-from-dotenv");
    expect(targetEnv.LANGFUSE_SECRET_KEY).toBe("sk-from-dotenv");
  });

  it("returns loaded=false cleanly when the .env file does not exist", () => {
    const root = mkdtempSync(join(tmpdir(), "srb-dotenv-missing-"));
    tempDirs.push(root);
    const missingPath = join(root, ".env");
    const targetEnv: NodeJS.ProcessEnv = {};

    const result = loadDotEnvIfPresent(missingPath, targetEnv);
    expect(result).toEqual({
      loaded: false,
      path: missingPath,
      appliedKeys: []
    });
    expect(Object.keys(targetEnv)).toEqual([]);
  });
});
