import { describe, expect, it } from "vitest";
import { ConfigError } from "../../src/models/errors.js";
import { parseRuntimeEnv } from "../../src/services/env.js";

describe("parseRuntimeEnv", () => {
  it("parses valid docs/api-contract.md environment variables", () => {
    const parsed = parseRuntimeEnv({
      CORPUS_DIR: "/tmp/corpus",
      PORT: "8080",
      GEMINI_API_KEY: "test-key"
    });

    expect(parsed).toEqual({
      CORPUS_DIR: "/tmp/corpus",
      PORT: 8080,
      GEMINI_API_KEY: "test-key"
    });
  });

  it("fails closed with ConfigError when required variables are missing or invalid", () => {
    expect(() => parseRuntimeEnv({})).toThrow(ConfigError);
    expect(() =>
      parseRuntimeEnv({
        CORPUS_DIR: "/tmp/corpus",
        PORT: "not-a-port",
        GEMINI_API_KEY: "test-key"
      })
    ).toThrow(ConfigError);
  });
});
