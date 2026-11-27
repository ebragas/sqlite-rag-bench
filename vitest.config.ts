import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/utils/**/*.ts", "src/models/**/*.ts", "src/usecases/**/*.ts"],
      reporter: ["text", "json-summary"],
      thresholds: {
        lines: 85,
        functions: 85
      }
    }
  }
});
