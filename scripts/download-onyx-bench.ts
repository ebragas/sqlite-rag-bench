import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ONYX_QUESTIONS_URL =
  "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/questions.jsonl";

async function main(): Promise<void> {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = resolve(scriptDir, "..");
  const dataDir = resolve(repoRoot, "data");
  mkdirSync(dataDir, { recursive: true });

  console.log(`[download:onyx] Downloading 500 MIT-licensed questions from ${ONYX_QUESTIONS_URL}`);
  const response = await fetch(ONYX_QUESTIONS_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch Onyx questions.jsonl: HTTP ${String(response.status)}`);
  }
  const text = await response.text();
  const questionsPath = resolve(dataDir, "questions.jsonl");
  writeFileSync(questionsPath, text, "utf8");
  const lineCount = text.trim().split("\n").filter(Boolean).length;
  console.log(`[download:onyx] Saved ${String(lineCount)} questions to ${questionsPath}`);
  console.log("");
  console.log("Next step: download the MIT-licensed EnterpriseRAG-Bench corpus archives from:");
  console.log("  https://github.com/onyx-dot-app/EnterpriseRAG-Bench");
  console.log("  https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench");
  console.log("Place the JSON source directories under data/corpus/<source>/ and run:");
  console.log('  PORT=8080 CORPUS_DIR="$PWD/data/corpus" GEMINI_API_KEY="..." ./serve');
}

void main().catch((error: unknown) => {
  console.error(`[download:onyx] ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
