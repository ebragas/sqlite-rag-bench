import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { readJsonlFile } from "../src/adapters/jsonl.js";
import { ConfigError } from "../src/models/errors.js";
import { DevQuestionSchema, SplitManifestSchema } from "../src/models/eval.js";
import { buildStratifiedSplits } from "../src/usecases/splits.js";

function parseArgs(argv: readonly string[]): {
  seed: number;
  preserveHoldout: boolean;
  questionsPath: string;
  outPath: string;
} {
  let seed = 20261125;
  let preserveHoldout = false;
  let questionsPath = resolve("data/questions.jsonl");
  let outPath = resolve("eval/splits.json");

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--seed") {
      const value = Number(argv[i + 1]);
      if (!Number.isInteger(value)) {
        throw new ConfigError(`--seed requires an integer, got: ${argv[i + 1] ?? ""}`);
      }
      seed = value;
      i += 1;
    } else if (arg === "--preserve-holdout") {
      preserveHoldout = true;
    } else if (arg === "--questions") {
      const next = argv[i + 1];
      if (!next) throw new ConfigError("--questions requires a path");
      questionsPath = resolve(next);
      i += 1;
    } else if (arg === "--out") {
      const next = argv[i + 1];
      if (!next) throw new ConfigError("--out requires a path");
      outPath = resolve(next);
      i += 1;
    } else {
      throw new ConfigError(`Unknown argument: ${arg ?? ""}`);
    }
  }

  return { seed, preserveHoldout, questionsPath, outPath };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const questions = readJsonlFile(args.questionsPath, DevQuestionSchema);

  let lockedHoldoutIds: readonly string[] | undefined;
  let lockedHoldoutSeed: number | undefined;

  if (args.preserveHoldout && existsSync(args.outPath)) {
    const existingRaw = JSON.parse(readFileSync(args.outPath, "utf8")) as unknown;
    const existing = SplitManifestSchema.parse(existingRaw);
    lockedHoldoutIds = existing.holdout;
    lockedHoldoutSeed = existing.holdout_seed;
  }

  const splitOptions =
    lockedHoldoutIds !== undefined && lockedHoldoutSeed !== undefined
      ? {
          seed: args.seed,
          lockedHoldoutIds,
          lockedHoldoutSeed
        }
      : { seed: args.seed };

  const manifest = buildStratifiedSplits(questions, splitOptions);

  mkdirSync(dirname(args.outPath), { recursive: true });
  writeFileSync(args.outPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(
    `Wrote ${args.outPath} (seed=${manifest.seed}, holdout_seed=${manifest.holdout_seed}): ` +
      `dev=${manifest.dev.length}, val=${manifest.val.length}, holdout=${manifest.holdout.length}`
  );
}

main();
