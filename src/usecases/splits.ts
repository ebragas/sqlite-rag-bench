import { ValidationError } from "../models/errors.js";
import type {
  DevQuestion,
  QuestionStratum,
  SplitCounts,
  SplitManifest
} from "../models/eval.js";
import { deterministicShuffle } from "../utils/prng.js";

export interface SplitOptions {
  readonly seed: number;
  readonly holdoutRatio?: number;
  readonly valRatio?: number;
  readonly lockedHoldoutIds?: readonly string[];
  readonly lockedHoldoutSeed?: number;
}

const DEFAULT_HOLDOUT_RATIO = 0.3;
const DEFAULT_VAL_RATIO = 0.25;

export function classifyQuestionStratum(question: DevQuestion): QuestionStratum {
  const count = question.expected_doc_ids.length;
  if (count === 0) return "unanswerable";
  if (count === 1) return "single_doc";
  if (count === 2) return "two_doc";
  return "multi_doc";
}

function emptyCounts(): Record<QuestionStratum, number> {
  return {
    unanswerable: 0,
    single_doc: 0,
    two_doc: 0,
    multi_doc: 0
  };
}

function toSplitCounts(record: Record<QuestionStratum, number>): SplitCounts {
  return {
    unanswerable: record.unanswerable,
    single_doc: record.single_doc,
    two_doc: record.two_doc,
    multi_doc: record.multi_doc
  };
}

export function buildStratifiedSplits(
  questions: readonly DevQuestion[],
  options: SplitOptions
): SplitManifest {
  const holdoutRatio = options.holdoutRatio ?? DEFAULT_HOLDOUT_RATIO;
  const valRatio = options.valRatio ?? DEFAULT_VAL_RATIO;

  if (holdoutRatio <= 0 || holdoutRatio >= 0.5) {
    throw new ValidationError(`holdoutRatio must be between 0 and 0.5, got ${holdoutRatio}`);
  }
  if (valRatio <= 0 || valRatio >= 0.5) {
    throw new ValidationError(`valRatio must be between 0 and 0.5, got ${valRatio}`);
  }

  const byStratum: Record<QuestionStratum, DevQuestion[]> = {
    unanswerable: [],
    single_doc: [],
    two_doc: [],
    multi_doc: []
  };

  const seenIds = new Set<string>();
  for (const question of questions) {
    if (seenIds.has(question.question_id)) {
      throw new ValidationError(`Duplicate question_id: ${question.question_id}`);
    }
    seenIds.add(question.question_id);
    byStratum[classifyQuestionStratum(question)].push(question);
  }

  const lockedHoldoutSet = options.lockedHoldoutIds
    ? new Set(options.lockedHoldoutIds)
    : undefined;

  if (lockedHoldoutSet !== undefined) {
    for (const id of lockedHoldoutSet) {
      if (!seenIds.has(id)) {
        throw new ValidationError(`Locked holdout id not found in questions: ${id}`);
      }
    }
  }

  const dev: string[] = [];
  const val: string[] = [];
  const holdout: string[] = [];
  const devCounts = emptyCounts();
  const valCounts = emptyCounts();
  const holdoutCounts = emptyCounts();

  const strata: readonly QuestionStratum[] = [
    "unanswerable",
    "single_doc",
    "two_doc",
    "multi_doc"
  ];

  for (let stratumIndex = 0; stratumIndex < strata.length; stratumIndex += 1) {
    const stratum = strata[stratumIndex];
    if (stratum === undefined) {
      continue;
    }

    const sortedGroup = [...byStratum[stratum]].sort((a, b) =>
      a.question_id.localeCompare(b.question_id)
    );

    let stratumHoldout: DevQuestion[];
    let remaining: DevQuestion[];

    if (lockedHoldoutSet !== undefined) {
      stratumHoldout = sortedGroup.filter((q) => lockedHoldoutSet.has(q.question_id));
      remaining = deterministicShuffle(
        sortedGroup.filter((q) => !lockedHoldoutSet.has(q.question_id)),
        options.seed + stratumIndex * 997
      );
    } else {
      const shuffled = deterministicShuffle(
        sortedGroup,
        options.seed + stratumIndex * 997
      );
      const holdoutCount = Math.max(1, Math.round(shuffled.length * holdoutRatio));
      stratumHoldout = shuffled.slice(0, holdoutCount);
      remaining = shuffled.slice(holdoutCount);
    }

    const valCount = Math.max(1, Math.round(remaining.length * valRatio));
    const stratumVal = remaining.slice(0, valCount);
    const stratumDev = remaining.slice(valCount);

    for (const item of stratumHoldout) {
      holdout.push(item.question_id);
      holdoutCounts[stratum] += 1;
    }
    for (const item of stratumVal) {
      val.push(item.question_id);
      valCounts[stratum] += 1;
    }
    for (const item of stratumDev) {
      dev.push(item.question_id);
      devCounts[stratum] += 1;
    }
  }

  return {
    version: 1,
    seed: options.seed,
    holdout_seed: options.lockedHoldoutSeed ?? options.seed,
    total_questions: questions.length,
    counts: {
      dev: toSplitCounts(devCounts),
      val: toSplitCounts(valCounts),
      holdout: toSplitCounts(holdoutCounts)
    },
    dev: dev.sort(),
    val: val.sort(),
    holdout: holdout.sort()
  };
}
