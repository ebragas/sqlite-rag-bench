import { describe, expect, it } from "vitest";
import { ContractError, ValidationError } from "../../src/models/errors.js";
import type { DevQuestion } from "../../src/models/eval.js";
import {
  buildStratifiedSplits,
  classifyQuestionStratum
} from "../../src/usecases/splits.js";

function buildSyntheticQuestions(): DevQuestion[] {
  const specs: ReadonlyArray<{ readonly count: number; readonly docs: number }> = [
    { count: 12, docs: 0 },
    { count: 32, docs: 1 },
    { count: 12, docs: 2 },
    { count: 24, docs: 3 }
  ];
  const questions: DevQuestion[] = [];
  let index = 1;
  for (const spec of specs) {
    for (let i = 0; i < spec.count; i += 1) {
      const id = `q_${String(index).padStart(4, "0")}`;
      const expectedDocIds = Array.from(
        { length: spec.docs },
        (_, d) => `dsid_synth_${String(index)}_${String(d)}`
      );
      questions.push({
        question_id: id,
        question: `Synthetic benchmark question ${id}`,
        gold_answer: `Gold answer ${id}`,
        answer_facts: spec.docs === 0 ? [] : [`Fact for ${id}`],
        expected_doc_ids: expectedDocIds,
        source_types: spec.docs === 0 ? [] : ["confluence"]
      });
      index += 1;
    }
  }
  return questions;
}

describe("buildStratifiedSplits", () => {
  it("partitions all 80 synthetic questions into disjoint stratified dev, val, and holdout sets", () => {
    const questions = buildSyntheticQuestions();
    const manifest = buildStratifiedSplits(questions, { seed: 20261125 });

    const allIds = [...manifest.dev, ...manifest.val, ...manifest.holdout];
    const uniqueIds = new Set(allIds);

    expect(manifest.total_questions).toBe(80);
    expect(allIds.length).toBe(80);
    expect(uniqueIds.size).toBe(80);

    expect(manifest.counts.holdout.unanswerable).toBeGreaterThanOrEqual(3);
    expect(manifest.counts.val.unanswerable).toBeGreaterThanOrEqual(2);
    expect(manifest.counts.dev.unanswerable).toBeGreaterThanOrEqual(5);

    expect(manifest.counts.holdout.multi_doc).toBeGreaterThanOrEqual(6);
    expect(manifest.counts.val.multi_doc).toBeGreaterThanOrEqual(4);
    expect(manifest.counts.dev.multi_doc).toBeGreaterThanOrEqual(10);
  });

  it("produces identical splits for the same seed", () => {
    const questions = buildSyntheticQuestions();
    const first = buildStratifiedSplits(questions, { seed: 42 });
    const second = buildStratifiedSplits(questions, { seed: 42 });

    expect(first).toEqual(second);
  });

  it("re-seeds dev and val while keeping locked holdout ids unchanged", () => {
    const questions = buildSyntheticQuestions();
    const base = buildStratifiedSplits(questions, { seed: 20261125 });
    const reseeded = buildStratifiedSplits(questions, {
      seed: 999999,
      lockedHoldoutIds: base.holdout,
      lockedHoldoutSeed: base.holdout_seed
    });

    expect(reseeded.holdout).toEqual(base.holdout);
    expect(reseeded.holdout_seed).toBe(base.holdout_seed);
    expect(reseeded.dev).not.toEqual(base.dev);
    expect(reseeded.val).not.toEqual(base.val);
  });

  it("classifies question strata by expected_doc_ids length", () => {
    const base = {
      question_id: "q",
      question: "test",
      gold_answer: "ans",
      answer_facts: [],
      source_types: []
    };

    expect(classifyQuestionStratum({ ...base, expected_doc_ids: [] })).toBe("unanswerable");
    expect(classifyQuestionStratum({ ...base, expected_doc_ids: ["a"] })).toBe("single_doc");
    expect(classifyQuestionStratum({ ...base, expected_doc_ids: ["a", "b"] })).toBe("two_doc");
    expect(classifyQuestionStratum({ ...base, expected_doc_ids: ["a", "b", "c"] })).toBe("multi_doc");
  });

  it("throws ValidationError on duplicate ids, bad ratios, or unknown locked holdout ids", () => {
    const questions = buildSyntheticQuestions();
    const firstQuestion = questions[0];
    expect(firstQuestion).toBeDefined();

    expect(() =>
      buildStratifiedSplits([firstQuestion!, firstQuestion!], { seed: 1 })
    ).toThrow(ValidationError);

    expect(() =>
      buildStratifiedSplits(questions, { seed: 1, holdoutRatio: 0.6 })
    ).toThrow(ValidationError);

    expect(() =>
      buildStratifiedSplits(questions, { seed: 1, valRatio: 0 })
    ).toThrow(ValidationError);

    expect(() =>
      buildStratifiedSplits(questions, {
        seed: 1,
        lockedHoldoutIds: ["does_not_exist"]
      })
    ).toThrow(ValidationError);

    const err = new ContractError("bad response");
    expect(err.code).toBe("CONTRACT_ERROR");
    expect(err.name).toBe("ContractError");
  });
});
