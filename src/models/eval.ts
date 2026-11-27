import { z } from "zod";

export const DevQuestionSchema = z.object({
  question_id: z.string().min(1),
  question: z.string().min(1),
  gold_answer: z.string().min(1),
  answer_facts: z.array(z.string()),
  expected_doc_ids: z.array(z.string()),
  source_types: z.array(z.string())
});

export type DevQuestion = z.infer<typeof DevQuestionSchema>;

export const QuestionStratumSchema = z.enum([
  "unanswerable",
  "single_doc",
  "two_doc",
  "multi_doc"
]);

export type QuestionStratum = z.infer<typeof QuestionStratumSchema>;

export const SplitCountsSchema = z.object({
  unanswerable: z.number().int().nonnegative(),
  single_doc: z.number().int().nonnegative(),
  two_doc: z.number().int().nonnegative(),
  multi_doc: z.number().int().nonnegative()
});

export type SplitCounts = z.infer<typeof SplitCountsSchema>;

export const SplitManifestSchema = z.object({
  version: z.literal(1),
  seed: z.number().int(),
  holdout_seed: z.number().int(),
  total_questions: z.number().int().positive(),
  counts: z.object({
    dev: SplitCountsSchema,
    val: SplitCountsSchema,
    holdout: SplitCountsSchema
  }),
  dev: z.array(z.string().min(1)),
  val: z.array(z.string().min(1)),
  holdout: z.array(z.string().min(1))
});

export type SplitManifest = z.infer<typeof SplitManifestSchema>;
