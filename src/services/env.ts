import { z } from "zod";
import { ConfigError } from "../models/errors.js";

const RuntimeEnvSchema = z.object({
  CORPUS_DIR: z.string().min(1),
  PORT: z.coerce.number().int().min(1).max(65535),
  GEMINI_API_KEY: z.string().min(1)
});

export type RuntimeEnv = z.infer<typeof RuntimeEnvSchema>;

export function parseRuntimeEnv(env: NodeJS.ProcessEnv): RuntimeEnv {
  const result = RuntimeEnvSchema.safeParse({
    CORPUS_DIR: env.CORPUS_DIR,
    PORT: env.PORT,
    GEMINI_API_KEY: env.GEMINI_API_KEY
  });

  if (!result.success) {
    throw new ConfigError(`Invalid runtime environment: ${result.error.message}`);
  }
  return result.data;
}
