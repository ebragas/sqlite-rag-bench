import { readFileSync } from "node:fs";
import type { ZodType } from "zod";
import { ValidationError } from "../models/errors.js";

export function readJsonlFile<T>(filePath: string, schema: ZodType<T>): T[] {
  const raw = readFileSync(filePath, "utf8");
  const lines = raw.split(/\r?\n/);
  const items: T[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const trimmed = (lines[index] ?? "").trim();
    if (trimmed.length === 0) {
      continue;
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(trimmed);
    } catch (error) {
      throw new ValidationError(
        `Invalid JSON at ${filePath}:${index + 1}`,
        { cause: error }
      );
    }

    const result = schema.safeParse(parsedJson);
    if (!result.success) {
      throw new ValidationError(
        `Schema validation failed at ${filePath}:${index + 1}: ${result.error.message}`
      );
    }
    items.push(result.data);
  }

  return items;
}
