import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import {
  IngestionError,
  RawCorpusDocumentSchema,
  type RawCorpusDocument
} from "../models/corpus.js";
import { sha256Hex } from "../utils/hash.js";

export interface CorpusFileRecord {
  filePath: string;
  relativePath: string;
  source: string;
  contentHash: string;
  raw: RawCorpusDocument;
}

export interface CorpusReader {
  readCorpus(signal?: AbortSignal): AsyncGenerator<CorpusFileRecord>;
}

function checkAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new IngestionError("Corpus ingestion aborted", {
      cause: signal.reason
    });
  }
}

function inferSourceFromRelativePath(relativePath: string): string {
  const segments = relativePath.split(sep).filter((segment) => segment.length > 0);
  if (segments.length <= 1) {
    return "root";
  }
  return segments[0] ?? "root";
}

async function* walkJsonFiles(
  currentDir: string,
  signal?: AbortSignal
): AsyncGenerator<string> {
  checkAborted(signal);

  let entries;
  try {
    entries = await readdir(currentDir, { withFileTypes: true });
  } catch (error) {
    throw new IngestionError(`Failed to read corpus directory: ${currentDir}`, {
      cause: error
    });
  }

  entries.sort((left, right) => left.name.localeCompare(right.name));

  for (const entry of entries) {
    checkAborted(signal);
    const fullPath = join(currentDir, entry.name);
    if (entry.isDirectory()) {
      yield* walkJsonFiles(fullPath, signal);
    } else if (entry.isFile() && entry.name.endsWith(".json")) {
      yield fullPath;
    }
  }
}

export async function* readCorpusDirectory(
  corpusDir: string,
  signal?: AbortSignal
): AsyncGenerator<CorpusFileRecord> {
  checkAborted(signal);

  let stats;
  try {
    stats = await stat(corpusDir);
  } catch (error) {
    throw new IngestionError(`Corpus directory does not exist: ${corpusDir}`, {
      cause: error
    });
  }

  if (!stats.isDirectory()) {
    throw new IngestionError(`Corpus path is not a directory: ${corpusDir}`);
  }

  for await (const filePath of walkJsonFiles(corpusDir, signal)) {
    checkAborted(signal);

    let rawText: string;
    try {
      rawText = await readFile(filePath, "utf8");
    } catch (error) {
      throw new IngestionError(`Failed to read corpus file: ${filePath}`, {
        cause: error
      });
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(rawText);
    } catch (error) {
      throw new IngestionError(`Invalid JSON in corpus file: ${filePath}`, {
        cause: error
      });
    }

    const validated = RawCorpusDocumentSchema.safeParse(parsedJson);
    if (!validated.success) {
      throw new IngestionError(
        `Invalid corpus document schema in ${filePath}: ${validated.error.message}`
      );
    }

    const relativePath = relative(corpusDir, filePath);
    yield {
      filePath,
      relativePath,
      source: inferSourceFromRelativePath(relativePath),
      contentHash: sha256Hex(rawText),
      raw: validated.data
    };
  }
}

export function createFilesystemCorpusReader(corpusDir: string): CorpusReader {
  return {
    readCorpus(signal?: AbortSignal): AsyncGenerator<CorpusFileRecord> {
      return readCorpusDirectory(corpusDir, signal);
    }
  };
}
