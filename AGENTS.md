# Agent & Engineering Guidelines

This file defines the principles, quality gates, and style standards for `sqlite-rag-bench`. `CLAUDE.md` symlinks here.

## Goal

Build and benchmark a local-first enterprise RAG engine (`./serve`) backed by single-file SQLite (FTS5 + `Float32Array` dense vectors + deterministic Link Graph), evaluated rigorously against open benchmarks (starting with MIT-licensed [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)) with Langfuse v4 OpenTelemetry tracing and `pass@k` / `pass^k` consistency testing.

## Engineering Principles

- **Evidence before complexity:** Start with the simplest working baseline (`l0-fts`). Add retrieval layers (metadata filters, dense embeddings, Link Graph traversal, Turn 0 pre-search) only when controlled ablations prove a real gain above Judge noise.
- **Abstention over fabrication:** No answer beats a wrong answer. When the corpus lacks evidence, state that clearly and return `document_ids: []`. Track both abstention recall and false-abstention rate.
- **Evaluation and spend honesty:** Separate tuning splits from untouched test sets. Track token and dollar spend on every run. Measure consistency (`pass^k`) alongside capability (`pass@k`). Report misses openly.
- **Incremental-ready storage:** Key every document, chunk, embedding, alias, and graph edge by `(dataset_doc_uuid, content_hash)` so live updates require small atomic deltas instead of full re-indexing.
- **Modest-hardware discipline:** Target CPU-only 8 to 16 GB Linux or macOS hosts (`<= 4 GB` peak RSS, `<= 30 min` cold ingest on ~21.6K docs, `<= 60 s` p95 latency at concurrency 4, plain Node 20+).
- **Own the core, keep SDKs thin:** Use Vercel AI SDK (`ai`) as thin multi-provider plumbing. Own ingestion, SQLite storage, deterministic graph extraction, retrieval tools, citation validation, and the eval harness directly.

## Mechanical Quality Gates (`npm run check`)

Every change must pass all four gates with zero errors:

1. `npm run lint` (`oxlint` + `@interfere/landmines` for test honesty and anti-slop rules)
2. `npm run typecheck` (`tsc --noEmit` with `strict`, `noUncheckedIndexedAccess`, zero `any`)
3. `npm run arch` (`dependency-cruiser` enforcing unidirectional 5-layer Clean Architecture)
4. `npm test` (`vitest` with real SQLite temp databases and real HTTP servers)

Rules enforced by convention and tooling:
- Keep functions small and flat (prefer early returns over deep nesting).
- Parse all external boundaries (corpus JSON, HTTP bodies, LLM structured outputs, env vars) with `zod`.
- Raise typed error classes from `src/models/errors.ts`. Never swallow errors silently.
- Keep secrets and `.env` files out of git.

## Voice: ASD-STE100

Write docs, comments, and commits in plain ASD-STE100 Simplified Technical English:
- Do not use the em-dash character (U+2014). Use commas, colons, parentheses, or separate sentences.
- Keep sentences short and active (20 words or fewer for instructions, 25 or fewer for descriptions).
- Use consistent terms from `GLOSSARY.md`.
