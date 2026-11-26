# Stack Decision: Python vs. TypeScript

> Note: This is our Day-0 stack decision snapshot. For final pinned package versions (`better-sqlite3@12.9.0`) and verified Docker metrics, see `package.json` and `README.md`.

This document records our weighted tool-selection analysis before writing application code. Each row awards `+1` (minor edge), `+2` (moderate edge), `+3` (major edge), or `0` (tie), grounded in primary-source research (`docs/rag-research.md`) and `docs/api-contract.md`.

## Decision Outcome

**Chosen stack:** TypeScript on plain `node` (20+) with `npm`, `better-sqlite3` (FTS5), in-memory `Float32Array` exact cosine search, Vercel AI SDK (`ai`), `zod`, `oxlint` + `@interfere/landmines`, `dependency-cruiser`, and `vitest`.

We removed the unused starter `pyproject.toml` and `.python-version` files so the repository has one unambiguous stack.

## Weighted Comparison Matrix

| Criterion | Why It Matters | Python 3.12 (`uv`) | TypeScript (`node 20+`) | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Contract & fresh-clone safety** | `./serve` must run unattended on Linux or macOS with zero native build surprises. | `pyproject.toml` shipped; stdlib `sqlite3` includes FTS5 everywhere. | `node 20+` guaranteed (not `bun`). Built-in `node:sqlite` lacks FTS5, so we use `better-sqlite3` (N-API prebuilds on macOS and glibc Linux) and verify in `node:20-slim` Docker. | **Python +2** |
| **Anti-slop AI harness** | `README.md` explicitly states reviewers read how you drive coding agents. | `ruff`, `mypy`, `pyright`, `import-linter`, `pytest-cov` (custom Python test-honesty rules needed). | Native `@interfere/landmines` + `oxlint` + `tsc --strict` + `dependency-cruiser` + `vitest`. | **TypeScript +2** |
| **Multi-provider plumbing** | Strict path uses `GEMINI_API_KEY`; dev/bonus path tests local Ollama (`clef-flash`, `clef:27b`), OpenRouter, or Vercel AI Gateway. | Custom `google-genai` + `openai` (`base_url`) adapters (~40 lines). | **Vercel AI SDK (`ai`)** gives unified typed structured outputs, tool calling, and embeddings across Gemini, Ollama, OpenRouter, and gateways. | **TypeScript +2** |
| **Human review & debug velocity** | Tight time box (~8 to 10 hours); fast human review catches subtle bugs early. | Good with agents; less daily muscle memory. | Matches daily TypeScript fluency and strict type-system instincts. | **TypeScript +2** |
| **SOTA RAG / GraphRAG / reranker need** | Research (`docs/rag-research.md`) proved SOTA at 21.6K docs is SQLite FTS5 + Gemini embeddings + deterministic JSON link graph + search-first bounded agent loop. Small CPU cross-encoders hurt multi-hop (-8.7 pts); LLM GraphRAG fails cost, time, and factual QA. | Strong if local CPU rerankers or `networkx` were needed, but research ruled both out. | Equally strong: `better-sqlite3` FTS5 + `Float32Array` exact cosine (<15 ms, 234 MiB RAM) + SQLite link tables + optional 40-line CSR Personalized PageRank + Vercel AI SDK loop. | **Tie (0)** |

**Final Tally:** TypeScript **+6** vs. Python **+3**.

## Mitigations for TypeScript Risks

1. **Plain Node 20+ runtime:** Do not depend on `bun` in `./serve`. Use `npm ci` and compiled JS (`dist/`) or `tsx`/`node` compatible with Node 20+.
2. **FTS5 safety:** Never use `node:sqlite` (omits FTS5). Use `better-sqlite3` and test a clean cold build inside a `node:20-slim` Docker container before packaging the release package.
3. **Architecture enforcement:** Use `dependency-cruiser` to match the strict layer enforcement of Python's `import-linter`.
4. **Dev linter Node requirement:** `@interfere/landmines` + `oxlint` run during development and CI (Node 22.18+); `./serve` never invokes dev linters at runtime.
