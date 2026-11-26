---
status: accepted
---

# Eval Scoring Rubric and Opt-In Langfuse OpenTelemetry Boundary

We evaluate the system with a strict **Composite Score** (`0.70 * Answer Score + 0.30 * Citation Score`), where **Answer Score** zeroes on any fabricated or contradicting claim (Onyx rule), and we export dev/eval observability to **Langfuse Cloud (Hobby tier)** via AI SDK v6 OpenTelemetry (`experimental_telemetry: { isEnabled: true }` + `@langfuse/otel`) only when Langfuse environment variables are set. We keep prompts in git, skip Langfuse AI Gateway and custom HTML dashboards, and print a terminal scorecard when `npm run eval` runs with only `GEMINI_API_KEY`.

## Considered Options

- **In-repo JSONL + custom static HTML dashboard:** Zero external accounts, but adds UI build and maintenance cost in a 6-to-8-hour window and lacks native dataset experiment workflows.
- **Self-hosted Langfuse v3 (Docker Compose):** Unlimited volume, but requires 6 containers (including ClickHouse) and 16 GB RAM recommended. Deferred until Langfuse Cloud Hobby (50K units/month) is exhausted.
- **PostHog Cloud (`@posthog/ai/otel`):** Already free and works via OpenTelemetry, but is weaker than Langfuse on side-by-side experiment matrices and custom per-item eval score tables.
- **Evalite:** Deprecated upstream in favor of Langfuse.

## Consequences

- Default `./serve` obeys `docs/api-contract.md`'s single-key (`GEMINI_API_KEY`) boundary with zero Langfuse dependency or network calls.
- Reviewers can re-run `npm run eval` with only `GEMINI_API_KEY` (terminal scorecard), or supply Langfuse keys for full visual traces.
- **Responder** and **Judge** adapters are pluggable (AI SDK providers plus System One **Decision Models** following the `typesafe-ai` skill), with `Judge !== Responder` enforced and **Frontier Tier** (or >= 90% calibrated) judging required on `val` and locked `holdout`.
- Pre-flight spend enforcement stays local in `eval/spend-ledger.jsonl` because cloud telemetry computes cost after ingestion.
