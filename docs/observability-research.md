# Local & Cloud Observability and Eval Dashboard Research (AI SDK v6)

Primary-source comparison collected on 2026-11-25 for Node.js / TypeScript on CPU-only hosts running Vercel AI SDK v6 (`ai` `^6.0.0`) and Vitest 3.

## Key AI SDK v6 Compatibility Fact

- In **AI SDK v6 (`^6.0.0`)**, telemetry uses built-in per-call `experimental_telemetry: { isEnabled: true }` paired with `@opentelemetry/sdk-node`.
- `@ai-sdk/otel`, `registerTelemetry`, `@langfuse/vercel-ai-sdk`, and `@ai-sdk/devtools` 1.x (`DevToolsTelemetry`) target **AI SDK v7** (not v6).

## Comparison Table

| Tool | UI & Storage | Local Containers / RAM | AI SDK v6 (`^6.0.0`) Path | Side-by-Side Eval Matrix | External Account | Key Caveat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Langfuse Cloud (Hobby)** | Cloud web UI (50K units/mo free, 30d retention) + Langfuse CLI | 0 containers, 0 local RAM | `@langfuse/otel` (`LangfuseSpanProcessor`) + `experimental_telemetry` | Yes (Datasets, Experiments, custom scores, trace waterfalls) | Free Hobby account (no card) | ~145 full `dev` runs/mo under 50K unit cap. Opt-in for dev/eval only. |
| **PostHog Cloud** | Cloud web UI (existing free account) | 0 containers, 0 local RAM | `@posthog/ai/otel` (`PostHogTraceExporter`) + `experimental_telemetry` | Limited (trace waterfalls, cost/tokens, online/offline evals; less matrix-focused than Langfuse) | Existing PostHog account | Already available; best for trace/cost waterfall, pair with local scorecard for sweeps. |
| **Promptfoo** | Local web UI (`npx promptfoo view`, SQLite `~/.promptfoo`) | 0 containers, low RAM | HTTP `POST /ask`, custom `.ts` providers, built-in OTLP receiver | Yes (side-by-side prompt/model/hyperparameter matrix + TS graders) | None | Requires **Node 22.22.0+** (dev-only tool; never used inside `node:20-slim` `./serve`). |
| **Arize Phoenix** | Local web UI (`:6006`, OTLP `:4317`, SQLite default) | 1 Docker container; 1 GB min, 2 to 4 GB rec | `@arizeai/openinference-vercel@^2` + `experimental_telemetry` | Yes (`@arizeai/phoenix-client`, `@arizeai/phoenix-evals`) | None | Best zero-account 1-container local OTel + eval UI; mount `/root/.phoenix` to persist. |
| **Vercel `@ai-sdk/devtools`** | Local web UI (`:4983`, `.devtools/generations.json`) | 0 containers, very low RAM | Legacy `devToolsMiddleware` on v6 | No (traces, tool calls, tokens only) | None (`AI Gateway` is cloud-only) | No eval tables or run comparison. |
| **Langfuse v3 (self-hosted)** | Local web UI (`:3000`; Postgres, ClickHouse, MinIO, Redis) | 6 Docker containers; 4 to 6 GB min, 16 GB rec | `@langfuse/otel` + `experimental_telemetry` | Yes | Local keys | Too heavy for modest local dev hosts (ClickHouse OOM risk). |
| **Evalite** | Local Vitest web UI | 0 containers | Unmaintained | Yes | None | **Deprecated:** Archived by Matt Pocock in favor of Langfuse. Do not adopt. |
| **Braintrust** | Cloud UI only (`bt eval --dev`) | 0 local containers | Supported (v3 to v7) | Yes | Required | No offline local UI; self-hosting is Enterprise BYOC only. |
| **Aspire Dashboard** | Local OTLP viewer (`:18888`, in-memory) | 1 container or `npx` | Partial (`ai.*` vs. `gen_ai.*` spans) | No | None | No persistence and no eval score tables. |

## Langfuse Experiment & Run Profile Integration (JS/TS SDK v4)

Verified against Langfuse docs on 2026-11-25 for `@langfuse/client`, `@langfuse/otel`, and `@langfuse/tracing`:

- **Local data with side-by-side compare:** `runExperiment` accepts local question arrays with stable item `id` values. Langfuse aligns rows by `id` in **Experiments -> Compare** (green/red deltas for score, cost, latency, and trace drill-down) without uploading `eval/splits.json` or locked `holdout` items into a cloud dataset.
- **Run Profile visibility:** Pass `profileName`, full resolved **Run Profile** knobs, `gitSha`, `split`, and `splitSeed` in `runExperiment` `metadata` (plus `tags` via `propagateAttributes` or `experimental_telemetry.metadata`). Every knob is searchable and visible on each experiment run and trace.
- **Item and run scores:** Item-level `evaluators` attach `composite_score`, `answer_score`, `citation_score`, `abstained`, `false_abstention`, and `invalid_citations` to each question trace. Run-level `runEvaluators` attach suite and stratum averages to the experiment run.
- **Cross-process HTTP tracing (`POST /ask`):** When Langfuse keys are set, OpenTelemetry W3C `traceparent` headers propagate from `npm run eval` to `./serve`. Every agent turn, tool call, and AI SDK v6 `experimental_telemetry` generation inside `./serve` nests directly under the matching experiment item trace.

## Primary Sources

- Vercel AI SDK & AI Gateway: https://ai-sdk.dev, https://github.com/vercel/ai, https://vercel.com/docs/ai-gateway
- Langfuse Cloud, CLI, OTel & Self-Hosting: https://langfuse.com/docs, https://langfuse.com/self-hosting, https://github.com/langfuse/langfuse
- PostHog LLM Analytics & Evals: https://posthog.com/docs/llm-analytics
- Promptfoo: https://www.promptfoo.dev/docs/intro/, https://github.com/promptfoo/promptfoo
- Arize Phoenix: https://arize.com/docs/phoenix, https://github.com/Arize-ai/phoenix, https://www.npmjs.com/package/@arizeai/openinference-vercel
- Evalite (deprecated): https://github.com/mattpocock/evalite, https://evalite.dev
- Braintrust: https://www.braintrust.dev/docs
- Aspire Dashboard: https://aspire.dev
