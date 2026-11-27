# SOTA Enterprise RAG, GraphRAG, and Compute Research

> Note: This is our Day-0 research snapshot collected before coding. For final promoted models (`gemini-3.8-flash`, `gemini-embedding-2`) and verified results, see `README.md` and `eval/profiles/prod.json`.

Primary-source findings collected on 2026-11-25 before writing application code. Every candidate architecture was filtered against our hardware and contract scorecard:

- **Hardware:** CPU-only, peak RAM <= 4 GB RSS (safe on an 8 to 16 GB Linux VM or laptop).
- **Latency:** Cold ingestion of ~21.6K docs <= 30 minutes (hard limit: 4 hours); query p95 <= 60 seconds at concurrency 4 (hard limit: 180s).
- **Cost:** Gemini ingest <= \$15, query <= \$0.10 (hard cap: \$500 total).
- **Quality:** High answer correctness and completeness, tight citation precision and recall (`document_ids`), strong multi-doc synthesis (3 to 14 docs), and strict abstention ("no answer beats a wrong answer").

## 1. Measured Corpus & Question Profile (`data/`)

- **Corpus size:** 21,612 JSON documents, **119.41 MB**, ~**31.3M tokens** across 9 sources (`slack` 11,606, `gmail` 4,957, `linear` 1,546, `google_drive` 1,116, `hubspot` 661, `confluence` 461, `fireflies` 440, `jira` 418, `github` 407).
- **Document length:** p50 5.2 KB (~1.3K tokens), p95 10.5 KB (~2.6K tokens), max 35.9 KB (~9K tokens).
- **Built-in structured graph:** Every source carries explicit cross-links (`linked_linear`, `linked_jira`, `linked_fireflies`, `linked_gmail_threads`, `linked_drive_docs`, `linked_support_tickets`, `related_github_prs`, `related_confluence_pages`, `related_pages`, `linked_issues`, `linked_artifacts`, `parent_issue`, `dependencies`, `customer_company`, `related_account`, `participants`, timestamps, plus `title_field_name` and `content_field_names`).
- **Dev questions (`data/questions.jsonl`, 79 total):**
  1. **Single-doc deep lookups (31):** Exact ticket, PR, call, or account details.
  2. **Structured filter and counting queries (~12+ multi-doc, up to 14 docs):** Counting or listing Jira Sev0 tickets, HubSpot qualified/enterprise accounts, merged PRs in a repo, or finalized Drive docs. Vector search alone misses documents here; a **SQL metadata filter tool** hits 100% recall in milliseconds.
  3. **Cross-source multi-doc tracing (23 multi-source, 24 with >=3 docs):** Tracing one feature, outage, or billing dispute across Confluence, GitHub, Jira, Linear, Slack, and Drive.
  4. **Latest-state / timeline questions:** Resolving current status or dates via timestamps and governing-record priority.
  5. **Adversarial unanswerable traps (11 = 14%):** Ask about *real* customers and tickets in the corpus, but request an outcome or decision that never happened or was never recorded.

## 2. What Top `EnterpriseRAG-Bench` Systems Do (Nov 2026)

On Onyx's `EnterpriseRAG-Bench` (512K docs across the same 9 sources; scoring `mean(I[correct] * completeness_pct)` where any wrong or hedged value scores 0), one-shot baselines score 37.7 (vector) to 50.6 (BM25) and cloud managed RAG scores 41.9 to 61.0. Top leaderboard systems (Intera AI 96.20, AJNT 94.24, KT-Agentic-Search 93.48, PipesHub 93.24, Neurond AI 92.74, benchmark/Aethos 92.64, Flexor.ai 91.48) converge on six patterns:

1. **Hybrid BM25 + Dual-Granularity Dense Vectors:** Keep short docs intact; segment long docs by headings/structure plus a doc-header index.
2. **Zero-LLM Deterministic Entity & Link Graph:** AJNT (#2) explicitly rejected slow LLM GraphRAG extraction. Extract entities, ticket/PR IDs, and cross-links deterministically from JSON fields and regex patterns in seconds for \$0.
3. **Search-First Bounded Agent Loop:** Run hybrid search *before* Turn 1, then expose 5 minimal tools (`search`, `filter_metadata` / SQL, `fetch_doc`, `expand_graph_neighbors`, `count`/tally). In PipesHub's FRAMES 18-pipeline ablation, this jumped from 78.9% best one-shot to **92.7%** overall (**90% vs. 62%** on 5+ doc questions) at \$0.017/query.
4. **Read Full Governing Documents:** Do not synthesize from isolated snippets; fetch full candidate documents before answering.
5. **Governing-Record Adjudication & Citation Pruning:** Resolve look-alikes by timestamp, status, and qualifier; commit to one value without hedging; prune `document_ids` strictly to supporting docs.
6. **Strict Abstention:** Every top-8 system achieves 100% on unanswerable queries.

## 3. Approaches Rejected by Evidence

- **Microsoft GraphRAG:** Fails ingest time, cost, RAM, and factual QA (HippoRAG 2 Table 12: 115.5M input / 36.1M output tokens, 277 min on 4x H100s for 11.6K passages).
- **LightRAG:** Fails ingest cost/time and collapses on factual multi-hop QA (6.6 avg F1 in HippoRAG 2 Table 2 due to lost passage grounding).
- **Full HippoRAG 2 (OpenIE):** Fails RAM (~15 to 20 GB at 21.6K enterprise docs) and OpenIE ingest cost (>\$30). Only its bipartite Personalized PageRank (PPR) idea over deterministic links is worth testing.
- **Small CPU Cross-Encoder Rerankers (MiniLM / FlashRank):** Fast (~35 to 75 ms, 23 MB INT8), but PipesHub's FRAMES ablation showed small cross-encoders **hurt multi-hop accuracy by -8.7 pts** by dropping intermediate linking documents. Do not add unless ablations prove a net gain on multi-doc recall.

## 4. Compute, RAM, and Nov 2026 Gemini Pricing

- **Active Gemini models (Nov 2026):** `gemini-2.5-flash` and `gemini-2.5-pro` deprecate Dec 1, 2026. Use:
  - `gemini-embedding-001` (MRL 768/1,536/3,072 dims): **\$0.15/1M** std (**\$0.075/1M** Batch). Full 31.3M-token corpus = **~\$4.70** (~6 to 35 min depending on rate-limit tier).
  - `gemini-3.5-flash-lite`: **\$0.30** in / **\$0.03** cached (90% off) / **\$2.50** out.
  - `gemini-3.8-flash`: **\$0.75** in / **\$0.075** cached / **\$3.75** out (intro pricing through Dec 31, 2026).
- **Per-query cost (3 to 6 agent turns):** **\$0.008 to \$0.020** cached (`3.5-flash-lite`) or **\$0.017 to \$0.041** cached (`3.8-flash`).
- **Memory footprint (80K chunks x 768-dim):** `float32` matrix = **234.4 MiB** (`int8` = 58.6 MiB). Single SQLite DB (text + external-content FTS5 + vectors) = **~450 MB on disk**, **~550 to 850 MiB peak RSS** (over 3 GB headroom under our 4 GB cap).
