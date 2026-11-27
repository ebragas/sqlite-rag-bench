# sqlite-rag-bench Glossary

Ubiquitous language for the local-first enterprise RAG engine, evaluation harness, and storage layers.

## Evaluation Language

**Composite Score**:
The primary headline metric for an evaluation run, combining Answer Score (70%) and Citation Score (30%).
_Avoid_: Total score, overall grade, RAG score

**Answer Score**:
The correctness and fact-completeness score of a Responder output, reduced to zero if any claim is fabricated or contradicts ground truth (Onyx zeroing).
_Avoid_: Accuracy, truthfulness score

**Citation Score**:
The precision-recall F1 score comparing returned `document_ids` against expected corpus documents, equal to 1 on an unanswerable question only when `document_ids` is empty.
_Avoid_: Reference score, source score

**Abstention**:
An explicit statement that the corpus lacks evidence to answer a question, paired with an empty `document_ids` list.
_Avoid_: Refusal, fallback, punt

**False Abstention**:
An abstention returned for a question that the corpus contains enough evidence to answer.
_Avoid_: Missed answer, over-refusal

**Invalid Citation**:
A returned document ID that is malformed or does not match any `dataset_doc_uuid` in the active corpus.
_Avoid_: Hallucinated link, bad reference

**Responder**:
The system configuration producing the answer under test.
_Avoid_: Generator, target model, answerer

**Judge**:
The independent model evaluating a Responder output against ground truth (always distinct from the Responder model).
_Avoid_: Grader, evaluator, critic

**Fast Tier**:
Lower-latency, lower-cost models used for rapid iteration or lightweight steps (for example, `gemini-3.5-flash-lite` or local `clef-flash`).

**Frontier Tier**:
Higher-intelligence reasoning models used for authoritative evaluation or hard synthesis (for example, `gemini-4-neon`, `gemini-3.8-flash`).

**Decision Model**:
A non-generative System One classifier (`/v1/systemone` API: hosted or local Ollama `clef:27b-mxfp8` / `clef-flash`) returning calibrated probabilities with zero output tokens.

**Run Profile**:
A named, immutable set of index-time and query-time configuration knobs (`eval/profiles/*.json`) identifying a single system variant across evaluation runs.

**Promotion Gate**:
The acceptance check a candidate Run Profile must pass (clearing Judge noise without regressing citations, abstention, or latency) before replacing the default profile.

## Storage & Graph Language

**Corpus Document**:
A single source JSON record ingested from `CORPUS_DIR`, uniquely identified by `dataset_doc_uuid` and versioned by `content_hash`.

**Link Graph**:
The deterministic SQLite graph combining direct document-to-document `graph_edges` and bipartite `entity_mentions` extracted from structured JSON fields and regex IDs at zero LLM cost.

**Schema Catalog**:
The ingest-time summary (`source_schema_catalog`) of `metadata_json` keys and low-cardinality enum values per source so query tools discover exact filter paths on unseen corpora.

**Governing Record**:
The most authoritative, up-to-date Corpus Document that settles a conflicting or evolving fact (by timestamp, finalized status, and official source priority).
