# Runtime HTTP API Contract

`./serve` exposes a minimal, stateless HTTP interface for AI agents and benchmark runners.

## 1. Startup and Environment

Run `./serve` from the repository root with Node 20+. It installs production dependencies (`npm ci --omit=dev`), compiles TypeScript, ingests `CORPUS_DIR` into `.cache/`, and starts the HTTP server in the foreground.

| Variable | Required | Meaning |
| :--- | :--- | :--- |
| `CORPUS_DIR` | Yes | Path to the read-only corpus directory (`<source>/*.json` with `dataset_doc_uuid`). |
| `PORT` | Yes | TCP port to listen on (`127.0.0.1:$PORT`). |
| `GEMINI_API_KEY` | Yes | Gemini API key for embeddings, routing, and synthesis. |
| `BRAIN_PROFILE` | No | Optional Run Profile name under `eval/profiles/` (defaults to `prod`). |
| `BRAIN_PROFILE_OVERRIDES` | No | Optional comma-separated overrides (for example, `topK=20,maxAgentTurns=5`). |

On `SIGTERM`, the process drains in-flight requests and exits cleanly within 10 seconds with zero leaked child processes.

## 2. Readiness: `GET /health`

- Returns `503 Service Unavailable` while ingesting, embedding, or draining.
- Returns `200 OK` only when ingestion and vector loading are complete.

## 3. Question Answering: `POST /ask`

Request (`Content-Type: application/json`):

```json
{"question": "What are the default multipart file upload size limits?"}
```

Response (`200 OK`, `<= 1 MiB`):

```json
{
  "answer": "The default limits are 10 MiB per file and 50 MiB total per request.",
  "document_ids": ["dsid_ae068ee4aa9640159427cd941bef0238"]
}
```

- Each request is stateless and supports at least 4 concurrent `/ask` calls.
- Every ID in `document_ids` is verified against the active corpus and the query read set.
- When the corpus lacks evidence, the response states that clearly and returns `document_ids: []`.

## 4. Bare Container Verification

Verify fresh-clone unattended operation inside `node:20-slim` (no `gcc`, `g++`, `make`, or `python3`, minimal env vars, `503 -> 200`, concurrency 4, peak RSS `<= 4 GiB`, clean `SIGTERM` exit):

```bash
npm run verify:docker -- --live-ask
```
