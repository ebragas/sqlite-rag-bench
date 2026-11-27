# Security Policy

## Supported versions

Security fixes apply to the latest release on `main`.

| Version | Supported |
| ------- | --------- |
| 0.x     | Yes       |

## Reporting a vulnerability

Do not open a public GitHub issue for a security vulnerability.

Report vulnerabilities privately through GitHub Security Advisories for `ebragas/sqlite-rag-bench`:

- https://github.com/ebragas/sqlite-rag-bench/security/advisories/new

Include these details in your report:

1. The affected file, command (`./serve`, `npm run ask`, `npm run eval`), or dependency version.
2. Steps to reproduce the issue, including any request payloads or sample corpus files.
3. Your assessment of impact.

## Response timeline

- **Acknowledgement:** within 48 hours of receipt.
- **Triage and severity assessment:** within 5 business days.
- **Fix and coordinated disclosure:** target 14 days for critical or high severity issues, and 30 days for medium or low severity issues.

We credit reporters in the release notes and security advisory unless you ask to stay anonymous.

## Trust boundaries for local deployments

- **Credential boundary:** Default `./serve` reads only `CORPUS_DIR`, `PORT`, and `GEMINI_API_KEY`. Keep API keys and optional Langfuse tracing keys out of version control (`.env` is gitignored).
- **Single-tenant corpus trust:** `CORPUS_DIR` is opened read-only and treated as a single-tenant corpus. Do not point `./serve` at mixed-permission multi-tenant data without adding caller-level access control first.
- **Parameterized metadata filtering:** The `filter_metadata` tool validates field paths against a strict allowlist pattern and runs parameterized SQLite queries. User input and model outputs are never interpolated into raw SQL.
- **Network exposure:** `./serve` binds a local HTTP server with no built-in authentication. Place it behind a reverse proxy with TLS and caller authentication before exposing it beyond `localhost`.
