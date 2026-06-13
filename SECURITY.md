# Security Policy

## Reporting a vulnerability

Please report security issues **privately** so they can be fixed before
public disclosure. Use GitHub's private vulnerability reporting:

**https://github.com/MakerViking/mimir/security/advisories/new**

(Settings → Security → "Report a vulnerability".) If you can't use that,
open a minimal issue asking for a private contact and we'll follow up — do
not include exploit details in a public issue.

Please include: affected version (`mimir --version`), a description, and a
minimal reproduction if you have one. We aim to acknowledge within a few
days.

## Threat model

Mimir is a local-first, single-user tool: the store, the search index, and
the models all live on your machine, and there is **zero telemetry**. The
attack surfaces we care about:

- **Untrusted content reaching the store.** An AI agent may relay untrusted
  text (web pages, repo files, pastes) into a memory. Memory titles, bodies,
  and tags are therefore treated as attacker-influenceable, and the HTML
  surfaces (`dashboard`, `graph viz`) escape/sanitize them — they render in a
  `file://` page, so injection there would run with local-file privileges.
- **Generated files.** Dashboard/graph HTML is written owner-only (`0600`).
- **Model downloads.** Embedding/reranker models download over HTTPS only on
  explicit opt-in (`mimir init` / `mimir embed --fetch`); a marker file gates
  silent network access so an agent call never fetches on its own.

Out of scope: multi-tenant/server deployments (Mimir is not designed as a
shared service), and anything requiring local write access to your own
config or database (an attacker with that already owns the box).

## Supported versions

Fixes land on the latest release line. Please upgrade to the newest version
before reporting.
