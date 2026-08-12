# ADR 0005 — Hybrid proxy with an owned Go-enabled v0.26.4 image

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B, `infra/rag-platform/`
* Supersedes: the temporary pure-Python decision made earlier in Faz 0.

## Context

`docker/entrypoint.sh` uses `API_PROXY_SCHEME` both to select nginx routing and
to decide which server processes start:

| scheme | processes |
|---|---|
| `python` | Python API 9380, Python admin 9381 |
| `go` | Go API 9384, Go admin 9383 |
| `hybrid` | all four |

Faz 0 requires `hybrid` and direct health/readiness evidence for all four
services. The published `infiniflow/ragflow:v0.26.4` image could not satisfy
that by itself: its `/ragflow/bin` directory contained only `.gitkeep`, although
its entrypoint tries to execute `bin/ragflow_server` in Go or hybrid mode.

The omission is reproducible from upstream source. `Dockerfile` copies `bin/`
but never invokes `build.sh`; `.gitignore` excludes build products. The binary
requires upstream's C++ tokenizer plus the pinned `office_oxide`, `pdfium` and
`pdf_oxide` static libraries.

During the unfinished part of Faz 0, the deployment temporarily selected
`python` and documented 124 unique unreachable routes. That state was safe but
did not meet the phase acceptance criteria, so it was never a valid completion
state.

## Decision

Use `API_PROXY_SCHEME=hybrid` in the owned environment layer.

Build `rag-platform-backend:0.26.4` with
`infra/rag-platform/build-backend-image.sh`. The script:

1. resolves local backend tag `v0.26.4` and refuses to continue unless it is
   commit `cb93883f3f8c975eecb2fed81210effeb3bdb06f`;
2. creates a disposable clean `git archive`, so the backend checkout and its
   working tree are not modified;
3. builds the Go executable for `linux/amd64` with upstream `build.sh` and the
   exact native-library versions expected by that tag;
4. copies only `/ragflow/bin/ragflow_server` over the pinned upstream runtime
   image; and
5. tags the result `rag-platform-backend:0.26.4` with source/base provenance
   labels.

Container-internal paths, Python/Go packages, environment variable names and
the rest of the upstream image are unchanged. Compose exposes 9380, 9381, 9383
and 9384 and retains the owned project/container/network/volume identities.

The route inventory treats the Go binary as present, but runtime smoke is the
authoritative confirmation. A build alone is not permission to mark a route
enabled if its process or proxy target fails.

## Alternatives rejected

* **Published image + `python`** — maximises the available subset but violates
  the four-service acceptance criterion and leaves Go-only public capabilities
  disabled.
* **Published image + `hybrid`** — nginx forwards selected families to dead
  ports, turning working Python routes into 502s.
* **Modify or commit generated binaries in `rag-backend`** — pollutes the source
  fork, conflicts with upstream ignore rules and loses build provenance.
* **Build from backend `main`** — mixes a moving API source with a pinned
  v0.26.4 Python runtime and invalidates the captured contracts.

## Consequences

* The owned alias is a derived image, not a byte-identical retag. Its base
  digest and added binary provenance must stay in `THIRD_PARTY_NOTICES.md`.
* Building the alias requires Docker, network access to the pinned native
  archives and an amd64 build (emulated on arm64 hosts).
* Any version bump must update the source tag/commit, base image, native
  dependency pins, fixture metadata, inventory and four-service smoke evidence
  together.
* Rollback is `API_PROXY_SCHEME=python` plus the unmodified upstream image tag;
  that is service-safe but returns to a known incomplete Faz 0 state.
