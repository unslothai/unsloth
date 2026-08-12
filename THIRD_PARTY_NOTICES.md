# Third-Party Notices

Rag Platform is assembled from third-party open-source work. This file
records what that work is, under which licence it arrives, and where the
unmodified licence text lives in this repository.

Nothing here is a licence grant by Rag Platform. Each component stays
under its own licence, and the obligations of those licences — including
attribution and source-availability requirements — continue to apply to
every distribution of Rag Platform.

Rebranding note: Rag Platform presents its own product name in the user
interface. That renaming never extends to licence texts, copyright
notices, or attribution. The files and headers listed below are preserved
verbatim, upstream names included.

---

## 1. Frontend — Unsloth Studio

| | |
|---|---|
| Upstream project | Unsloth |
| Upstream repository | https://github.com/unslothai/unsloth |
| Copyright | 2024- Unsloth AI. Inc team, Daniel Han-Chen & Michael Han-Chen |
| Fork point | `3bbed688a8e2e32e6d30e8593c71df749b5393fa` (merge base with `upstream/main`) |
| Files derived from upstream | 1232 tracked paths, all but four under `studio/` |

Upstream applies two licences by directory, stated in the appendix of its
root `LICENSE`:

> Files under `unsloth/*`, `tests/*`, `scripts/*` are Apache 2.0 licensed.
> Files under `studio/*`, `unsloth_cli/*` which is optional to install are
> AGPLv3 licensed.

This repository is a frontend-only fork: 1231 of its tracked files are
under `studio/`. **The effective licence for essentially all inherited
code is therefore AGPL-3.0, not Apache-2.0.** The AGPL's network-use
clause (§13) applies to Rag Platform deployments that let users interact
with the software over a network.

Licence texts in this repository:

* `LICENSE` — upstream root licence, byte-identical to
  `upstream/main:LICENSE`. Contains the Apache-2.0 terms plus the
  per-directory appendix quoted above.
* `studio/LICENSE.AGPL-3.0` — GNU Affero General Public License v3,
  byte-identical to `upstream/main:studio/LICENSE.AGPL-3.0`.

Both files had been removed from this fork before Faz 0 (`e9bb2796f`
"chore: remove LICENSE file" and `2bea3f537` "chore: remove non-frontend
files") and were restored from `upstream/main` during Faz 0. The
restoration also repaired 50 source files whose per-file headers point at
a path that did not exist:

```
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
```

Those headers, and the upstream licence link in
`studio/frontend/src/features/settings/tabs/about-tab.tsx`, are
attribution. They are excluded from the branding rename by an explicit
allowlist and must not be edited.

---

## 2. Backend — RAGFlow

| | |
|---|---|
| Upstream project | RAGFlow |
| Upstream repository | https://github.com/infiniflow/ragflow |
| Copyright | 2025 The InfiniFlow Authors |
| Licence | Apache License 2.0 |
| Licence text | `LICENSE` in the backend checkout, byte-identical to `upstream/main:LICENSE` |
| Fork | https://github.com/acrbaran/rag-backend (`origin`), with `upstream` pointing at the project above |

The backend is a separate repository and is not vendored here. Its
`LICENSE` is unmodified and 1967 of its Python and Go sources carry
`Copyright ... InfiniFlow` headers, all preserved.

### 2.1 Container image

Rag Platform derives its local image alias from the upstream-published image.
The base runtime layers stay unchanged. The owned layer adds the Go API/admin
executable that the published image omits, installs the runtime key wrapper and
removes the upstream image's static PEM pair. The executable is compiled from
the matching release tag.

| | |
|---|---|
| Source image | `infiniflow/ragflow:v0.26.4` |
| Digest | `sha256:16d24d1968ab59e2715a85d2590f1569c9539e0362344a42f3a23e8be06a655b` |
| Base image ID | `16d24d1968ab` |
| Created | 2026-07-07T13:30:41Z |
| Platform | `linux/amd64` (single-architecture manifest; no `arm64` variant is published) |
| Owned alias | `rag-platform-backend:0.26.4` |
| Derived image ID | `sha256:fe17fda6fb5a1e244fd9a081d44ae8b9e0af320403df15e71f2e55c509586f71` |
| Go source tag | `v0.26.4` |
| Go source commit | `cb93883f3f8c975eecb2fed81210effeb3bdb06f` |
| Added runtime files | `/ragflow/bin/ragflow_server`, `/usr/local/bin/rag-platform-entrypoint` |
| Removed image files | `/ragflow/conf/private.pem`, `/ragflow/conf/public.pem` |
| Go profile | `CGO_ENABLED=0`, cross-compiled to `linux/amd64` |

Reproduce the alias with:

```sh
infra/rag-platform/build-backend-image.sh
```

The Go builder runs on the build host's native architecture and cross-compiles a
pure-Go `linux/amd64` executable. This avoids the upstream native C++ analyzer's
static-initialization crash under Apple Silicon amd64 emulation. Two owned,
Apache-2.0-compatible no-CGO adapters are copied into the disposable source
tree: a deterministic Unicode tokenizer fallback and a PDF adapter that reports
the optional native compressed-PDF page-count fallback as unavailable. The
Python API/ingestion service retains upstream's full native document parsing.
The tokenizer fallback has a build-stage unit test; all four runtime services
and the hybrid proxy are smoke-tested separately.

`build-backend-image.sh` refuses a tag whose commit does not match the value
above and uses a clean disposable archive, so uncommitted backend files cannot
enter the image. The tag is pinned and never `:latest`, so a captured contract
fixture always maps to one known source and base image. Runtime RSA material is
generated on first start in the named `rag-platform-key-material` volume and is
not present in either Git tree or any image layer.

Inside the container, upstream paths (`/ragflow/...`), Python and Go
import and package names, and upstream environment variable names are
left exactly as upstream ships them. Only host-side names — Compose
project, container, service, network, volume and host directory names —
belong to Rag Platform.

---

## 3. Infrastructure images

Pulled unmodified by the backend's own Compose base file; listed for
completeness.

| Image | Component | Licence |
|---|---|---|
| `elasticsearch:8.11.3` | Search / vector store | Elastic License 2.0 |
| `mysql:8.0.40` | Relational store | GPL-2.0 with the MySQL FOSS Exception |
| `valkey/valkey:8` | Cache and task queue | BSD-3-Clause |
| `nats:2.14.2` | Go API/admin and ingestor message queue | Apache-2.0 |
| `pgsty/minio:RELEASE.2026-03-25T00-00-00Z` | Object store, and the `mc` client used by the owned bucket bootstrap | AGPL-3.0 (MinIO server and client) |

Consult each image's own licence for the authoritative terms; the table
records which licence applies, not its text.

---

## 4. Application dependencies

Frontend npm dependencies are declared in
`studio/frontend/package.json` and pinned in
`studio/frontend/package-lock.json`. Backend Python and Go dependencies
are declared in the backend checkout. Neither set is enumerated here;
generate a current list from the lockfiles when one is needed for a
release.
