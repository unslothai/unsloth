# ADR 0000a — Two repositories, `origin` is ours, `upstream` is theirs, and upstream never lands on `main` directly

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0A, both repositories; `docs/maintenance/upstream-sync.md`
* Supersedes: nothing. Superseded by: nothing.

## Context

Two forks of two different upstream projects are being combined into one
product. Both keep full upstream history, so both have two remotes, and the two
remotes mean opposite things. Pushing to the wrong one publishes our work to a
public project; pulling from the wrong one silently rewrites ours.

### Verified backend state

Read from the repository, not assumed:

```
$ cd /Users/baran/Desktop/rag-backend
$ git remote get-url origin
https://github.com/acrbaran/rag-backend.git
$ git remote get-url upstream
https://github.com/infiniflow/ragflow.git
$ git rev-parse --abbrev-ref HEAD
main
$ git rev-parse main
a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2
$ git rev-parse origin/main
a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2
$ git rev-list --count main
8204
$ git log --reverse --format='%h %ad %s' --date=short | head -1
93f90bad6 2023-12-12 Initial commit
$ git status --porcelain | wc -l
0
```

So: `main == origin/main`, 8204 commits reaching the genuine 2023-12-12 root
commit, clean tree. Full upstream history is present — no squash, no
re-baselined "initial commit", no force-push. That satisfies the plan's
"Backend repository'si tam kaynak geçmişiyle GitHub'a aktarılmıştır; yeni
baseline geçmişi uydurulmayacaktır."

The plan deliberately does not hardcode the upstream URL: "Kaynak URL dokümana
sabitlenmeyecek; gerektiğinde `git remote get-url upstream` ile mevcut
repository yapılandırmasından okunacaktır." The URL above is quoted as *observed
output*, not as a value to configure from this document.

### Verified frontend state

`origin` is `https://github.com/acrbaran/rag-frontend.git`; the working branch
is `feature/rag-platform-phase-0`; every Faz 0 addition is still untracked
(`LICENSE`, `THIRD_PARTY_NOTICES.md`, `docs/`, `infra/`,
`scripts/rag-platform/`, `studio/LICENSE.AGPL-3.0`). No tracked file was
modified, so no pre-existing user change was overwritten.

Upstream for the frontend is the Unsloth repository; the fork point is recorded
in `THIRD_PARTY_NOTICES.md` as `3bbed688a8e2e32e6d30e8593c71df749b5393fa`.

### Why "no direct upstream merge to main" is not bureaucracy

The backend contract is the authority for every phase of this migration, and it
is read from local source. An upstream merge can move a route, change a response
envelope or rename a blueprint prefix. Three generated artifacts —
`route-inventory.json`, `endpoint-coverage-matrix.json`,
`contract-matrix.md` — are derived from that source and are release gates. If
upstream lands on `main` unreviewed, those gates go stale in the same commit
that invalidates them, and the frontend keeps calling routes that moved.

There is already a live instance of this class of drift, recorded in
`docs/rag-platform/fixtures/README.md`: the captured container image
(`infiniflow/ragflow:v0.26.4`) mounts commit routes under
`/datasets` + `/workspace` + `/folders`, while local `main` uses
`/datasets` + `/workspaces`. The image is behind local HEAD. The paths in the
fixtures are therefore not automatically the contract.

### Data that must never be pushed

Both trees hold, or can hold, material that is not source: `.env` files,
private keys, logs, Docker volume contents, uploaded documents, model files and
database dumps. `rag-backend/docker/.env` is **git-tracked upstream** — it ships
with upstream defaults, and a local edit to it is a tracked modification, which
is one reason ADR 0005 pins the proxy scheme in our own
`infra/rag-platform/.env.rag-platform` instead.

## Decision

**1. Two repositories, never merged.** `acrbaran/rag-frontend` at
`/Users/baran/Desktop/rag-frontend` and `acrbaran/rag-backend` at
`/Users/baran/Desktop/rag-backend`, each with independent history, CI and
release cycle. No monorepo conversion. `infra/rag-platform/` may reference both
by explicit path or config; that composition does not move the boundary.

**2. Remote roles are fixed.** In both repositories `origin` is the user's
GitHub repository and is the only push target. `upstream` is the official
source, is fetch-only, and its URL is read with `git remote get-url upstream`
rather than copied from documentation. Enforce fetch-only mechanically:

```
git remote set-url --push upstream DISABLED
```

**3. Upstream lands through a branch and a PR, never on `main`.** The procedure
is fixed as: `git fetch upstream` → branch `chore/upstream-sync-<version>` →
merge → **regenerate all three artifacts and run their `--check` gates** → run
compatibility tests → open a pull request → merge after review. No
`git pull upstream main` on `main`, ever. No force-push to `main` in either
repository.

Regenerating the artifacts is part of the sync, not a follow-up: a sync whose
`--check` gates fail is an incomplete sync.

**4. The runbook is a file, not a habit.** `docs/maintenance/upstream-sync.md`
holds the exact commands, the fetch-only enforcement, the artifact regeneration
step and the push-target check. It exists because the failure mode here is a
one-keystroke mistake with a public consequence.

**5. Nothing but source is pushed.** Both `.gitignore` files are audited for
secret, log, volume, upload, model and generated-config coverage, and a secret
scan runs before any commit. Per the user's standing instruction, a scan hit on
real credentials, logs, database/object-store data or user documents **halts the
commit and push** rather than being cleaned up in passing.

**6. Local source is the contract authority; the running image is not.** Where
the pinned image and local `main` disagree — as they currently do on commit
route prefixes — `route-inventory.md`, generated from local source, is correct.
Fixture *paths* must be re-verified against a container built from local HEAD
before any phase treats them as the contract; fixture *field shapes* remain
usable meanwhile.

## Alternatives rejected

* **Monorepo** — would destroy both upstream histories' independence and make
  every upstream sync a cross-project merge. The plan forbids it outright.
* **Squash the backend to a fresh baseline** — loses the 8204-commit history
  that makes `git merge upstream/main` possible at all, and would make an
  upstream sync a manual patch exercise.
* **Merging upstream straight onto `main`** — puts contract drift and stale
  release gates in the same commit, with no review point.
* **Keeping only one remote and switching its URL when syncing** — one forgotten
  switch pushes our fork to the public upstream. The failure is silent until it
  is public.
* **Documenting the upstream URL as a constant** — the plan explicitly requires
  reading it from repository configuration, so the document cannot drift from
  reality.

## Consequences

* Backend provenance is verifiable in one command set and currently passes:
  `main == origin/main == a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`, 8204
  commits, clean tree.
* Every upstream update costs a branch, a PR, an artifact regeneration and a
  test run. That cost is the point.
* `rag-backend/docker/.env` stays at its upstream value. Deployment
  configuration lives in `infra/rag-platform/.env.rag-platform` (see ADR 0005).
* GitHub-side controls — branch protection, required PRs, secret scanning,
  Dependabot, backend CI — are outside this session's authority. The exact steps
  are documented in `docs/maintenance/upstream-sync.md` and require user action;
  they are reported as outstanding rather than marked done.
* The captured fixtures carry a known path caveat until the image is rebuilt
  from local HEAD. Recorded in `docs/rag-platform/fixtures/README.md`, not
  silently tolerated.
