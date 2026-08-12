# Upstream sync runbook

This product is composed from two independent forks. In both repositories,
`origin` is the only push target and `upstream` is fetch-only. Never merge the
repositories, never run `git pull upstream main` on `main`, and never force-push
either `main` branch.

## One-time local protection

Run in both `/Users/baran/Desktop/rag-frontend` and
`/Users/baran/Desktop/rag-backend`:

```sh
git remote set-url --push upstream DISABLED
git remote get-url origin
git remote get-url upstream
git remote get-url --push upstream
```

The last command must print `DISABLED`. Before every push, also run
`git remote get-url --push origin` and confirm that it is the corresponding
`https://github.com/acrbaran/...` fork.

## Backend sync

```sh
cd /Users/baran/Desktop/rag-backend
git switch main
git status --short --branch
git fetch upstream --tags
git switch -c chore/upstream-sync-<version>
git merge --no-ff upstream/main
```

Do not resolve route or response-shape conflicts from memory. The merged local
source is the contract authority. The backend test suite and secret scan must
pass before the branch is pushed to `origin`.

The generated integration artifacts live in the frontend repository. After a
backend sync, regenerate them from the frontend checkout:

```sh
cd /Users/baran/Desktop/rag-frontend
node scripts/rag-platform/route-inventory.mjs --backend-ref <version>
node scripts/rag-platform/proxy-config.mjs
node scripts/rag-platform/route-inventory.mjs --backend-ref <version>
node scripts/rag-platform/coverage-matrix.mjs
node scripts/rag-platform/contract-matrix.mjs
node scripts/rag-platform/proxy-config.mjs --check
node scripts/rag-platform/route-inventory.mjs --backend-ref <version> --check
node scripts/rag-platform/coverage-matrix.mjs --check
node scripts/rag-platform/contract-matrix.mjs --check
```

Any route added, removed, made unreachable or left unclassified blocks the
sync. The inventory must use the same tag/ref as the image build; scanning the
moving backend worktree while deploying an older pinned image is a release
blocker, even when some response shapes still look compatible.

## Frontend sync

The frontend already provides `scripts/sync-upstream.sh`, which keeps the pure
mirror branch separate from product work and stops before committing. Run it
from a clean, dedicated sync branch and follow the verification commands it
prints:

```sh
cd /Users/baran/Desktop/rag-frontend
git switch main
git status --short --branch
git switch -c chore/upstream-sync-<version>
./scripts/sync-upstream.sh
```

Before opening the pull request, run the full frontend checks plus:

```sh
node scripts/rag-platform/branding-scan.mjs
node scripts/rag-platform/route-inventory.mjs --check
node scripts/rag-platform/coverage-matrix.mjs --check
node scripts/rag-platform/contract-matrix.mjs --check
```

Never “fix” branding scan failures by renaming licence text, upstream URLs,
CLI/package identifiers, environment variables, persistence keys or model ids.
Add an allowlist rule only when the value is demonstrably a non-display
identifier or required attribution, and record the reason in the rule.

## Secret/data gate

Stop before commit or push if either tree contains real credentials, private
keys, access/session tokens, logs, database/object-store contents, uploaded
documents, model files or dumps. Check both tracked content and staged content;
do not paste suspected values into an issue or CI log. The repositories'
ignore rules must continue to cover `.env` overrides, keys, logs, volumes,
uploads, model artifacts, dumps and generated local configuration.

## GitHub controls

For both forks, configure the GitHub repository settings with:

1. `main` branch protection or ruleset: pull request required, force-push and
   deletion disabled, approval required, conversations resolved.
2. Required status checks: project tests, route/coverage/contract gates where
   applicable, secret scanning and dependency review.
3. Secret scanning and push protection enabled.
4. Dependabot alerts and security updates enabled.
5. Backend image CI restricted to a protected `acrbaran/rag-backend` commit or
   tag and publishing SBOM, scan and provenance artifacts.

These controls were applied to both forks on 2026-08-12. The API readback is
recorded in `docs/rag-platform/github-governance-evidence.json`. The required
check definitions live in the two repository working trees; the owned backend
image workflow verifies the protected `v0.26.4` commit and produces SBOM,
HIGH/CRITICAL vulnerability-scan and provenance evidence. Re-read the GitHub
API state during every release audit instead of treating this dated evidence as
permanent.

## Rollback

Before the sync PR is merged, delete the sync branch and return to `main`.
After merge, revert the merge commit through a new pull request; do not reset or
force-push shared history. Restore generated contract artifacts by regeneration
against the reverted backend commit, never by hand-editing them.
