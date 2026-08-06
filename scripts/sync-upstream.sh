#!/usr/bin/env bash
#
# Sync upstream unsloth into this frontend-only fork.
#
# What it does, in order:
#   1. Fetches the upstream remote and fast-forwards the vendor branch to it.
#      The vendor branch is a plain mirror of upstream and is never edited.
#   2. Creates a dated sync branch off the base branch (develop).
#   3. Merges the vendor branch with --no-commit, so a real merge base is
#      recorded and the next sync only has to look at new commits.
#   4. Throws away everything the merge dragged in from outside the frontend,
#      then auto-resolves the conflicts whose only difference is a convention
#      this fork applies on top of upstream (blanked SPDX headers).
#   5. Prints whatever is genuinely ours to decide, and stops.
#
# It never commits unless --commit is passed, and never pushes.
#
# Usage:
#   scripts/sync-upstream.sh [options]
#
#   --dry-run          Report only. No refs, index, or files are written.
#   --no-fetch         Skip the network. Use the vendor branch as it stands.
#   --base <branch>    Branch to merge into. Default: develop
#   --branch <name>    Sync branch name. Default: sync/unsloth-frontend-<date>
#   --include-tests    Keep studio/frontend/tests/ instead of dropping it.
#   --blank-spdx       Also blank SPDX headers on brand-new upstream files,
#                      not just on the files this script resolves.
#   --commit           Create the merge commit, but only if nothing is left
#                      to resolve by hand.
#   -h, --help         This text.
#
# Environment overrides: UPSTREAM_REMOTE, UPSTREAM_BRANCH, VENDOR_BRANCH,
# BASE_BRANCH, KEEP_PREFIX, KEEP_LOCALES.

set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
VENDOR_BRANCH="${VENDOR_BRANCH:-vendor/unsloth}"
BASE_BRANCH="${BASE_BRANCH:-develop}"

# The only tree this fork ships. Everything else upstream sends is discarded.
KEEP_PREFIX="${KEEP_PREFIX:-studio/frontend/}"
# Repo-root files kept, but always at the base branch's version. Upstream's
# README describes the whole project, which is not what this repo is.
KEEP_FILES=".gitignore README.md"
# Locales this fork ships. src/i18n/messages.ts declares only these, so a new
# upstream locale file would merge cleanly and then fail i18n:check.
KEEP_LOCALES="${KEEP_LOCALES:-en.ts}"
LOCALES_DIR="studio/frontend/src/i18n/locales/"

DRY_RUN=0
DO_FETCH=1
SYNC_BRANCH=""
INCLUDE_TESTS=0
BLANK_SPDX=0
DO_COMMIT=0

die() { printf 'sync-upstream: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }
hdr() { printf '\n== %s ==\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)       DRY_RUN=1 ;;
    --no-fetch)      DO_FETCH=0 ;;
    --include-tests) INCLUDE_TESTS=1 ;;
    --blank-spdx)    BLANK_SPDX=1 ;;
    --commit)        DO_COMMIT=1 ;;
    --base)          shift; [ $# -gt 0 ] || die "--base needs a branch"; BASE_BRANCH="$1" ;;
    --branch)        shift; [ $# -gt 0 ] || die "--branch needs a name"; SYNC_BRANCH="$1" ;;
    -h|--help)       awk 'NR>2 && /^#/ {sub(/^# ?/, ""); print; next} NR>2 {exit}' "$0"; exit 0 ;;
    *)               die "unknown option: $1" ;;
  esac
  shift
done

DROP_PREFIXES=""
[ "$INCLUDE_TESTS" -eq 1 ] || DROP_PREFIXES="${KEEP_PREFIX}tests/"

# ---------------------------------------------------------------- preflight

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not a git repository"
cd "$ROOT"

command -v python3 >/dev/null 2>&1 || die "python3 is required"

[ -e .git/MERGE_HEAD ] && die "a merge is already in progress; finish or 'git merge --abort' it first"
git rev-parse -q --verify HEAD >/dev/null || die "no commits yet"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no >&2
  die "working tree has uncommitted changes; commit or stash them first"
fi

git rev-parse -q --verify "refs/heads/${BASE_BRANCH}" >/dev/null \
  || die "base branch '${BASE_BRANCH}' does not exist"
git rev-parse -q --verify "refs/heads/${VENDOR_BRANCH}" >/dev/null \
  || die "vendor branch '${VENDOR_BRANCH}' does not exist"
git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 \
  || die "remote '${UPSTREAM_REMOTE}' is not configured"

CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || echo '')"
[ "$CURRENT_BRANCH" = "$VENDOR_BRANCH" ] \
  && die "checked out on '${VENDOR_BRANCH}'; switch to '${BASE_BRANCH}' first"

# ------------------------------------------------- vendor branch = upstream

hdr "upstream"

if [ "$DO_FETCH" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    say "dry-run: would fetch ${UPSTREAM_REMOTE}"
  else
    git fetch --prune "$UPSTREAM_REMOTE"
  fi
else
  say "skipping fetch (--no-fetch)"
fi

UPSTREAM_REF="refs/remotes/${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"
git rev-parse -q --verify "$UPSTREAM_REF" >/dev/null \
  || die "'${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}' not found; fetch first"

UPSTREAM_SHA="$(git rev-parse "$UPSTREAM_REF")"
VENDOR_SHA="$(git rev-parse "refs/heads/${VENDOR_BRANCH}")"

if [ "$VENDOR_SHA" = "$UPSTREAM_SHA" ]; then
  say "${VENDOR_BRANCH} already at ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} ($(git rev-parse --short "$UPSTREAM_SHA"))"
elif git merge-base --is-ancestor "$VENDOR_SHA" "$UPSTREAM_SHA"; then
  say "${VENDOR_BRANCH}: $(git rev-parse --short "$VENDOR_SHA") -> $(git rev-parse --short "$UPSTREAM_SHA") (fast-forward)"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "dry-run: ref not moved"
  else
    git update-ref "refs/heads/${VENDOR_BRANCH}" "$UPSTREAM_SHA" "$VENDOR_SHA"
  fi
else
  die "${VENDOR_BRANCH} ($(git rev-parse --short "$VENDOR_SHA")) is not an ancestor of ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} ($(git rev-parse --short "$UPSTREAM_SHA")).
     Upstream was rewritten, or the vendor branch has local commits on it.
     The vendor branch must stay a pure mirror. Inspect it, then reset it by hand."
fi

# In dry-run the ref was not moved, so measure against upstream directly.
if [ "$DRY_RUN" -eq 1 ]; then
  MERGE_SRC="$UPSTREAM_SHA"
else
  MERGE_SRC="refs/heads/${VENDOR_BRANCH}"
fi

MERGE_BASE="$(git merge-base "$BASE_BRANCH" "$MERGE_SRC")"
PENDING="$(git rev-list --count "${MERGE_BASE}..${MERGE_SRC}")"
PENDING_FE="$(git rev-list --count "${MERGE_BASE}..${MERGE_SRC}" -- "$KEEP_PREFIX")"

say "merge base: $(git rev-parse --short "$MERGE_BASE")"
say "new upstream commits: ${PENDING} (${PENDING_FE} touch ${KEEP_PREFIX})"

if [ "$PENDING" -eq 0 ]; then
  say "nothing to sync."
  exit 0
fi
if [ "$PENDING_FE" -eq 0 ]; then
  say "no upstream commit touches ${KEEP_PREFIX}."
  say "a merge would still be useful to move the merge base forward, continuing."
fi

hdr "frontend commits"
git log --oneline --no-decorate "${MERGE_BASE}..${MERGE_SRC}" -- "$KEEP_PREFIX"

# ------------------------------------------------------------- sync branch

if [ -z "$SYNC_BRANCH" ]; then
  STEM="sync/unsloth-frontend-$(date +%F)"
  SYNC_BRANCH="$STEM"
  n=2
  while git rev-parse -q --verify "refs/heads/${SYNC_BRANCH}" >/dev/null; do
    SYNC_BRANCH="${STEM}-${n}"
    n=$((n + 1))
  done
else
  git rev-parse -q --verify "refs/heads/${SYNC_BRANCH}" >/dev/null \
    && die "branch '${SYNC_BRANCH}' already exists"
fi

hdr "merge"
say "branch: ${SYNC_BRANCH} (from ${BASE_BRANCH})"

if [ "$DRY_RUN" -eq 1 ]; then
  say "dry-run: stopping before branch creation."
  say ""
  say "To run for real:"
  say "  scripts/sync-upstream.sh"
  exit 0
fi

BASE_SHA="$(git rev-parse "$BASE_BRANCH")"
git switch --quiet --create "$SYNC_BRANCH" "$BASE_BRANCH"

# --no-ff so the merge is always recorded, even when it could fast-forward:
# the recorded merge is what makes the next sync incremental.
if git merge --no-commit --no-ff "$MERGE_SRC" >/dev/null 2>&1; then
  say "merged with no conflicts (scope still needs trimming)"
else
  say "conflicts raised; resolving the mechanical ones"
fi

# ------------------------------------------------------------- resolve

BASE_SHA="$BASE_SHA" \
KEEP_PREFIX="$KEEP_PREFIX" \
KEEP_FILES="$KEEP_FILES" \
DROP_PREFIXES="$DROP_PREFIXES" \
LOCALES_DIR="$LOCALES_DIR" \
KEEP_LOCALES="$KEEP_LOCALES" \
BLANK_SPDX="$BLANK_SPDX" \
python3 - <<'PY'
"""Trim the merge to this fork's scope and auto-resolve convention-only conflicts.

Two facts drive everything here:

  * This fork tracks only KEEP_PREFIX. Every path the merge brought in from
    elsewhere is dropped outright.
  * Against upstream, this fork's own diff for the overwhelming majority of
    files is nothing but blanked SPDX/copyright header lines. Whenever that is
    the entire local diff, upstream's version wins and the blanking is
    reapplied, which keeps the header from re-conflicting on every future sync.

Anything that fails those tests is left conflicted and reported.
"""

import os
import re
import subprocess
import sys

KEEP_PREFIX = os.environ["KEEP_PREFIX"]
KEEP_FILES = set(os.environ["KEEP_FILES"].split())
DROP_PREFIXES = tuple(p for p in os.environ["DROP_PREFIXES"].split() if p)
LOCALES_DIR = os.environ["LOCALES_DIR"]
KEEP_LOCALES = set(os.environ["KEEP_LOCALES"].split())
BASE_SHA = os.environ["BASE_SHA"]
BLANK_NEW = os.environ["BLANK_SPDX"] == "1"

# Header lines this fork blanks. Kept narrow on purpose: it must not match a
# line of real code that happens to mention a copyright.
HEADER = re.compile(
    rb"^\s*(?://|#|/\*|\*|<!--)?\s*"
    rb"(?:SPDX-License-Identifier\s*:"
    rb"|Copyright\s+\d{4}-present\s+the\s+Unsloth\s+AI\s+Inc\.)",
)


def git(*args, check=True):
    return subprocess.run(("git",) + args, capture_output=True, check=check)


def git_text(*args):
    return git(*args).stdout.decode("utf-8", "replace")


def blob(sha):
    return git("cat-file", "blob", sha).stdout


def base_blob(path):
    r = git("cat-file", "blob", f"{BASE_SHA}:{path}", check=False)
    return None if r.returncode else r.stdout


def blank_headers(data):
    """Blank header lines, preserving the line count so diffs stay aligned."""
    if b"\x00" in data:  # binary; nothing to strip
        return data
    lines = data.split(b"\n")
    return b"\n".join(b"" if HEADER.match(line) else line for line in lines)


def in_scope(path):
    if path.startswith(DROP_PREFIXES):
        return False
    return path.startswith(KEEP_PREFIX) or path in KEEP_FILES


def unwanted_locale(path):
    return (
        path.startswith(LOCALES_DIR)
        and path[len(LOCALES_DIR):] not in KEEP_LOCALES
    )


def read_index():
    """Return (stage0 paths, unmerged {path: {stage: sha}})."""
    stage0 = []
    for entry in git_text("ls-files", "-s", "-z").split("\0"):
        if entry:
            stage0.append(entry.split("\t", 1)[1])
    unmerged = {}
    for entry in git_text("ls-files", "-u", "-z").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        _mode, sha, stage = meta.split()
        unmerged.setdefault(path, {})[int(stage)] = sha
    return stage0, unmerged


def remove(paths):
    """Drop paths from the index and the worktree, in chunks."""
    paths = list(paths)
    for i in range(0, len(paths), 200):
        git("rm", "-q", "-f", "--ignore-unmatch", "--", *paths[i:i + 200])


def stage_content(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    git("add", "--", path)


stage0, unmerged = read_index()
all_paths = set(stage0) | set(unmerged)

report = {
    "out_of_scope": [],
    "excluded": [],
    "pinned_to_base": [],
    "kept_deleted": [],
    "accepted_deletion": [],
    "took_upstream": [],
    "blanked_new": [],
}
manual = []

# 1. Anything outside the frontend, plus explicitly excluded subtrees.
out_of_scope = sorted(p for p in all_paths if not in_scope(p))
excluded = sorted(p for p in all_paths if p.startswith(DROP_PREFIXES))
report["out_of_scope"] = out_of_scope
report["excluded"] = excluded
remove(out_of_scope + excluded)

# 2. Locale files this fork does not ship. Deletions on our side arrive as
#    conflicts and are handled in step 4; this catches locales upstream *adds*,
#    which would otherwise merge cleanly and break i18n:check.
locales = sorted(
    p for p in all_paths
    if in_scope(p) and unwanted_locale(p) and base_blob(p) is None
)
if locales:
    report["excluded"] += locales
    remove(locales)

# 3. Root files that stay at the base branch's version, whatever upstream did.
for path in sorted(KEEP_FILES):
    if path not in all_paths:
        continue
    want = base_blob(path)
    if want is None:
        remove([path])
    else:
        stage_content(path, want)
    report["pinned_to_base"].append(path)

# 4. Remaining conflicts.
for path in sorted(unmerged):
    if path in out_of_scope or path in excluded or path in KEEP_FILES:
        continue
    if path in locales:
        continue

    stages = unmerged[path]
    base = blob(stages[1]) if 1 in stages else None
    ours = blob(stages[2]) if 2 in stages else None
    theirs = blob(stages[3]) if 3 in stages else None

    # We deleted it; upstream changed it. Our deletion was deliberate.
    if ours is None and base is not None:
        remove([path])
        report["kept_deleted"].append(path)
        continue

    # Upstream deleted it. Safe to follow only if we never really changed it.
    if theirs is None:
        if base is not None and ours == blank_headers(base):
            remove([path])
            report["accepted_deletion"].append(path)
        else:
            manual.append((path, "upstream deleted a file we changed"))
        continue

    # Both sides have content. Upstream wins when our diff was header-only.
    if base is not None and ours == blank_headers(base):
        stage_content(path, blank_headers(theirs))
        report["took_upstream"].append(path)
    elif base is None and ours == blank_headers(theirs):
        stage_content(path, blank_headers(theirs))
        report["took_upstream"].append(path)
    else:
        kind = "both modified" if base is not None else "both added"
        manual.append((path, kind))

# 5. Optional: extend the header convention to files upstream just added.
if BLANK_NEW:
    stage0, _ = read_index()
    for path in sorted(stage0):
        if not in_scope(path) or base_blob(path) is not None:
            continue
        r = git("cat-file", "blob", f":{path}", check=False)
        if r.returncode:
            continue
        blanked = blank_headers(r.stdout)
        if blanked != r.stdout:
            stage_content(path, blanked)
            report["blanked_new"].append(path)

# ------------------------------------------------------------------ report


def block(title, paths, cap=12):
    if not paths:
        return
    print(f"\n{title}: {len(paths)}")
    for p in paths[:cap]:
        print(f"  {p}")
    if len(paths) > cap:
        print(f"  ... and {len(paths) - cap} more")


print("\n-- automatic --")
block("dropped, outside " + KEEP_PREFIX, report["out_of_scope"])
block("dropped, excluded from this fork", sorted(set(report["excluded"])))
block("pinned to " + BASE_SHA[:9], report["pinned_to_base"])
block("kept deleted (our deletion stands)", report["kept_deleted"])
block("deleted upstream, we only had header changes", report["accepted_deletion"])
block("took upstream, reapplied header convention", report["took_upstream"])
block("blanked headers on new upstream files", report["blanked_new"])

if not BLANK_NEW:
    stage0, _ = read_index()
    carried = 0
    for path in stage0:
        if not in_scope(path) or base_blob(path) is not None:
            continue
        r = git("cat-file", "blob", f":{path}", check=False)
        if not r.returncode and blank_headers(r.stdout) != r.stdout:
            carried += 1
    if carried:
        print(
            f"\nnote: {carried} new upstream files keep their SPDX headers."
            f"\n      This fork blanks them elsewhere. Rerun with --blank-spdx"
            f"\n      to make it uniform, or leave them as upstream wrote them."
        )

if manual:
    print(f"\n-- yours to resolve: {len(manual)} --")
    for path, kind in manual:
        print(f"  [{kind}] {path}")

with open(".git/sync-upstream-manual", "w") as fh:
    for path, _kind in manual:
        fh.write(path + "\n")

sys.exit(0)
PY

MANUAL_FILE=".git/sync-upstream-manual"
rm -f "$MANUAL_FILE"

REMAINING="$(git ls-files -u | awk '{print $4}' | sort -u | grep -c . || true)"

hdr "state"
say "branch:            ${SYNC_BRANCH}"
say "unresolved paths:  ${REMAINING}"
say "changed vs ${BASE_BRANCH}: $(git diff --cached --name-only "$BASE_SHA" | grep -c . || true) files"

STRAY=""
for p in $(git diff --cached --name-only "$BASE_SHA"); do
  case "$p" in
    "${KEEP_PREFIX}"*) continue ;;
  esac
  keep=0
  for k in $KEEP_FILES; do
    [ "$p" = "$k" ] && keep=1
  done
  [ "$keep" -eq 1 ] || STRAY="${STRAY} ${p}"
done
if [ -n "$STRAY" ]; then
  say ""
  say "WARNING: staged changes outside ${KEEP_PREFIX}:"
  printf '  %s\n' $STRAY
fi

# --------------------------------------------------------------- finish

if [ "$REMAINING" -gt 0 ]; then
  hdr "next"
  say "Resolve the paths listed above, then:"
  say "  git add <path>...                   # per resolved file"
  say "  (cd ${KEEP_PREFIX} && npm run typecheck && npm run lint \\"
  say "     && npm run i18n:check && npm run catalog:check && npx vite build)"
  say "  git commit                          # completes the merge"
  say ""
  say "Compare check output against ${BASE_BRANCH} before blaming the merge:"
  say "  git worktree add /tmp/baseline ${BASE_BRANCH}"
  say "  ln -s \"\$PWD/${KEEP_PREFIX}node_modules\" /tmp/baseline/${KEEP_PREFIX}node_modules"
  say "  (cd /tmp/baseline/${KEEP_PREFIX} && npm run typecheck; npm run lint)"
  say "  git worktree remove /tmp/baseline"
  say ""
  say "To throw this away:  git merge --abort && git switch ${BASE_BRANCH} \\"
  say "                       && git branch -D ${SYNC_BRANCH}"
  exit 1
fi

if [ "$DO_COMMIT" -eq 1 ]; then
  hdr "commit"
  git commit -F - <<MSG
Merge upstream ${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH} into frontend-only tree

Merge ${VENDOR_BRANCH} ($(git rev-parse --short "$UPSTREAM_SHA")) into ${BASE_BRANCH}, keeping
only ${KEEP_PREFIX}. ${PENDING} new upstream commits, ${PENDING_FE} of which touch the frontend.

Everything outside ${KEEP_PREFIX} is dropped: this fork ships the frontend only.
Conflicts whose entire local diff was the blanked SPDX header were resolved to
upstream's version with the blanking reapplied.

Produced by scripts/sync-upstream.sh.
MSG
  say ""
  say "committed: $(git log --oneline -1)"
  say "Run the checks before pushing:"
  say "  (cd ${KEEP_PREFIX} && npm run typecheck && npm run lint && npx vite build)"
  exit 0
fi

hdr "next"
say "Nothing left to resolve. The merge is staged but not committed."
say "  (cd ${KEEP_PREFIX} && npm run typecheck && npm run lint \\"
say "     && npm run i18n:check && npm run catalog:check && npx vite build)"
say "  git commit                          # completes the merge"
say ""
say "To throw this away:  git merge --abort && git switch ${BASE_BRANCH} \\"
say "                       && git branch -D ${SYNC_BRANCH}"
