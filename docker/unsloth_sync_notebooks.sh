#!/usr/bin/env bash
# Populate and refresh /workspace/unsloth-notebooks from unslothai/notebooks.
#
# The image bakes a read-only template at /opt/unsloth-notebooks so the
# notebooks are present in JupyterLab instantly and offline. On boot this script
# copies the template into /workspace/unsloth-notebooks (first run only) and then
# best-effort refreshes from GitHub when upstream has actually advanced.
#
# The user's edits ALWAYS win. We remember the content hash of every file we
# wrote; on refresh a file whose current hash differs from what we last wrote is
# treated as user-modified and is left untouched. So a refresh only updates files
# the user has not changed and adds new ones -- it never clobbers an edited
# notebook and never produces merge conflicts.
#
# Opt-out / tuning (all optional):
#   UNSLOTH_SKIP_NOTEBOOK_SYNC=1      do nothing (no populate, no refresh)
#   UNSLOTH_SKIP_NOTEBOOK_REFRESH=1   populate from the baked template only;
#                                     never touch the network
#   UNSLOTH_NOTEBOOKS_DIR=<path>      target dir (default /workspace/unsloth-notebooks)
#   UNSLOTH_NOTEBOOKS_REPO=<url>      source repo (default unslothai/notebooks)
#   UNSLOTH_NOTEBOOK_FETCH_TIMEOUT=N  seconds for each network op (default 60)
set -u

TEMPLATE="${UNSLOTH_NOTEBOOKS_TEMPLATE:-/opt/unsloth-notebooks}"
DEST="${UNSLOTH_NOTEBOOKS_DIR:-/workspace/unsloth-notebooks}"
REMOTE="${UNSLOTH_NOTEBOOKS_REPO:-https://github.com/unslothai/notebooks}"
STATE="$DEST/.unsloth_sync_state"     # "sha256  relpath" of what we last wrote
SYNCED="$DEST/.unsloth_sync_commit"   # upstream commit we last synced to
TIMEOUT="${UNSLOTH_NOTEBOOK_FETCH_TIMEOUT:-60}"

[ "${UNSLOTH_SKIP_NOTEBOOK_SYNC:-0}" = "1" ] && exit 0
[ -d "$TEMPLATE" ] || exit 0
mkdir -p "$DEST" 2>/dev/null || exit 0

hash_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# Record "<hash>  <relpath>" for every file currently under DEST (skip metadata).
record_state() {
    : > "$STATE.tmp" 2>/dev/null || return 0
    ( cd "$DEST" && find . -type f -print0 ) | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in
            .unsloth_sync_state|.unsloth_sync_commit) continue ;;
        esac
        printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
    done
    mv "$STATE.tmp" "$STATE" 2>/dev/null || rm -f "$STATE.tmp"
}

# 1) First-boot populate from the baked template (instant, works offline).
if [ ! -f "$STATE" ]; then
    ( cd "$TEMPLATE" && find . -type f -print0 ) | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in .unsloth_template_commit) continue ;; esac
        mkdir -p "$DEST/$(dirname "$rel")" 2>/dev/null || true
        cp -a "$TEMPLATE/$rel" "$DEST/$rel" 2>/dev/null || true
    done
    record_state
    cp -a "$TEMPLATE/.unsloth_template_commit" "$SYNCED" 2>/dev/null || true
    echo "[unsloth-nb] notebooks ready at $DEST"
fi

# 2) Best-effort GitHub refresh -- only when upstream has advanced. Edits win.
[ "${UNSLOTH_SKIP_NOTEBOOK_REFRESH:-0}" = "1" ] && exit 0
command -v git >/dev/null 2>&1 || exit 0
command -v sha256sum >/dev/null 2>&1 || exit 0

last="$(cat "$SYNCED" 2>/dev/null || true)"
remote="$(timeout "$TIMEOUT" git ls-remote "$REMOTE" HEAD 2>/dev/null | cut -f1)"
[ -z "$remote" ] && exit 0            # offline / unreachable -> keep what we have
[ "$remote" = "$last" ] && exit 0     # nothing new since last sync -> done

TMP="$(mktemp -d)"
if ! timeout "$TIMEOUT" git clone -q --depth 1 "$REMOTE" "$TMP" 2>/dev/null; then
    rm -rf "$TMP"; exit 0             # network died mid-way -> keep what we have
fi

declare -A LAST
if [ -f "$STATE" ]; then
    while read -r h p; do
        [ -n "${p:-}" ] && LAST["$p"]="$h"
    done < "$STATE"
fi

TMPSTATE="$(mktemp)"
updated=0; kept=0
while IFS= read -r -d '' f; do
    rel="${f#"$TMP"/}"
    case "$rel" in .git|.git/*) continue ;; esac
    dst="$DEST/$rel"
    if [ -e "$dst" ]; then
        rec="${LAST[$rel]:-}"
        if [ -n "$rec" ] && [ "$(hash_of "$dst")" != "$rec" ]; then
            # User changed this file since we wrote it -> keep theirs, keep marker.
            printf '%s  %s\n' "$rec" "$rel" >> "$TMPSTATE"
            kept=$((kept + 1))
            continue
        fi
    fi
    mkdir -p "$(dirname "$dst")" 2>/dev/null || true
    if cp -a "$f" "$dst" 2>/dev/null; then
        printf '%s  %s\n' "$(hash_of "$dst")" "$rel" >> "$TMPSTATE"
        updated=$((updated + 1))
    fi
done < <(find "$TMP" -type f -print0)

mv "$TMPSTATE" "$STATE" 2>/dev/null || rm -f "$TMPSTATE"
echo "$remote" > "$SYNCED" 2>/dev/null || true
rm -rf "$TMP"
echo "[unsloth-nb] notebooks refreshed from GitHub: $updated updated, $kept kept (your edits)"
exit 0
