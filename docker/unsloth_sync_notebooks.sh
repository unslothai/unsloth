#!/usr/bin/env bash
# Populate and refresh /workspace/unsloth-notebooks from unslothai/notebooks.
#
# On boot this copies the baked read-only template into /workspace/unsloth-notebooks
# (first run), then best-effort refreshes from GitHub when upstream advances.
#
# The user's edits ALWAYS win: each written file's hash is recorded; on refresh a
# file whose hash differs is left untouched. So a refresh only updates unchanged
# files and adds new ones.
#
# Opt-out / tuning (all optional):
#   UNSLOTH_SKIP_NOTEBOOK_SYNC=1      do nothing (no populate, no refresh)
#   UNSLOTH_SKIP_NOTEBOOK_REFRESH=1   populate from the baked template only;
#                                     never touch the network
#   UNSLOTH_KEEP_DELETED_NOTEBOOKS=1  do not restore notebooks the user deleted
#                                     (default: deleted files are healed back)
#   UNSLOTH_NOTEBOOKS_DIR=<path>      target dir (default /workspace/unsloth-notebooks)
#   UNSLOTH_NOTEBOOKS_REPO=<url>      source repo (default unslothai/notebooks)
#   UNSLOTH_NOTEBOOK_FETCH_TIMEOUT=N  seconds for each network op (default 60)
#   UNSLOTH_SKIP_NOTEBOOK_VIEW=1      do not build the categorized folder view
#   UNSLOTH_NOTEBOOKS_VIEW_DIR=<path> categorized view dir
#                                     (default "/workspace/Unsloth Notebooks")
#   UNSLOTH_NB_GPU=amd|cuda           force AMD-* notebook visibility (default:
#                                     autodetect; AMD-* shown only on AMD/HIP)
#   UNSLOTH_KEEP_COLAB_INTRO=1        keep the Colab "Run all on Colab" sentence
#                                     (default: strip it for the Docker image)
set -u

TEMPLATE="${UNSLOTH_NOTEBOOKS_TEMPLATE:-/opt/unsloth-notebooks}"
DEST="${UNSLOTH_NOTEBOOKS_DIR:-/workspace/unsloth-notebooks}"
REMOTE="${UNSLOTH_NOTEBOOKS_REPO:-https://github.com/unslothai/notebooks}"
STATE="$DEST/.unsloth_sync_state"     # "sha256  relpath" of what we last wrote
SYNCED="$DEST/.unsloth_sync_commit"   # upstream commit we last synced to
LOCK="$DEST/.unsloth_sync.lock"       # serialises this script against itself
PARTIAL="$DEST/.unsloth_sync_partial" # set when first-boot populate left files behind
TIMEOUT="${UNSLOTH_NOTEBOOK_FETCH_TIMEOUT:-60}"
LOCK_WAIT="${UNSLOTH_NOTEBOOK_LOCK_TIMEOUT:-600}"

# Resolve a helper script ($1 override, $2 PATH command, $3 sibling filename),
# echoing the path or nothing. Used for SIG, VIEW and STRIP helpers.
PYBIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
_self_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
resolve_helper() {
    if [ -n "$1" ]; then printf '%s' "$1"; return 0; fi
    if command -v "$2" >/dev/null 2>&1; then command -v "$2"; return 0; fi
    [ -n "$_self_dir" ] && [ -f "$_self_dir/$3" ] && printf '%s' "$_self_dir/$3"
    return 0
}
SIG_HELPER="$(resolve_helper "${UNSLOTH_NB_SIG_HELPER:-}" unsloth-nb-content-sig unsloth_nb_content_sig.py)"
VIEW_HELPER="$(resolve_helper "${UNSLOTH_NB_VIEW_HELPER:-}" unsloth-nb-view unsloth_nb_view.py)"
STRIP_HELPER="$(resolve_helper "${UNSLOTH_NB_STRIP_HELPER:-}" unsloth-nb-strip-colab unsloth_nb_strip_colab.py)"

# True only when both are .ipynb and the SIG helper reports the non-boilerplate
# middle identical, so a refresh doesn't rewrite a notebook when only boilerplate
# moved. Any failure returns false.
middle_unchanged() {
    case "$1" in *.ipynb) : ;; *) return 1 ;; esac
    [ -n "$PYBIN" ] && [ -n "$SIG_HELPER" ] || return 1
    [ "${UNSLOTH_NOTEBOOK_BODY_AWARE:-1}" = "1" ] || return 1
    [ "$("$PYBIN" "$SIG_HELPER" "$1" "$2" 2>/dev/null)" = "SAME" ] || return 1
    return 0
}

[ "${UNSLOTH_SKIP_NOTEBOOK_SYNC:-0}" = "1" ] && exit 0
[ -d "$TEMPLATE" ] || exit 0
mkdir -p "$DEST" 2>/dev/null || exit 0

hash_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# Give staged file $1 the metadata destination $2 must keep (the bash twin of
# unsloth_run.py's _stage_metadata). rename swaps the DIRECTORY ENTRY, so the
# staged inode's owner and mode become the published file's and the clone's
# root:root 0644 lands on a bind-mounted notebook the host user owns, locking
# them out. Such a file is reachable because the populate above records a
# bind mount matching the baked template as managed without copying it. Best
# effort: a filesystem refusing chmod/chown must not cost the user the refresh.
stage_metadata() {
    [ -e "$2" ] || return 0
    chmod --reference="$2" "$1" 2>/dev/null || true
    chown --reference="$2" "$1" 2>/dev/null || true
    return 0
}

# Copy $1 onto $2 without disturbing $2's owner/mode: a plain `cp` truncates the
# existing destination inode and leaves its metadata alone, where `cp -a`
# (--preserve=all) would chown the host user's file to root. `cp -a` is still
# right when there is no destination yet.
cp_keep_meta() {
    if [ -e "$2" ]; then cp "$1" "$2" 2>/dev/null; else cp -a "$1" "$2" 2>/dev/null; fi
}

# --- mutual exclusion --------------------------------------------------------
# Every phase below mutates $DEST and rewrites $STATE, and the GitHub refresh
# runs in a DETACHED child of this same script, so two copies are live at once by
# design. Without a lock the parent's strip/view pass interleaved with the child's
# `cp -a` + state rewrite: six identical boots reported "cleaned" 279/289/293/297/
# 300/306/307/315/330 notebooks, and every notebook the child copied while the
# parent was hashing it ended up permanently marked user-edited (its recorded
# hash no longer matched), so it was skipped by every later strip.
#
# One exclusive lock covers a whole invocation. The child therefore cannot start
# until the parent has finished and exited, which also fixes the ORDER: strip and
# view rebuild always run over a quiesced tree. flock is best-effort -- when it is
# unavailable, or $DEST cannot hold the lock file, we fall back to running
# unlocked (the parent still finalizes before forking, see below).
_LOCK_HELD=0
lock_acquire() {
    [ "$_LOCK_HELD" = "1" ] && return 0
    command -v flock >/dev/null 2>&1 || return 0
    # Group-redirect, not `exec ... 2>/dev/null`: bash reports a failed exec
    # redirection before the redirection it was given applies, so a read-only
    # $DEST would print "Permission denied" into the container log.
    { exec 9>>"$LOCK"; } 2>/dev/null || return 0
    flock -w "$LOCK_WAIT" 9 2>/dev/null || return 0
    _LOCK_HELD=1
    return 0
}
lock_release() {
    [ "$_LOCK_HELD" = "1" ] || return 0
    _LOCK_HELD=0
    flock -u 9 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
    return 0
}

# --- categorized folder view + Docker-only Colab cleanups --------------------
# AMD/HIP detection: AMD-*.ipynb are shown only on an AMD GPU. UNSLOTH_NB_GPU
# forces it (amd|cuda); otherwise probe nvidia-smi then the ROCm tools.
nb_gpu_is_amd() {
    case "${UNSLOTH_NB_GPU:-}" in
        amd|AMD|hip|HIP|rocm|ROCm|ROCM) return 0 ;;
        cuda|CUDA|nvidia|NVIDIA|nv|NV) return 1 ;;
    esac
    if command -v nvidia-smi >/dev/null 2>&1 \
       && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        return 1
    fi
    if command -v rocm-smi >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1; then
        return 0
    fi
    return 1   # default: treat as non-AMD (hide AMD-* notebooks)
}

# Rebuild the sibling symlink VIEW from scratch. Symlinks live OUTSIDE $DEST, so
# the sync state machine (find -type f) never sees them.
build_categorized_view() {
    [ "${UNSLOTH_SKIP_NOTEBOOK_VIEW:-0}" = "1" ] && return 0
    [ -n "$PYBIN" ] && [ -n "$VIEW_HELPER" ] || return 0
    [ -d "$DEST/nb" ] || return 0
    _view="${UNSLOTH_NOTEBOOKS_VIEW_DIR:-/workspace/Unsloth Notebooks}"
    if nb_gpu_is_amd; then
        "$PYBIN" "$VIEW_HELPER" "$DEST" "$_view" --amd 2>/dev/null || true
    else
        "$PYBIN" "$VIEW_HELPER" "$DEST" "$_view" 2>/dev/null || true
    fi
}

# Strip the Colab-only "Run all on Colab" sentence from notebooks WE own and the
# user has not edited (STATE-aware), updating their recorded hashes in place.
strip_colab_intros() {
    [ "${UNSLOTH_KEEP_COLAB_INTRO:-0}" = "1" ] && return 0
    [ -n "$PYBIN" ] && [ -n "$STRIP_HELPER" ] || return 0
    [ -f "$STATE" ] || return 0
    "$PYBIN" "$STRIP_HELPER" --state "$STATE" --dest "$DEST" 2>/dev/null || true
}

# Apply both on EVERY exit after the basic guards, so the view + cleanups also
# run on the common "nothing to refresh" / offline paths. Both are idempotent.
# Run-once: the parent calls this explicitly BEFORE it forks the refresh child
# (so the strip can never overlap the child's copy even where flock is missing),
# and the EXIT trap then has nothing left to do.
_FINALIZED=0
finalize() {
    [ "$_FINALIZED" = "1" ] && return 0
    _FINALIZED=1
    strip_colab_intros
    build_categorized_view
    return 0
}
trap 'finalize; lock_release' EXIT

# Everything past this point mutates $DEST / $STATE, so hold the lock for the
# whole run. A detached refresh child blocks here until its parent has exited.
lock_acquire

# Record "<hash>  <relpath>" for every file currently under DEST (skip metadata).
record_state() {
    : > "$STATE.tmp" 2>/dev/null || return 0
    ( cd "$DEST" && find . -type f -print0 ) | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in
            .unsloth_sync_state|.unsloth_sync_state.tmp|.unsloth_sync_commit) continue ;;
            .unsloth_sync.lock|.unsloth_sync_partial) continue ;;
        esac
        printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
    done
    mv "$STATE.tmp" "$STATE" 2>/dev/null || rm -f "$STATE.tmp"
}

# 1) First-boot populate from the baked template (instant, works offline).
#
# Re-runs when the previous populate left files behind. A file whose `cp -a`
# failed (a destination subdirectory that is momentarily unwritable, a full
# disk) gets no state entry, so 1b below never restores it, and stamping the
# sync marker unconditionally told the refresh in 2) that this commit was
# already synced, so it exited early too and the notebook stayed missing for
# good. The marker is now left off and this block re-attempted on the next
# start, which is the offline half of the retry: 2) needs a clone.
#
# Process substitution, not a pipeline, so the failure counter survives the
# loop -- the same reason the refresh loop below reads from one.
if [ ! -f "$STATE" ] || [ -f "$PARTIAL" ]; then
    : > "$STATE.tmp" 2>/dev/null || true
    populate_failed=0
    while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in .unsloth_template_commit) continue ;; esac
        mkdir -p "$DEST/$(dirname "$rel")" 2>/dev/null || true
        # A pre-existing file (bind-mounted or hand-created) is user data: keep it
        # and do NOT record it, else the refresh below would treat it as pristine
        # and overwrite it. Only files we lay down are recorded as managed.
        if [ -e "$DEST/$rel" ]; then
            if [ "$(hash_of "$DEST/$rel")" != "$(hash_of "$TEMPLATE/$rel")" ]; then
                echo "[unsloth-nb] kept existing user file: $DEST/$rel"
                continue
            fi
            # Same bytes already on disk (a bind-mounted checkout of the same
            # notebooks). cp -a is --preserve=all, so copying would only stamp the
            # baked root:root ownership, mode and build mtime onto the host user's
            # own file and lock them out of editing it. Record it as managed -- the
            # hash is identical, so the state is byte-for-byte what cp would write.
            printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
            continue
        fi
        if cp -a "$TEMPLATE/$rel" "$DEST/$rel" 2>/dev/null; then
            printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
        else
            populate_failed=$((populate_failed + 1))
        fi
    done < <(cd "$TEMPLATE" && find . -type f -print0)
    mv "$STATE.tmp" "$STATE" 2>/dev/null || rm -f "$STATE.tmp"
    if [ "$populate_failed" -eq 0 ]; then
        rm -f "$PARTIAL" 2>/dev/null || true
        cp -a "$TEMPLATE/.unsloth_template_commit" "$SYNCED" 2>/dev/null || true
        echo "[unsloth-nb] notebooks ready at $DEST"
    else
        : > "$PARTIAL" 2>/dev/null || true
        rm -f "$SYNCED" 2>/dev/null || true
        echo "[unsloth-nb] $populate_failed notebook(s) could not be written; leaving the sync marker off so the next start retries"
    fi
fi

# 1b) Every-boot OFFLINE restore of deleted notebooks: a file we wrote that the
# user DELETED comes back from the baked template (no network). Existing files are
# never touched. Opt out with UNSLOTH_KEEP_DELETED_NOTEBOOKS=1.
if [ -f "$STATE" ] && [ "${UNSLOTH_KEEP_DELETED_NOTEBOOKS:-0}" != "1" ]; then
    restored=0
    RS_TMP="$(mktemp)"
    while IFS= read -r line; do
        h="${line%%  *}"; rel="${line#*  }"
        if [ -n "$rel" ] && [ "$rel" != "$line" ] \
           && [ ! -e "$DEST/$rel" ] && [ -f "$TEMPLATE/$rel" ]; then
            mkdir -p "$DEST/$(dirname "$rel")" 2>/dev/null || true
            if cp -a "$TEMPLATE/$rel" "$DEST/$rel" 2>/dev/null; then
                printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$RS_TMP"
                restored=$((restored + 1))
                continue
            fi
        fi
        printf '%s\n' "$line" >> "$RS_TMP"
    done < "$STATE"
    mv "$RS_TMP" "$STATE" 2>/dev/null || rm -f "$RS_TMP"
    [ "$restored" -gt 0 ] \
        && echo "[unsloth-nb] restored $restored deleted notebook(s) from the baked set"
fi

# 2) Best-effort GitHub refresh -- only when upstream has advanced. Edits win.
# Detached: the local populate above already ran, and the refresh can spend up
# to 2x TIMEOUT on ls-remote + clone when offline, which must not delay
# container startup. The child re-enters past phase 1 (hash state makes it a
# no-op) and the flag keeps it from forking again.
[ "${UNSLOTH_SKIP_NOTEBOOK_REFRESH:-0}" = "1" ] && exit 0
command -v git >/dev/null 2>&1 || exit 0
command -v sha256sum >/dev/null 2>&1 || exit 0
if [ "${UNSLOTH_NB_REFRESH_CHILD:-0}" != "1" ]; then
    # Finalize BEFORE the fork, not from the EXIT trap after it: the trap used to
    # fire while the child was already copying refreshed notebooks in, which is
    # what made "cleaned N" differ on every boot. Doing it here also keeps the
    # ordering deterministic on hosts without flock. Container startup is not
    # delayed any further -- the trap ran exactly this work in the parent before.
    finalize
    lock_release
    UNSLOTH_NB_REFRESH_CHILD=1 "$0" >/dev/null 2>&1 &
    exit 0
fi

# --- refresh child -----------------------------------------------------------
# The parent has already stripped + built the view for the tree as it stands, so
# suppress the EXIT-trap finalize; it is re-armed below only if this refresh
# actually rewrites notebooks, which keeps an up-to-date boot a true no-op.
_FINALIZED=1

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
updated=0; kept=0; unchanged=0; failed=0
while IFS= read -r -d '' f; do
    rel="${f#"$TMP"/}"
    case "$rel" in .git|.git/*) continue ;; esac
    dst="$DEST/$rel"
    if [ -e "$dst" ]; then
        rec="${LAST[$rel]:-}"
        if [ -z "$rec" ]; then
            # In DEST but never recorded -> a pre-existing user/bind-mounted file.
            # Keep it and don't adopt it into the state (stays protected).
            kept=$((kept + 1))
            continue
        fi
        if [ -n "$rec" ] && [ "$(hash_of "$dst")" != "$rec" ]; then
            # User changed this file since we wrote it -> keep theirs, keep marker.
            printf '%s  %s\n' "$rec" "$rel" >> "$TMPSTATE"
            kept=$((kept + 1))
            continue
        fi
        if [ -n "$rec" ] && middle_unchanged "$dst" "$f"; then
            # Untouched notebook whose only upstream change is the announcements/
            # footer or install-cell cosmetics (comments, spacing). Body identical
            # and the same packages requested, so keep it and its marker. A changed
            # package spec is NOT cosmetic and falls through to the refresh below.
            printf '%s  %s\n' "$rec" "$rel" >> "$TMPSTATE"
            unchanged=$((unchanged + 1))
            continue
        fi
    elif [ -n "${LAST[$rel]:-}" ] && [ "${UNSLOTH_KEEP_DELETED_NOTEBOOKS:-0}" = "1" ]; then
        # We wrote this notebook and the user DELETED it; with the opt-out set,
        # honor the deletion. Keep the record as managed-but-deleted.
        printf '%s  %s\n' "${LAST[$rel]}" "$rel" >> "$TMPSTATE"
        kept=$((kept + 1))
        continue
    fi
    mkdir -p "$(dirname "$dst")" 2>/dev/null || true
    # Publish through a same-dir temp + rename. This child is forked before the
    # entrypoint execs the container command, so JupyterLab is already serving
    # $DEST while this loop runs: cp -a writes in place (the inode is reused), so
    # a reader can catch half-written JSON, and a save made between the recorded-
    # hash check above and this write is destroyed and then recorded as pristine.
    # rename(2) is atomic, and re-reading the hash once the temp is complete
    # shrinks the check-to-write window to the rename itself. The staging name is
    # dot-prefixed and per-PID so a killed refresh leaves nothing visible in the
    # file browser; unsloth_nb_strip_colab.py already publishes these same files
    # this way.
    new="$(dirname "$dst")/.unsloth_nb_new.$$"
    if cp -a "$f" "$new" 2>/dev/null; then
        # Hash what is about to be PUBLISHED, not what is live afterwards. $new is
        # dot-prefixed and per-PID, so nothing else can write it; re-reading $dst
        # after the rename adopts whatever save landed there in the meantime and
        # records the user's own bytes as the sync-owned pristine version, which
        # is exactly what lets the NEXT refresh overwrite their work.
        staged="$(hash_of "$new")"
        if [ -e "$dst" ] && [ "$(hash_of "$dst")" != "${LAST[$rel]:-}" ]; then
            # Saved while we were copying -> their edit wins, keep the marker.
            rm -f "$new"
            printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
            kept=$((kept + 1))
            continue
        fi
        # Publish with the metadata $dst already has, not the clone's root:root.
        stage_metadata "$new" "$dst"
        # A single-FILE bind mount cannot be renamed over (EBUSY); fall back to the
        # previous in-place copy there so that setup keeps working as it does today.
        if mv -f "$new" "$dst" 2>/dev/null || { rm -f "$new"; cp_keep_meta "$f" "$dst"; }; then
            # $staged, not hash_of "$dst": both branches write the bytes of "$f".
            printf '%s  %s\n' "$staged" "$rel" >> "$TMPSTATE"
            updated=$((updated + 1))
        else
            # Publish failed (a read-only single-file bind mount, a full disk).
            # Carry the PREVIOUS record forward: dropping it would leave $dst with
            # no recorded hash, and the next refresh reads that as a user-owned
            # file and keeps the stale copy forever. Count it so $SYNCED is not
            # advanced below, otherwise remote == last short-circuits the next
            # boot and this file is never retried.
            printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
            failed=$((failed + 1))
        fi
    else
        # Even STAGING failed, so nothing was attempted for this path. Same
        # treatment as a failed publish: keep the old record and do not let the
        # sync marker advance, or a read-only directory is recorded as fully
        # synced and never retried.
        printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
        failed=$((failed + 1))
    fi
done < <(find "$TMP" -type f -print0)

mv "$TMPSTATE" "$STATE" 2>/dev/null || rm -f "$TMPSTATE"
# Only claim this commit is synced when every intended publish landed. Recording
# it after a failure makes the next boot exit on remote == last, so the file that
# could not be written is never retried.
if [ "$failed" -eq 0 ]; then
    echo "$remote" > "$SYNCED" 2>/dev/null || true
else
    echo "[unsloth-nb] $failed notebook(s) could not be written; leaving the sync marker so the next start retries"
fi
rm -rf "$TMP"
echo "[unsloth-nb] notebooks refreshed from GitHub: $updated updated, $kept kept (your edits), $unchanged kept (only header/footer changed upstream)"
# Freshly copied notebooks arrive with the upstream Colab intro, and new files
# have to enter the view, so re-arm the finalize -- but only when something was
# actually copied. Still under the lock, so nothing else is touching the tree.
if [ "$updated" -gt 0 ]; then
    _FINALIZED=0
    finalize
fi
exit 0
