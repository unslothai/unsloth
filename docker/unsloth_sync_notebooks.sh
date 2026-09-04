#!/usr/bin/env bash
# Populate and refresh /workspace/unsloth-notebooks from unslothai/notebooks. The
# user's edits always win: each written file's hash is recorded, and a file whose
# hash differs is left untouched.
#
# Opt-out / tuning (all optional):
#   UNSLOTH_SKIP_NOTEBOOK_SYNC=1      do nothing (no populate, no refresh)
#   UNSLOTH_SKIP_NOTEBOOK_REFRESH=1   populate from the baked template only;
#                                     never touch the network
#   UNSLOTH_KEEP_DELETED_NOTEBOOKS=1  do not restore notebooks the user deleted
#                                     (default: deleted files are healed back)
#   UNSLOTH_KEEP_REMOVED_NOTEBOOKS=1  keep pristine notebooks that upstream deleted
#                                     or renamed (default: they are removed too)
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

# so a refresh does not rewrite a notebook when only boilerplate moved
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

# Give a file that has no destination to inherit from the mode a plain write would
# have produced and the OWNER of the directory it lands in. Under -v $PWD:/workspace
# that directory is the host user's, so without this the clone's or template's
# root:root 0644 is published and they cannot edit their own notebook. This is the
# new-file branch of unsloth_run.py's _stage_metadata, which the shell twin lacked.
own_like_dir() {
    chmod "$(printf '%03o' "$(( 0666 & ~0$(umask) ))")" "$1" 2>/dev/null || true
    chown --reference="$2" "$1" 2>/dev/null || true
    return 0
}

# rename swaps the DIRECTORY ENTRY, so without this the clone's root:root 0644 lands
# on a bind-mounted notebook the host user owns. Best effort.
stage_metadata() {
    if [ ! -e "$2" ]; then
        # A notebook upstream just added: no destination metadata exists, and
        # returning here published the clone's ownership unchanged.
        own_like_dir "$1" "$(dirname "$2")"
        return 0
    fi
    chmod --reference="$2" "$1" 2>/dev/null || true
    chown --reference="$2" "$1" 2>/dev/null || true
    return 0
}

# plain `cp` truncates the existing inode and leaves its metadata alone, where
# `cp -a` would chown the host user's file to root
cp_keep_meta() {
    if [ -e "$2" ]; then cp "$1" "$2" 2>/dev/null; else cp -a "$1" "$2" 2>/dev/null; fi
}

# mkdir(2) gives the new directory the CALLER's uid and only setgid carries down, so
# a new category folder would land root:root. Anchored on the nearest existing
# ancestor, not $DEST, because a bind mount can nest.
mkdir_keep_owner() {
    local dir="$1" anchor parent rest cur part
    local -a parts
    [ -d "$dir" ] && return 0
    anchor="$dir"
    while [ ! -d "$anchor" ]; do
        parent="$(dirname "$anchor")"
        [ "$parent" = "$anchor" ] && break
        anchor="$parent"
    done
    mkdir -p "$dir" 2>/dev/null || return 0
    [ -d "$anchor" ] || return 0
    rest="${dir#"$anchor"}"
    cur="$anchor"
    IFS='/' read -r -a parts <<< "$rest"
    for part in ${parts[@]+"${parts[@]}"}; do
        [ -z "$part" ] && continue
        cur="$cur/$part"
        chown --reference="$anchor" "$cur" 2>/dev/null || true
    done
    return 0
}

# The refresh runs in a DETACHED child, so two copies mutate $DEST and $STATE at once
# by design and a notebook the child copied while the parent hashed it is permanently
# marked user-edited. One lock per invocation also fixes the ORDER.
_LOCK_HELD=0
lock_acquire() {
    [ "$_LOCK_HELD" = "1" ] && return 0
    command -v flock >/dev/null 2>&1 || return 0
    # group-redirect: bash reports a failed exec redirection before applying it
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
    return 1   # default: non-AMD, so AMD-* stay hidden
}

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

strip_colab_intros() {
    [ "${UNSLOTH_KEEP_COLAB_INTRO:-0}" = "1" ] && return 0
    [ -n "$PYBIN" ] && [ -n "$STRIP_HELPER" ] || return 0
    [ -f "$STATE" ] || return 0
    "$PYBIN" "$STRIP_HELPER" --state "$STATE" --dest "$DEST" 2>/dev/null || true
}

# called explicitly BEFORE the fork, so the strip never overlaps the child's copy
_FINALIZED=0
finalize() {
    [ "$_FINALIZED" = "1" ] && return 0
    _FINALIZED=1
    strip_colab_intros
    build_categorized_view
    return 0
}
trap 'finalize; lock_release' EXIT

lock_acquire

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

# 1) First-boot populate. Re-runs when the last one left files behind: a file with no
# state entry is never restored by 1b, and stamping the marker anyway makes 2) exit
# early too. Process substitution, not a pipeline, so the counter survives the loop.
if [ ! -f "$STATE" ] || [ -f "$PARTIAL" ]; then
    : > "$STATE.tmp" 2>/dev/null || true
    populate_failed=0
    while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in .unsloth_template_commit) continue ;; esac
        mkdir_keep_owner "$DEST/$(dirname "$rel")"
        # user data: do NOT record it, else the refresh reads it as pristine
        if [ -e "$DEST/$rel" ]; then
            if [ "$(hash_of "$DEST/$rel")" != "$(hash_of "$TEMPLATE/$rel")" ]; then
                echo "[unsloth-nb] kept existing user file: $DEST/$rel"
                continue
            fi
            # same bytes already there, so copying would only stamp root:root on it
            printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
            continue
        fi
        if cp -a "$TEMPLATE/$rel" "$DEST/$rel" 2>/dev/null; then
            # cp -a preserves the TEMPLATE's root:root 0644; hand it to the host user.
            own_like_dir "$DEST/$rel" "$(dirname "$DEST/$rel")"
            printf '%s  %s\n' "$(hash_of "$DEST/$rel")" "$rel" >> "$STATE.tmp"
        else
            populate_failed=$((populate_failed + 1))
        fi
    done < <(cd "$TEMPLATE" && find . -type f -print0)
    # A RETRY must not throw away what we already manage. Between the failed populate
    # and this run the refresh published newer bytes for template files and added
    # notebooks that exist only upstream, and the loop above sees both as user files:
    # the newer ones differ from the template so they hit the "kept existing user
    # file" branch, and the upstream-only ones are never visited at all. Keeping only
    # this loop's records hands all of them to the user permanently, while the commit
    # marker below is stamped anyway, so it looks converged.
    if [ -f "$STATE" ]; then
        declare -A POPULATED=()
        while IFS= read -r line; do
            p="${line#*  }"
            [ -n "$p" ] && [ "$p" != "$line" ] && POPULATED["$p"]=1
        done < "$STATE.tmp"
        while IFS= read -r line; do
            p="${line#*  }"
            [ -n "$p" ] && [ "$p" != "$line" ] || continue
            [ -n "${POPULATED[$p]:-}" ] && continue
            printf '%s\n' "$line" >> "$STATE.tmp"
        done < "$STATE"
        unset POPULATED
    fi
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

if [ -f "$STATE" ] && [ "${UNSLOTH_KEEP_DELETED_NOTEBOOKS:-0}" != "1" ]; then
    restored=0
    downgraded=0
    RS_TMP="$(mktemp)"
    while IFS= read -r line; do
        h="${line%%  *}"; rel="${line#*  }"
        if [ -n "$rel" ] && [ "$rel" != "$line" ] \
           && [ ! -e "$DEST/$rel" ] && [ -f "$TEMPLATE/$rel" ]; then
            mkdir_keep_owner "$DEST/$(dirname "$rel")"
            if cp -a "$TEMPLATE/$rel" "$DEST/$rel" 2>/dev/null; then
                # cp -a preserves the TEMPLATE's root:root 0644; hand it to the host user.
                own_like_dir "$DEST/$rel" "$(dirname "$DEST/$rel")"
                new_h="$(hash_of "$DEST/$rel")"
                printf '%s  %s\n' "$new_h" "$rel" >> "$RS_TMP"
                restored=$((restored + 1))
                # The record was ahead of the baked copy, so this notebook has just
                # gone BACKWARDS to whatever the image shipped. The refresh child
                # exits on remote == last, so leaving the marker alone would strand
                # it there until upstream happened to commit again.
                [ "$new_h" != "$h" ] && downgraded=$((downgraded + 1))
                continue
            fi
        fi
        printf '%s\n' "$line" >> "$RS_TMP"
    done < "$STATE"
    mv "$RS_TMP" "$STATE" 2>/dev/null || rm -f "$RS_TMP"
    # Only when one actually went backwards: dropping the marker on every restore
    # would make an ordinary delete cost a full clone even when the baked copy was
    # already current, and the marker is re-stamped as soon as the refresh succeeds.
    if [ "$downgraded" -gt 0 ]; then
        rm -f "$SYNCED" 2>/dev/null || true
    fi
    [ "$restored" -gt 0 ] \
        && echo "[unsloth-nb] restored $restored deleted notebook(s) from the baked set ($downgraded needing a refresh)"
fi

# 2) Best-effort GitHub refresh, detached because ls-remote + clone can spend 2x
# TIMEOUT when offline and must not delay container startup.
[ "${UNSLOTH_SKIP_NOTEBOOK_REFRESH:-0}" = "1" ] && exit 0
command -v git >/dev/null 2>&1 || exit 0
command -v sha256sum >/dev/null 2>&1 || exit 0
if [ "${UNSLOTH_NB_REFRESH_CHILD:-0}" != "1" ]; then
    # BEFORE the fork: the EXIT trap fired while the child was already copying
    finalize
    lock_release
    UNSLOTH_NB_REFRESH_CHILD=1 "$0" >/dev/null 2>&1 &
    exit 0
fi

# --- refresh child ---
# the parent already finalized, so re-arm below only if this refresh rewrites anything
_FINALIZED=1

last="$(cat "$SYNCED" 2>/dev/null || true)"
remote="$(timeout "$TIMEOUT" git ls-remote "$REMOTE" HEAD 2>/dev/null | cut -f1)"
[ -z "$remote" ] && exit 0            # offline: keep what we have
[ "$remote" = "$last" ] && exit 0

TMP="$(mktemp -d)"
if ! timeout "$TIMEOUT" git clone -q --depth 1 "$REMOTE" "$TMP" 2>/dev/null; then
    rm -rf "$TMP"; exit 0             # network died mid-way: keep what we have
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
            kept=$((kept + 1))
            continue
        fi
        if [ -n "$rec" ] && [ "$(hash_of "$dst")" != "$rec" ]; then
            printf '%s  %s\n' "$rec" "$rel" >> "$TMPSTATE"
            kept=$((kept + 1))
            continue
        fi
        if [ -n "$rec" ] && middle_unchanged "$dst" "$f"; then
            # a changed package spec is NOT cosmetic and falls through below
            printf '%s  %s\n' "$rec" "$rel" >> "$TMPSTATE"
            unchanged=$((unchanged + 1))
            continue
        fi
    elif [ -n "${LAST[$rel]:-}" ] && [ "${UNSLOTH_KEEP_DELETED_NOTEBOOKS:-0}" = "1" ]; then
        printf '%s  %s\n' "${LAST[$rel]}" "$rel" >> "$TMPSTATE"
        kept=$((kept + 1))
        continue
    fi
    mkdir_keep_owner "$(dirname "$dst")"
    # Same-dir temp + rename: JupyterLab is already serving $DEST, and an in-place
    # cp -a exposes half-written JSON and destroys a save made since the hash check.
    new="$(dirname "$dst")/.unsloth_nb_new.$$"
    if cp -a "$f" "$new" 2>/dev/null; then
        # hash what is about to be PUBLISHED: re-reading $dst after the rename would
        # record a save that landed meanwhile as the pristine version
        staged="$(hash_of "$new")"
        if [ -e "$dst" ] && [ "$(hash_of "$dst")" != "${LAST[$rel]:-}" ]; then
            rm -f "$new"
            printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
            kept=$((kept + 1))
            continue
        fi
        stage_metadata "$new" "$dst"
        # a single-FILE bind mount cannot be renamed over (EBUSY)
        if mv -f "$new" "$dst" 2>/dev/null || { rm -f "$new"; cp_keep_meta "$f" "$dst"; }; then
            # $staged, not hash_of "$dst": both branches write the bytes of "$f"
            printf '%s  %s\n' "$staged" "$rel" >> "$TMPSTATE"
            updated=$((updated + 1))
        else
            # carry the PREVIOUS record forward, or the next refresh reads $dst as
            # user-owned and keeps the stale copy forever
            printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
            failed=$((failed + 1))
        fi
    else
        printf '%s  %s\n' "${LAST[$rel]:-}" "$rel" >> "$TMPSTATE"
        failed=$((failed + 1))
    fi
done < <(find "$TMP" -type f -print0)

# Upstream deleted or renamed it. The loop above only walks the CLONE, so without
# this the copy we published stays on disk while its record is dropped: the next
# refresh reads it as user-owned, and unsloth_nb_view.py files it under "Other
# Notebooks" for good. Over the last year upstream deleted 10 and renamed 7 nb/
# notebooks, so these accumulate. Only a file that still hashes to what WE wrote is
# removed, so an edited notebook stays; it merely stops being managed, which it had
# already stopped being.
removed=0
if [ "${UNSLOTH_KEEP_REMOVED_NOTEBOOKS:-0}" != "1" ] && [ "${#LAST[@]}" -gt 0 ]; then
    # A case-only rename upstream looks like a deletion here, because the clone is on
    # a case-sensitive filesystem while $DEST may be a macOS or Windows bind mount
    # where the old path resolves to the file just published.
    declare -A CLONED_LOWER=()
    while IFS= read -r -d '' f; do
        p="${f#"$TMP"/}"
        case "$p" in .git|.git/*) continue ;; esac
        CLONED_LOWER["${p,,}"]=1
    done < <(find "$TMP" -type f -print0)
    for rel in "${!LAST[@]}"; do
        [ -e "$TMP/$rel" ] && continue
        [ -n "${CLONED_LOWER[${rel,,}]:-}" ] && continue
        dst="$DEST/$rel"
        [ -f "$dst" ] || continue
        [ -n "${LAST[$rel]}" ] || continue
        [ "$(hash_of "$dst")" = "${LAST[$rel]}" ] || continue
        if rm -f "$dst" 2>/dev/null; then
            removed=$((removed + 1))
            # one level, and never $DEST itself: `rmdir -p` would climb out of it
            d="$(dirname "$dst")"
            [ "$d" != "$DEST" ] && rmdir "$d" 2>/dev/null
        else
            # A writable single-FILE bind mount cannot be unlinked (EBUSY), the same
            # case the publish above has to work around. The file stays, so its record
            # has to stay with it: dropping it here would leave the stale copy on disk
            # with nothing claiming it, and the next refresh would read it as
            # user-owned and never retry. Counting it failed also holds the sync
            # marker back, which is what makes the retry happen at all.
            printf '%s  %s\n' "${LAST[$rel]}" "$rel" >> "$TMPSTATE"
            failed=$((failed + 1))
        fi
    done
    unset CLONED_LOWER
fi

mv "$TMPSTATE" "$STATE" 2>/dev/null || rm -f "$TMPSTATE"
# recording the commit after a failure makes the next boot exit on remote == last
if [ "$failed" -eq 0 ]; then
    echo "$remote" > "$SYNCED" 2>/dev/null || true
else
    echo "[unsloth-nb] $failed notebook(s) could not be written; leaving the sync marker so the next start retries"
fi
rm -rf "$TMP"
echo "[unsloth-nb] notebooks refreshed from GitHub: $updated updated, $kept kept (your edits), $unchanged kept (only header/footer changed upstream), $removed removed upstream"
if [ "$updated" -gt 0 ] || [ "$removed" -gt 0 ]; then
    _FINALIZED=0
    finalize
fi
exit 0
