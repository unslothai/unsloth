#!/usr/bin/env bash
# Point the Studio home's code entries at this image's copy.
#
# Studio's code (venv, src, Node, whisper.cpp, transformers tiers) lives in
# $UNSLOTH_STUDIO_APP; $UNSLOTH_STUDIO_HOME keeps only its data and is what users mount
# a volume on. Each code entry in the home is a symlink into the app dir, so Studio sees
# one directory and the venv's baked absolute paths keep resolving.
#
# The entrypoint runs this at every start, so a volume holding an earlier image's code
# as real directories, which is what froze Studio at the old version, is relinked to
# this image's. Nothing in the home is deleted: an entry that is in the way (an earlier
# image's real directory, or anything the user put there under a code entry's name) is
# moved aside to $UNSLOTH_STUDIO_HOME/.unsloth-studio-legacy/<name>, a rename on the
# same filesystem. `unsloth-studio-home --restore` puts them back, which is how a volume
# goes back to an older image; UNSLOTH_STUDIO_KEEP_LEGACY=0 deletes instead.
# Anything the app dir does not have is left alone.
set -euo pipefail

APP="${UNSLOTH_STUDIO_APP:-/opt/unsloth-studio-app}"
HOME_DIR="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
KEEP_LEGACY="${UNSLOTH_STUDIO_KEEP_LEGACY:-1}"
LEGACY_NAME=".unsloth-studio-legacy"

log() { echo "[unsloth-studio] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# `--restore` puts the entries kept aside back in place of the links. Run it under an
# image that ships Studio inside the home (before the code/data split) to roll back.
if [ "${1:-}" = "--restore" ]; then
    legacy="$HOME_DIR/$LEGACY_NAME"
    [ -d "$legacy" ] || { log "nothing to restore: no $legacy"; exit 0; }
    shopt -s dotglob nullglob
    # every link of ours goes, kept copy or not: an older image expects real entries
    # and Docker copies nothing into a volume that is not empty
    for target in "$HOME_DIR"/*; do
        [ -L "$target" ] || continue
        case "$(readlink "$target")" in
            "$APP"/*) rm -f -- "$target"; log "unlinked $target";;
        esac
    done
    for entry in "$legacy"/*; do
        name="${entry##*/}"
        target="$HOME_DIR/$name"
        if [ -L "$target" ] || [ ! -e "$target" ]; then
            rm -f "$target"
        else
            die "$target is a real entry, not a link; move it away before restoring $entry"
        fi
        mv -T "$entry" "$target"
        log "restored $target"
    done
    rmdir "$legacy" 2>/dev/null || true
    exit 0
fi

[ -d "$APP" ] || exit 0
mkdir -p "$HOME_DIR" || die "cannot create $HOME_DIR"

# Compare the two roots as the kernel sees them: an env override that points both at
# the same tree, or nests one in the other, would otherwise move the install aside.
app_real="$(cd -P -- "$APP" && pwd -P)"
home_real="$(cd -P -- "$HOME_DIR" && pwd -P)"
case "$home_real/" in
    "$app_real/"*) die "UNSLOTH_STUDIO_HOME ($HOME_DIR) must not be UNSLOTH_STUDIO_APP ($APP) or inside it";;
esac
case "$app_real/" in
    "$home_real/"*) die "UNSLOTH_STUDIO_APP ($APP) must not be inside UNSLOTH_STUDIO_HOME ($HOME_DIR)";;
esac

legacy="$HOME_DIR/$LEGACY_NAME"
moved=()

# set_aside <path>: get an entry that is in the way out of the way without losing it.
set_aside() {
    local path="$1" name="${1##*/}" kept="$legacy/${1##*/}"
    if [ "$KEEP_LEGACY" = "0" ]; then
        rm -rf -- "$path" || die "cannot remove $path"
        log "removed $path (UNSLOTH_STUDIO_KEEP_LEGACY=0)"
        return
    fi
    mkdir -p -m 0700 "$legacy" || die "cannot create $legacy"
    # one generation only: a second upgrade replaces the copy kept by the first
    if [ -e "$kept" ] || [ -L "$kept" ]; then
        rm -rf -- "$kept" || die "cannot clear the earlier copy at $kept"
    fi
    mv -T -- "$path" "$kept" || die "cannot move $path aside to $kept; nothing was changed"
    moved+=("$name")
    log "kept $path aside at $kept, this image's copy is linked in its place"
}

shopt -s dotglob nullglob
for entry in "$APP"/*; do
    name="${entry##*/}"
    [ "$name" = "$LEGACY_NAME" ] && continue
    # unsloth-studio-update's staging and previous-tree directories, left in the app dir
    # by a killed update: scratch, not code; never linked into the home
    case "$name" in .src-update.*|.src-prev.*) continue;; esac
    target="$HOME_DIR/$name"
    if [ -L "$target" ]; then
        [ "$(readlink "$target")" = "$entry" ] && continue
        # a link of ours to an older app path, or a link the user made: keep the user's
        case "$(readlink "$target")" in
            "$APP"/*) rm -f -- "$target";;
            *) set_aside "$target";;
        esac
    elif [ -e "$target" ]; then
        set_aside "$target"
    fi
    ln -sT -- "$entry" "$target" \
        || die "cannot link $target -> $entry$( [ -e "$legacy/$name" ] && echo "; the earlier entry is intact at $legacy/$name")"
done

# An entry an earlier image had and this one dropped, e.g. a transformers tier, is left
# as a link to nothing. Only links into the app dir are ours to remove.
for target in "$HOME_DIR"/*; do
    [ -L "$target" ] || continue
    link="$(readlink "$target")"
    case "$link" in
        # -L too: llama.cpp is a link in the app dir, and points outside the Studio home
        "$APP"/*) [ -e "$link" ] || [ -L "$link" ] || rm -f -- "$target";;
    esac
done

if [ "${#moved[@]}" -gt 0 ]; then
    size="$(du -sh -- "$legacy" 2>/dev/null | cut -f1)"
    log "kept aside in $legacy (${size:-?}): ${moved[*]}"
    log "  to go back to an image from before the code/data split, put them back first:"
    log "    docker run --rm -v <volume>:$HOME_DIR --entrypoint unsloth-studio-home <this image> --restore"
    log "  to free the space: rm -rf $legacy"
fi
