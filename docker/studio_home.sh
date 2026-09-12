#!/usr/bin/env bash
# Point the Studio home's code entries at this image's copy.
#
# Studio's code (venv, src, Node, whisper.cpp, transformers tiers, caches) lives in
# $UNSLOTH_STUDIO_APP; $UNSLOTH_STUDIO_HOME keeps only its data and is what users mount
# a volume on. Each code entry in the home is a symlink into the app dir, so Studio sees
# one directory and the venv's baked absolute paths keep resolving.
#
# The entrypoint runs this at every start, so a volume holding an earlier image's code
# as real directories, which is what froze Studio at the old version, is relinked to
# this image's. Anything the app dir does not have is left alone.
set -euo pipefail

APP="${UNSLOTH_STUDIO_APP:-/opt/unsloth-studio-app}"
HOME_DIR="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
[ -d "$APP" ] || exit 0
mkdir -p "$HOME_DIR"

shopt -s dotglob nullglob
for entry in "$APP"/*; do
    name="${entry##*/}"
    target="$HOME_DIR/$name"
    if [ -L "$target" ]; then
        [ "$(readlink "$target")" = "$entry" ] && continue
        rm -f "$target"
    elif [ -e "$target" ]; then
        echo "[unsloth-studio] replacing $target, left in the Studio home by an earlier image, with this image's copy" >&2
        rm -rf "$target"
    fi
    ln -s "$entry" "$target"
done

# An entry an earlier image had and this one dropped, e.g. a transformers tier, is left
# as a link to nothing. Only links into the app dir are ours to remove.
for target in "$HOME_DIR"/*; do
    [ -L "$target" ] || continue
    link="$(readlink "$target")"
    case "$link" in
        # -L too: llama.cpp is a link in the app dir, and points outside the Studio home
        "$APP"/*) [ -e "$link" ] || [ -L "$link" ] || rm -f "$target";;
    esac
done
