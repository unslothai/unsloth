#!/usr/bin/env bash
# Link each code entry of $UNSLOTH_STUDIO_APP (venv, src, Node, whisper.cpp, tiers) into
# $UNSLOTH_STUDIO_HOME, which keeps only data and is what users mount a volume on; the
# venv's baked absolute paths keep resolving through the links. Runs at every start, so
# a volume holding an earlier image's real code directories is relinked rather than
# frozen at that version. Nothing is deleted: an entry in the way is renamed to
# .unsloth-studio-legacy/<name> (`--restore` puts it back for an older image;
# UNSLOTH_STUDIO_KEEP_LEGACY=0 deletes). Entries the app dir lacks are left alone.
set -euo pipefail

APP="${UNSLOTH_STUDIO_APP:-/opt/unsloth-studio-app}"
HOME_DIR="${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"
KEEP_LEGACY="${UNSLOTH_STUDIO_KEEP_LEGACY:-1}"
LEGACY_NAME=".unsloth-studio-legacy"

log() { echo "[unsloth-studio] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# mkdir -p, rm -rf and mv all follow a link planted at the legacy name (into the app
# dir, say); refuse rather than write through it.
check_legacy() {
    if [ -L "$1" ] || { [ -e "$1" ] && [ ! -d "$1" ]; }; then
        die "$1 must be a directory, not a link or a file; move or remove it, then start the container again"
    fi
}

# `--restore`: put the kept-aside entries back in place of the links, for an image from before the split.
if [ "${1:-}" = "--restore" ]; then
    legacy="$HOME_DIR/$LEGACY_NAME"
    check_legacy "$legacy"
    shopt -s dotglob nullglob
    if [ ! -d "$legacy" ]; then
        # No kept copy (first used after the split, or deleted): a pre-split image finds its
        # own tree hidden under the mount and links to nothing, so give it real copies.
        [ -d "$APP" ] || die "no $legacy to restore and no $APP on this image: run --restore under an image that has the Studio code in $APP (the one that last ran this volume), it copies that code into the home for an older image"
        copied=()
        for target in "$HOME_DIR"/*; do
            [ -L "$target" ] || continue
            link="$(readlink "$target")"
            case "$link" in "$APP"/*) ;; *) continue;; esac
            name="${target##*/}"
            # the uv cache is this image's scratch (9 GB); no image before the split reads it
            if [ -e "$link" ] && [ "$name" != "uv-cache" ]; then
                # copy beside the link and swap only once whole: a half-written real
                # $target is skipped by this symlink-only loop on the rerun
                tmp="$target.restore-tmp"
                rm -rf -- "$tmp"
                cp -a -- "$link" "$tmp" || {
                    rm -rf -- "$tmp"
                    die "cannot copy $link to $target; nothing was changed, rerun --restore"
                }
                rm -f -- "$target"
                mv -T -- "$tmp" "$target" || die "cannot put $tmp at $target; move it there by hand"
                copied+=("$name")
            else
                rm -f -- "$target"
            fi
        done
        if [ "${#copied[@]}" -gt 0 ]; then
            log "no kept-aside copy on this volume; copied this image's code into the home instead (${copied[*]}), so an image from before the split can run it. The next start of a split image links its own code back in."
        else
            log "nothing to restore: no links into $APP and no $legacy"
        fi
        exit 0
    fi
    # every link of ours goes, kept copy or not: Docker copies nothing into a non-empty volume
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

# as the kernel sees them: an override pointing both at one tree would move the install aside
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
    check_legacy "$legacy"
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

# unsloth-studio-update renames src to .src-prev.* and the staged tree into place; a kill
# between the two leaves the previous tree as the only copy, so put it back before the
# loop below prunes the home's src link as dangling.
if [ ! -e "$APP/src" ]; then
    prev=("$APP"/.src-prev.*)
    if [ "${#prev[@]}" -eq 1 ] && [ -d "${prev[0]}" ]; then
        mv -T -- "${prev[0]}" "$APP/src" || die "cannot put ${prev[0]} back at $APP/src"
        log "put ${prev[0]} back at $APP/src: an update was interrupted between its two renames"
    elif [ "${#prev[@]}" -gt 1 ]; then
        log "WARNING: $APP/src is missing and several .src-prev.* trees exist; pick one and move it to $APP/src by hand"
    fi
fi

for entry in "$APP"/*; do
    name="${entry##*/}"
    [ "$name" = "$LEGACY_NAME" ] && continue
    # a killed update's staging and previous trees: scratch, never linked into the home
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

# an entry this image dropped (a transformers tier) leaves a dangling link; only links into the app dir are ours
for target in "$HOME_DIR"/*; do
    [ -L "$target" ] || continue
    link="$(readlink "$target")"
    case "$link" in
        # -L too: llama.cpp is a link in the app dir, and points outside the Studio home
        "$APP"/*) [ -e "$link" ] || [ -L "$link" ] || rm -f -- "$target";;
    esac
done

# A killed update (docker stop ends in SIGKILL) can leave its previous tree beside src
# and half-replaced packages; unsloth-studio-update keeps a record beside src until the
# update is committed. Finish that restore here, before supervisord starts Studio on the
# unverified tree. The home is linked by now, so the updater finds the venv.
if [ -s "$APP/.src-update.rollback" ]; then
    updater="${UNSLOTH_STUDIO_UPDATER:-/usr/local/bin/unsloth-studio-update}"
    if [ -x "$updater" ]; then
        log "an update was interrupted (record at $APP/.src-update.rollback); putting the previous install back before Studio starts"
        if UNSLOTH_STUDIO_HOME="$HOME_DIR" "$updater" --recover; then
            log "the previous install is back"
        else
            log "WARNING: the previous install could not be put back (see above); Studio may fail to start"
            log "  finish it by hand: docker exec <container> unsloth-studio-update --recover"
        fi
    else
        log "WARNING: an update was interrupted (record at $APP/.src-update.rollback) and $updater is not here to finish the restore"
    fi
fi

if [ "${#moved[@]}" -gt 0 ]; then
    size="$(du -sh -- "$legacy" 2>/dev/null | cut -f1)"
    log "kept aside in $legacy (${size:-?}): ${moved[*]}"
    log "  to go back to an image from before the code/data split, put them back first:"
    log "    docker run --rm -v <volume>:$HOME_DIR --entrypoint unsloth-studio-home <this image> --restore"
    log "  to free the space: rm -rf $legacy"
fi
