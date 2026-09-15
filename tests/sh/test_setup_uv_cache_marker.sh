#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# studio/setup.sh (the `unsloth studio update` entry point) used to pick $STUDIO_HOME/cache/uv
# unconditionally, the EMPTY one on a shared-mode install, and refetched every wheel. It now
# reads install.sh's marker on the CLI's terms (unsloth_cli/commands/studio.py): an explicit
# UV_CACHE_DIR wins and is never recorded, UV_NO_CACHE leaves it unset, the marker wins only
# while its cache holds packages, otherwise the Studio path if writable. setup.sh never
# WRITES the marker: a recorded inference outlives its install.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

# The real helpers and selector, sliced out of setup.sh by their anchors (the selector is not a function).
HELPERS=$(awk '
    /^_uv_is_bucket_name\(\) \{/ { grab = 1 }
    /^_uv_no_cache_requested\(\) \{/ { grab = 1 }
    /^_uv_cache_probe_writable\(\) \{/ { grab = 1 }
    /^_uv_cache_folds_case\(\) \{/ { grab = 1 }
    /^_uv_store_key\(\) \{/ { grab = 1 }
    /^_uv_cache_usable\(\) \{/ { grab = 1 }
    /^_uv_control_files_writable\(\) \{/ { grab = 1 }
    /^_uv_cache_warm\(\) \{/ { grab = 1 }
    /^_recorded_uv_cache\(\) \{/ { grab = 1 }
    /^_UV_MARKER_BOM=/ { print; next }
    /^_UV_MARKER_CR=/ { print; next }
    /^_UV_MARKER_LF=/ { print; next }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")
SELECTOR=$(awk '
    /^_uv_caller_value=false$/ { grab = 1 }
    grab { print }
    grab && /^fi$/ { exit }
' "$SETUP_SH")

for _need in _uv_is_bucket_name _uv_no_cache_requested _uv_cache_probe_writable _uv_cache_folds_case _uv_store_key _uv_cache_usable _uv_control_files_writable _uv_cache_warm _recorded_uv_cache; do
    if ! printf '%s\n' "$HELPERS" | grep -q "^${_need}() {"; then
        echo "FATAL: could not extract $_need from setup.sh" >&2
        exit 1
    fi
done
printf '%s\n' "$HELPERS" | grep -q '^_UV_MARKER_BOM=' || {
    echo "FATAL: the BOM constant is gone" >&2; exit 1; }
printf '%s\n' "$SELECTOR" | grep -q '_uv_cache_warm "\$_uv_recorded"' || {
    echo "FATAL: could not extract the selector from setup.sh" >&2; exit 1; }

PROBE_HELPERS="$WORK/helpers.sh"
printf '%s\n' "$HELPERS" > "$PROBE_HELPERS"
PROBE="$WORK/probe.sh"
{
    printf '%s\n' "$HELPERS"
    cat <<'PROBE_BODY'
case "$1" in
    unset) unset UV_CACHE_DIR ;;
    value) UV_CACHE_DIR=$2 ;;
    *) exit 2 ;;
esac
case "$3" in
    unset) unset UV_NO_CACHE ;;
    value) UV_NO_CACHE=$4 ;;
    *) exit 2 ;;
esac
STUDIO_HOME=$5
PROBE_BODY
    printf '%s\n' "$SELECTOR"
    cat <<'PROBE_TAIL'
printf '%s' "${UV_CACHE_DIR-<unset>}"
PROBE_TAIL
} > "$PROBE"

warm() {  # warm <cache dir> [bucket] [filename]
    mkdir -p "$1/${2:-archive-v0}/pkg"
    : > "$1/${2:-archive-v0}/pkg/${3:-payload.whl}"
}

record() {  # record <studio home> <bytes...>
    mkdir -p "$1/cache"
    printf "$2" > "$1/cache/uv-cache-dir"
}

run() {  # run <shell> <state> <input> <no-cache-state> <no-cache> <studio home>
    "$1" "$PROBE" "$2" "$3" "$4" "$5" "$6"
}

# The same probe under setup.sh's own options (line 5): `set -o pipefail` turns a SIGPIPE inside
# a command substitution into a failed pipeline, the difference between warm and refetched.
run_strict() {  # run_strict <state> <input> <no-cache-state> <no-cache> <studio home>
    bash -e -u -o pipefail "$PROBE" "$1" "$2" "$3" "$4" "$5"
}

# A bucket with more names than a 64K pipe holds, built once (touching it is the slow part).
# `find ... -print | head -n 1` reads THIS cache as cold under pipefail; every other fixture is
# a handful of files, which is why the suite stayed green while updates refetched warm caches.
BIG="$WORK/big cache/uv"
mkdir -p "$BIG/archive-v0/pkg"
awk -v d="$BIG/archive-v0/pkg" 'BEGIN { for (i = 0; i < 4000; i++)
    printf "%s/wheel_payload_%05d.whl%c", d, i, 0 }' | xargs -0 touch

echo "=== test_setup_uv_cache_marker ==="
for shell in sh bash; do
    command -v "$shell" >/dev/null 2>&1 || continue
    CASE="$WORK/$shell case"
    HOME_DIR="$CASE/studio home"
    STUDIO_CACHE="$HOME_DIR/cache/uv"
    SHARED="$CASE/shared cache/uv"
    mkdir -p "$HOME_DIR"

    assert_eq "$shell: no marker falls back to the Studio cache" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    warm "$SHARED"
    record "$HOME_DIR" "$SHARED\\n"
    assert_eq "$shell: a warm recorded cache is adopted" \
        "$SHARED" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # Same file, written by Windows PowerShell 5.1 `-Encoding utf8`: BOM plus CRLF.
    record "$HOME_DIR" "\\357\\273\\277$SHARED\\r\\n"
    assert_eq "$shell: a BOM and a CRLF do not hide the path" \
        "$SHARED" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A recorded cache gone read-only since the install: warm, and useless to uv, which aborts on it.
    RO="$CASE/read-only recorded/uv"
    warm "$RO"
    record "$HOME_DIR" "$RO\\n"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 555 "$RO" 2>/dev/null; then
        assert_eq "$shell: a warm recorded cache that is not writable loses to the Studio cache" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 755 "$RO" 2>/dev/null || true
    fi

    # A cache the user cleared: content, not the record, has the last word on emptiness.
    COLD="$CASE/emptied cache/uv"
    mkdir -p "$COLD"
    record "$HOME_DIR" "$COLD\\n"
    assert_eq "$shell: an emptied recorded cache loses to the default" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # uv 0.10 leaves .msgpack/.http in wheels-* after a bare resolve; that is not warm.
    META="$CASE/metadata only/uv"
    warm "$META" wheels-v6 resolve.msgpack
    warm "$META" wheels-v6 wheel.http
    record "$HOME_DIR" "$META\\n"
    assert_eq "$shell: metadata alone is not a warm cache" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    warm "$META" wheels-v6 torch.whl
    assert_eq "$shell: package bytes beside metadata do count" \
        "$META" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A writable root is not a usable cache: uv unpacks into the buckets, so a root-only probe
    # adopts a recorded cache uv then aborts on (uv 0.10.7: "failed to rename", exit 1).
    BLOCKED="$CASE/bucket blocked/uv"
    warm "$BLOCKED"
    record "$HOME_DIR" "$BLOCKED\\n"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$BLOCKED/archive-v0" 2>/dev/null; then
        assert_eq "$shell: a recorded cache with an unwritable bucket falls back to Studio" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0755 "$BLOCKED/archive-v0" 2>/dev/null || true
    fi
    assert_eq "$shell: the same cache is adopted once its bucket is writable again" \
        "$BLOCKED" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # Store versions reach two digits (uv 0.12.1 ships simple-v24), and `*-v[0-9]*` is "a digit
    # then anything", so it matches those too. This pins that against a narrower glob.
    MULTI="$CASE/multi-digit store/uv"
    warm "$MULTI"
    mkdir -p "$MULTI/simple-v24"
    record "$HOME_DIR" "$MULTI\\n"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$MULTI/simple-v24" 2>/dev/null; then
        assert_eq "$shell: an unwritable multi-digit store falls back to Studio" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0755 "$MULTI/simple-v24" 2>/dev/null || true
    fi

    # `archive-*` also matches `archive-v0.backup`, whose bytes uv cannot reuse. Counting them
    # reads this cache as warm and then fails the offline update it was picked for (measured on
    # uv 0.10.7: exit 1, "not found in the cache"). install.sh's scan rejects it the same way.
    LOOKALIKE="$CASE/lookalike bucket/uv"
    mkdir -p "$LOOKALIKE/archive-v0.backup/pkg"
    : > "$LOOKALIKE/archive-v0.backup/pkg/torch.whl"
    record "$HOME_DIR" "$LOOKALIKE\\n"
    assert_eq "$shell: a lookalike bucket is not warmth" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A store path occupied by a FILE is an existing path to mkdir(2), so uv cannot create the
    # store and aborts (measured: a plain file at archive-v0 exits 1, at interpreter-v4,
    # sdists-v9, simple-v20 or wheels-v6 exits 2). Skipping it would call the cache usable.
    BLOCKFILE="$CASE/store is a file/uv"
    warm "$BLOCKFILE"
    : > "$BLOCKFILE/interpreter-v4"
    record "$HOME_DIR" "$BLOCKFILE\\n"
    assert_eq "$shell: a file where a store belongs is not usable" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    rm -f "$BLOCKFILE/interpreter-v4"
    assert_eq "$shell: and the same cache is adopted once it is gone" \
        "$BLOCKFILE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # `*-v[0-9]*/.lock` also reaches `unused-v999`, a kind `_uv_is_bucket_name` rejects on
    # purpose. A read-only control-looking file there condemned a cache real uv uses fine
    # (measured: probe REJECT, uv exit 0), so the standalone update lost its warm cache.
    STRAY_CTL="$CASE/stray control file/uv"
    warm "$STRAY_CTL"
    mkdir -p "$STRAY_CTL/unused-v999"
    : > "$STRAY_CTL/unused-v999/.lock"
    record "$HOME_DIR" "$STRAY_CTL\\n"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0444 "$STRAY_CTL/unused-v999/.lock" 2>/dev/null; then
        assert_eq "$shell: a control file outside uv's stores does not condemn the cache" \
            "$STRAY_CTL" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0644 "$STRAY_CTL/unused-v999/.lock" 2>/dev/null || true
    fi
    # ...while a .git inside a store still does. Measured on uv 0.10.7: sdists-v9/.git at 0444
    # aborts with "Permission denied", exit 2. uv creates no per-store control files itself, so
    # this one is someone else's, and it is the one proven to break uv.
    mkdir -p "$STRAY_CTL/sdists-v9"
    : > "$STRAY_CTL/sdists-v9/.git"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0444 "$STRAY_CTL/sdists-v9/.git" 2>/dev/null; then
        assert_eq "$shell: a read-only .git inside a store still does" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0644 "$STRAY_CTL/sdists-v9/.git" 2>/dev/null || true
    fi
    # And the root .lock, the other one measured to abort (exit 2).
    : > "$STRAY_CTL/.lock"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0444 "$STRAY_CTL/.lock" 2>/dev/null; then
        assert_eq "$shell: a read-only root .lock is not usable either" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0644 "$STRAY_CTL/.lock" 2>/dev/null || true
    fi
    # A CACHEDIR.TAG at 0444 installs fine (measured), so it must not cost the warm cache.
    : > "$STRAY_CTL/CACHEDIR.TAG"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0444 "$STRAY_CTL/CACHEDIR.TAG" 2>/dev/null; then
        assert_eq "$shell: a read-only CACHEDIR.TAG does not, since uv tolerates it" \
            "$STRAY_CTL" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0644 "$STRAY_CTL/CACHEDIR.TAG" 2>/dev/null || true
    fi

    # uv cannot open a `.lock` that is not a regular file. Measured on uv 0.10.7: a directory
    # there, and a symlink to one, both exit 2 with "Could not acquire lock ... Is a directory".
    LOCKDIR="$CASE/lock is a directory/uv"
    warm "$LOCKDIR"
    mkdir -p "$LOCKDIR/.lock"
    record "$HOME_DIR" "$LOCKDIR\\n"
    assert_eq "$shell: a .lock directory is not a usable cache" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    rmdir "$LOCKDIR/.lock"
    assert_eq "$shell: and the same cache is adopted once it is a file again" \
        "$LOCKDIR" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    mkdir -p "$CASE/lock target"
    ln -s "$CASE/lock target" "$LOCKDIR/.lock"
    assert_eq "$shell: a .lock symlinked to a directory is not either" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    rm -f "$LOCKDIR/.lock"

    # The fold, at the two call sites rather than only in the helper. ext4 does not fold, so the
    # measurement is stubbed both ways and what is pinned is that each scan acts on the answer.
    fold_probe() {  # fold_probe <shell> <folds:0|1> <expr>
        # _FOLDS, not $2: inside the stub $2 is the STUB's argument, not the script's, so the
        # answer read as empty and every fold case passed for the wrong reason.
        "$1" -c '. "$1"
_FOLDS=$2
_uv_cache_folds_case() { [ "$_FOLDS" = 1 ]; }
if eval "$3"; then echo yes; else echo no; fi' _ "$PROBE_HELPERS" "$2" "$3"
    }
    FOLDED="$CASE/folded store/uv"
    mkdir -p "$FOLDED/Archive-V0/pkg"
    : > "$FOLDED/Archive-V0/pkg/torch.whl"
    assert_eq "$shell: a folded bucket counts as warmth on a folding filesystem" \
        "yes" "$(fold_probe "$shell" 1 '_uv_cache_warm "'"$FOLDED"'"')"
    assert_eq "$shell: and does not where the filesystem is case-sensitive" \
        "no" "$(fold_probe "$shell" 0 '_uv_cache_warm "'"$FOLDED"'"')"

    # Same for the control file: `Sdists-V9` passes the folded store test, so the .git inside it
    # has to be probed under uv's spelling, not the raw one.
    FOLDCTL="$CASE/folded sdists/uv"
    mkdir -p "$FOLDCTL/archive-v0/pkg" "$FOLDCTL/Sdists-V9"
    : > "$FOLDCTL/archive-v0/pkg/torch.whl"
    : > "$FOLDCTL/Sdists-V9/.git"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0444 "$FOLDCTL/Sdists-V9/.git" 2>/dev/null; then
        assert_eq "$shell: a read-only .git in a folded sdists store is caught" \
            "no" "$(fold_probe "$shell" 1 '_uv_cache_usable "'"$FOLDCTL"'"')"
        assert_eq "$shell: and is left alone where the name is not uv's" \
            "yes" "$(fold_probe "$shell" 0 '_uv_cache_usable "'"$FOLDCTL"'"')"
        chmod 0644 "$FOLDCTL/Sdists-V9/.git" 2>/dev/null || true
    fi

    # A folding filesystem opens `Archive-V0` as the store uv writes at `archive-v0`, and a
    # lowercase glob does not fold, so the warm scan read a full cache as cold and the offline
    # update failed. Both scans go through _uv_store_key now, so they agree on the entry.
    assert_eq "$shell: a folded store name resolves to uv's spelling" \
        "archive-v0" "$($shell -c '. "$1"; _uv_store_key "Archive-V0" 1' _ "$PROBE_HELPERS")"
    assert_eq "$shell: and is not claimed on a case-sensitive filesystem" \
        "" "$($shell -c '. "$1"; _uv_store_key "Archive-V0" 0 || true' _ "$PROBE_HELPERS")"
    assert_eq "$shell: a lookalike is never a store, folding or not" \
        "" "$($shell -c '. "$1"; _uv_store_key "Archive-V0.backup" 1 || true' _ "$PROBE_HELPERS")"

    # The FALLBACK Studio cache gets the same check as a recorded one. A root-only probe passes
    # on a Studio cache whose archive-v0 went read-only, and uv then aborts (measured: exit 1,
    # "Permission denied") instead of falling back to uv's own.
    SICK="$CASE/sick studio/cache/uv"
    mkdir -p "$SICK/archive-v0/pkg"
    : > "$SICK/archive-v0/pkg/x.whl"
    rm -f "$CASE/sick studio/cache/uv-cache-dir"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$SICK/archive-v0" 2>/dev/null; then
        assert_eq "$shell: an unusable Studio cache is dropped, not exported" \
            "<unset>" "$(run "$shell" unset "" unset "" "$CASE/sick studio" 2>/dev/null)"
        chmod 0755 "$SICK/archive-v0" 2>/dev/null || true
    fi
    assert_eq "$shell: and is used again once its store is writable" \
        "$SICK" "$(run "$shell" unset "" unset "" "$CASE/sick studio")"

    # Only the stores `uv pip install` writes, since that is the one uv command this file runs.
    # Measured on uv 0.10.7 at 0555: binaries-v0, environments-v2, flat-index-v2, git-v0, osv-v0
    # and python-v0 all install fine, so probing them only threw the warm cache away.
    OFFSCOPE="$CASE/unrelated store/uv"
    warm "$OFFSCOPE"
    record "$HOME_DIR" "$OFFSCOPE\\n"
    for store in binaries-v0 osv-v0 environments-v2 python-v0 flat-index-v2; do
        mkdir -p "$OFFSCOPE/$store"
        if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$OFFSCOPE/$store" 2>/dev/null; then
            assert_eq "$shell: a read-only $store does not condemn the cache" \
                "$OFFSCOPE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
            chmod 0755 "$OFFSCOPE/$store" 2>/dev/null || true
        fi
    done
    # ...while a store pip install DOES write still does. git-v0 counts: a `git+` requirement
    # writes it (measured: a git install creates git-v0 and builds-v0).
    for store in archive-v0 git-v0 builds-v0; do
        mkdir -p "$OFFSCOPE/$store"
        if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$OFFSCOPE/$store" 2>/dev/null; then
            assert_eq "$shell: a read-only $store still does" \
                "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
            chmod 0755 "$OFFSCOPE/$store" 2>/dev/null || true
        fi
    done

    # One level inside the INDEX stores, which uv rewrites on every resolve. Measured on uv
    # 0.10.7: a 0555 shard under simple-v20 or wheels-v6 aborts with "Failed to write to the
    # client cache", exit 2, while one under archive-v0 or interpreter-v4 installs fine.
    SHARD="$CASE/nested shard/uv"
    warm "$SHARD"
    mkdir -p "$SHARD/simple-v20/pypi" "$SHARD/wheels-v6/pypi" "$SHARD/interpreter-v4/abcd"
    record "$HOME_DIR" "$SHARD\\n"
    for blocked in simple-v20/pypi wheels-v6/pypi; do
        if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$SHARD/$blocked" 2>/dev/null; then
            assert_eq "$shell: an unwritable $blocked shard falls back to Studio" \
                "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
            chmod 0755 "$SHARD/$blocked" 2>/dev/null || true
        fi
    done
    # ...and not deeper than that, nor in the content stores, where uv tolerates it and
    # rejecting would throw away the warm cache over a shard it never rewrites.
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$SHARD/interpreter-v4/abcd" 2>/dev/null; then
        assert_eq "$shell: an unwritable interpreter shard does not" \
            "$SHARD" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0755 "$SHARD/interpreter-v4/abcd" 2>/dev/null || true
    fi
    mkdir -p "$SHARD/simple-v20/pypi/deeper"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$SHARD/simple-v20/pypi/deeper" 2>/dev/null; then
        assert_eq "$shell: nor a directory two levels down" \
            "$SHARD" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0755 "$SHARD/simple-v20/pypi/deeper" 2>/dev/null || true
    fi

    # A non-directory AT the index leaf. Measured on the pinned uv 0.12.1: a plain file and a
    # dangling symlink at simple-*/pypi or wheels-*/pypi both abort with "Failed to write to
    # the client cache", exit 2. A `*/` glob matched neither, so the cache read as usable.
    LEAFND="$CASE/leaf not a dir/uv"
    warm "$LEAFND"
    mkdir -p "$LEAFND/simple-v20"
    : > "$LEAFND/simple-v20/pypi"
    record "$HOME_DIR" "$LEAFND\\n"
    assert_eq "$shell: a file at the index leaf is not a usable cache" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    rm -f "$LEAFND/simple-v20/pypi"
    ln -s "$LEAFND/nowhere" "$LEAFND/simple-v20/pypi"
    assert_eq "$shell: nor a dangling symlink there" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    # A symlink TO a directory is not the same thing and must stay adopted: uv writes through
    # it (measured on the pinned uv 0.12.1, installs fine), and rejecting every link here would
    # discard a cache whose index leaf is deliberately relocated.
    rm -f "$LEAFND/simple-v20/pypi"
    mkdir -p "$CASE/relocated index"
    ln -s "$CASE/relocated index" "$LEAFND/simple-v20/pypi"
    assert_eq "$shell: a leaf symlinked to a real directory is still adopted" \
        "$LEAFND" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    rm -f "$LEAFND/simple-v20/pypi"
    mkdir -p "$LEAFND/simple-v20/pypi"
    assert_eq "$shell: and it is adopted once the leaf is a directory" \
        "$LEAFND" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A CUSTOM --index-url, which Studio uses for the torch wheels, puts metadata at
    # `<store>/index/<hash>`, not `<store>/pypi`. Measured on the pinned uv 0.12.1: a 0555
    # `simple-*/index/<hash>` was ADOPTED by a one-level probe and then aborted the install
    # with "Failed to write to the client cache". The only unsafe direction found so far.
    CIDX="$CASE/custom index/uv"
    warm "$CIDX"
    mkdir -p "$CIDX/simple-v20/index/e1d141a6ca947dff" "$CIDX/wheels-v6/index/e1d141a6ca947dff/idna"
    record "$HOME_DIR" "$CIDX\\n"
    for blocked in simple-v20/index/e1d141a6ca947dff wheels-v6/index/e1d141a6ca947dff; do
        if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$CIDX/$blocked" 2>/dev/null; then
            assert_eq "$shell: an unwritable $blocked falls back to Studio" \
                "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
            chmod 0755 "$CIDX/$blocked" 2>/dev/null || true
        fi
    done
    # ...and no deeper: the level below the hash is one directory per package.
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 0555 "$CIDX/wheels-v6/index/e1d141a6ca947dff/idna" 2>/dev/null; then
        assert_eq "$shell: a package directory under the hash does not" \
            "$CIDX" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 0755 "$CIDX/wheels-v6/index/e1d141a6ca947dff/idna" 2>/dev/null || true
    fi

    # An all-whitespace UV_CACHE_DIR is not a caller's choice. install.sh's selector decides
    # this with `case *[![:space:]]*`; a plain `-n` here would read the same value as a choice
    # and hand uv `--cache-dir '   '`, so the two selectors would answer differently for one
    # environment. Whitespace-only falls through to the marker, exactly as it does in install.sh.
    WS="$CASE/whitespace/uv"
    warm "$WS"
    record "$HOME_DIR" "$WS\\n"
    assert_eq "$shell: an all-whitespace UV_CACHE_DIR is not a caller value" \
        "$WS" "$(run "$shell" value "   " unset "" "$HOME_DIR")"
    assert_eq "$shell: a tab-only UV_CACHE_DIR is not a caller value" \
        "$WS" "$(run "$shell" value "$(printf '\t')" unset "" "$HOME_DIR")"
    # ...while one real character in it still is.
    assert_eq "$shell: a caller value with surrounding space is still a caller value" \
        " /caller/uv " "$(run "$shell" value " /caller/uv " unset "" "$HOME_DIR")"

    for bucket in archive-v0 builds-v0 built-wheels-v3 sdists-v9; do
        BUCKET_CACHE="$CASE/$bucket/uv"
        warm "$BUCKET_CACHE" "$bucket"
        record "$HOME_DIR" "$BUCKET_CACHE\\n"
        assert_eq "$shell: $bucket counts as package data" \
            "$BUCKET_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    done

    # ...and one warm by thousands of files: the old `-print | head -n 1` scan read it as cold.
    record "$HOME_DIR" "$BIG\\n"
    assert_eq "$shell: a cache too big for one pipe buffer is still warm" \
        "$BIG" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    if [ "$shell" = bash ]; then
        assert_eq "$shell: ...under setup.sh's own set -euo pipefail too" \
            "$BIG" "$(run_strict unset "" unset "" "$HOME_DIR")"
    fi

    # ...and one with a leaf this user cannot read, walked before the hit (find exits nonzero
    # after it, even once the hit printed, and `|| _uvw_hit=""` threw the hit away). Denied
    # leaves are created and named first, and added until `ls -f` shows one ahead of the hit.
    DENIED="$CASE/denied leaf/uv"
    mkdir -p "$DENIED/archive-v0/hidden 1"
    : > "$DENIED/archive-v0/hidden 1/other.whl"
    chmod 000 "$DENIED/archive-v0/hidden 1" 2>/dev/null || true
    warm "$DENIED" archive-v0 visible.whl
    _leaf=1
    while [ "$_leaf" -lt 40 ] && \
        [ "$(ls -f "$DENIED/archive-v0" | grep -v '^\.\.*$' | head -n 1)" = pkg ]; do
        _leaf=$((_leaf + 1))
        mkdir -p "$DENIED/archive-v0/hidden $_leaf"
        : > "$DENIED/archive-v0/hidden $_leaf/other.whl"
        chmod 000 "$DENIED/archive-v0/hidden $_leaf" 2>/dev/null || true
    done
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && [ ! -r "$DENIED/archive-v0/hidden 1" ]; then
        record "$HOME_DIR" "$DENIED\\n"
        assert_eq "$shell: a warm cache with an unreadable leaf is still warm" \
            "$DENIED" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        if [ "$shell" = bash ]; then
            assert_eq "$shell: ...under setup.sh's own set -euo pipefail too" \
                "$DENIED" "$(run_strict unset "" unset "" "$HOME_DIR")"
        fi
    fi
    for _leaf in "$DENIED/archive-v0"/hidden*; do
        chmod 755 "$_leaf" 2>/dev/null || true
    done

    # A relative record names a different directory in each phase: declined, not guessed at.
    record "$HOME_DIR" "relative/cache\\n"
    assert_eq "$shell: a relative record is declined" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    record "$HOME_DIR" "\\n"
    assert_eq "$shell: an empty record is declined" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A pathname ending in a newline: the CLI writes one delimiter and removes exactly one; a
    # reader that let substitution strip every trailing newline checked another directory.
    NLCACHE="$CASE/trailing newline"
    NLCACHE="$NLCACHE$(printf '\nx')"
    NLCACHE=${NLCACHE%x}
    warm "$NLCACHE"
    record "$HOME_DIR" "$NLCACHE\\n"
    assert_eq "$shell: a pathname ending in a newline round-trips" \
        "${NLCACHE}x" "$(run "$shell" unset "" unset "" "$HOME_DIR"; printf x)"

    # A caller value outranks every inference, and is never written anywhere.
    record "$HOME_DIR" "$SHARED\\n"
    OVERRIDE="$CASE/caller cache"
    assert_eq "$shell: an explicit UV_CACHE_DIR wins over the marker" \
        "$OVERRIDE" "$(run "$shell" value "$OVERRIDE" unset "" "$HOME_DIR")"

    for truthy in 1 true TRUE yes ON y t; do
        assert_eq "$shell: UV_NO_CACHE=[$truthy] leaves the cache unset" \
            "<unset>" "$(run "$shell" unset "" value "$truthy" "$HOME_DIR")"
    done
    # An exported EMPTY UV_CACHE_DIR is not a caller value; uv fails on it, so no-cache mode unsets it.
    assert_eq "$shell: an empty UV_CACHE_DIR under UV_NO_CACHE is unset, not kept" \
        "<unset>" "$(run "$shell" value "" value 1 "$HOME_DIR")"
    # A padded value is not truthy: clap rejects ` on ` outright rather than reading it as
    # true, so uv's cache stays ON and the selection must not stand down. install.sh and the
    # CLI both strip nothing for the same reason.
    for falsy in 0 false "" maybe " on "; do
        assert_eq "$shell: UV_NO_CACHE=[$falsy] changes nothing" \
            "$SHARED" "$(run "$shell" unset "" value "$falsy" "$HOME_DIR")"
    done

    # uv aborts on a cache it cannot create: an unwritable Studio path unsets rather than exports.
    BLOCKED="$CASE/blocked"
    : > "$BLOCKED"
    assert_eq "$shell: an uncreatable Studio cache is dropped, not exported" \
        "<unset>" "$(run "$shell" unset "" unset "" "$BLOCKED")"

    # ...and the probe file does not survive into the cache uv then fills.
    PROBED="$CASE/probed home"
    run "$shell" unset "" unset "" "$PROBED" >/dev/null
    assert_eq "$shell: the write probe cleans up after itself" \
        "" "$(ls -A "$PROBED/cache/uv" 2>/dev/null)"
done

# setup.sh must never become a marker writer: only an installer's own choice is one.
_writes=$(awk '
    /^_uv_is_bucket_name\(\) \{/ { grab = 1 }
    /^_uv_no_cache_requested\(\) \{/ { grab = 1 }
    /^if \[ -n "\$\{UV_CACHE_DIR:-\}" \]; then$/ { grab = 1 }
    grab { print }
    grab && /^fi$/ { exit }
' "$SETUP_SH" | grep -c 'uv-cache-dir"' || true)
assert_eq "the block reads the marker and never writes it" "1" "$_writes"
if grep -n 'uv-cache-dir' "$SETUP_SH" | grep -vq 'cat "\$STUDIO_HOME/cache/uv-cache-dir"'; then
    bad "setup.sh names the uv cache marker somewhere other than the one read"
else
    ok "setup.sh names the uv cache marker exactly once, to read it"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
