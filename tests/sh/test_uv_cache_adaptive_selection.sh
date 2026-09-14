#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards that _configure_uv_cache's adaptive selection is REACHABLE. Both halves of install.sh
# read correctly on their own and the defect lived in the ORDER they run, so this runs the early
# UV_CACHE_DIR block and _configure_uv_cache in that order rather than inspecting either half.
#
# The contract:
#   * caller-set UV_CACHE_DIR            -> custom, left exactly as the caller wrote it
#   * unset, uv's default is populated   -> shared, and the launch repoint becomes live
#   * unset, uv's default is empty       -> studio
#   * populated but not writable, root or bucket -> studio; uv aborts on either
#   * a relative cache-dir               -> resolved against UV_WORKING_DIR before scanning
#   * a bucket too big for one pipe read -> still warm, pipefail or not
#   * a caller's set -f                  -> the scan still expands its own globs
#   * a dangling bucket link             -> studio; mkdir(2) answers EEXIST on it
#   * a bucket-name lookalike            -> ignored; only <kind>-v<N> is uv's to write
#   * a name ending in `-v`               -> ignored; the version cannot be empty
#   * UV_NO_CACHE                        -> studio, with nothing probed or recorded
#   * --isolated-uv-cache                -> isolated, whatever else is true
#   * unwritable STUDIO_HOME             -> the early block unsets, and the choice still runs
#   * an unmarked warm Studio cache      -> kept; that is every install from before the marker
#   * a BOM/CRLF marker                  -> honoured; install.ps1 writes one, WSL shares a home
#   * a trailing slash in the marker     -> the same directory, so still `studio`
#   * shared with an unwritable root     -> the launch keeps the shared cache, not a dead one
#   * every shell that can be /bin/sh    -> same answers, with and without errexit
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_TMP=$(mktemp -d)
_EARLY=$(mktemp)
_BOOT=$(mktemp)
_FN=$(mktemp)
trap 'rm -rf "$_TMP" "$_EARLY" "$_FN" "$_BOOT"' EXIT

# The real code, anchored on text that PREDATES the fix so this suite also runs against the
# old code and fails on an assertion, not on an empty extraction.
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" > "$_EARLY"
awk '/^_configure_uv_cache\(\) \{$/,/^\}$/' "$INSTALL_SH" > "$_FN"
awk '/^_prepare_studio_uv_cache_for_launch\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_absolutize_uv_cache_dir\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_uv_is_bucket_name\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_uv_no_cache_requested\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_uv_cache_root_is_writable\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_uv_cache_is_writable\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"

if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_EARLY"; then
    echo "FAIL: could not extract the early UV_CACHE_DIR block from install.sh"
    exit 1
fi
if ! grep -q '_UV_CACHE_MODE=shared' "$_FN"; then
    echo "FAIL: could not extract _configure_uv_cache from install.sh"
    exit 1
fi
# Extracting the pieces is not the same as being able to RUN them: a helper this harness has
# not defined yet exits 127, `if !` reads that as an unwritable cache, and the block unsets the
# default every ordering case below depends on. The suite would pass while testing nothing.
# Built ONCE and sourced by the runner and the check alike, so the order is stated in one
# place. A check that repeats the order instead of sharing it stays green while the runner does
# something else, which is how this went unnoticed.
cat "$_FN" "$_EARLY" > "$_BOOT"
_selfcheck=$(
    STUDIO_HOME=$(mktemp -d "$_TMP/selfcheck.XXXXXX")
    export STUDIO_HOME
    unset UV_CACHE_DIR
    . "$_BOOT" 2>/dev/null
    printf '%s' "${UV_CACHE_DIR:-<unset>}"
)
# Only that the block kept a cache it could create, NOT that the installer-default flag is
# true: that flag is what this branch adds, and this suite must keep running against code that
# predates it.
case "$_selfcheck" in
    */cache/uv) ;;
    *)
        echo "FAIL: the early block did not keep its own default in this harness ($_selfcheck)."
        echo "      Every case below would run as though the cache were unwritable."
        exit 1
        ;;
esac
unset _selfcheck

_SH="${BASH:-/bin/bash}"

# $_BOOT is _FN then _EARLY, install.sh's own order: the helpers are defined around line 622
# and the early block RUNS at line 1001. Reversed, its call to _uv_cache_root_is_writable names
# a command that does not exist yet, and every unset-UV_CACHE_DIR case stops exercising the
# ordering bug this suite exists for.

# $1 = STUDIO_HOME, $2 = preset UV_CACHE_DIR ("" for unset), $3 = uv's default cache dir,
# $4 = "true" to isolate, $5 = UV_WORKING_DIR (also runs from $_TMP/cwd, so a relative $3
# resolved against the wrong base is visible), $6 = "true" to run under pipefail, $7 = "true"
# to run under set -f, $8 = "true" to run under set -e.
# Prints "<mode> <UV_CACHE_DIR> <after-launch-repoint>".
_run() {
    _stub_bin=$(mktemp -d)
    printf '#!/bin/sh\ncase "$1 $2" in "cache dir") printf "%%s\\n" "%s" ;; esac\n' \
        "$3" > "$_stub_bin/uv"
    chmod +x "$_stub_bin/uv"
    "$_SH" -c "
        [ '${8:-false}' = true ] && set -e
        [ '${6:-false}' = true ] && set -o pipefail
        [ '${7:-false}' = true ] && set -f
        STUDIO_HOME='$1'
        _ISOLATE_UV_CACHE='${4:-false}'
        if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
        if [ -n '${5:-}' ]; then
            UV_WORKING_DIR='${5:-}'; export UV_WORKING_DIR; cd '$_TMP/cwd'
        else
            unset UV_WORKING_DIR
        fi
        # Stubs for the surface _configure_uv_cache leans on, and nothing more: the point is to
        # run the real selection, not a paraphrase of it.
        step() { :; }
        _record_uv_cache_choice() { :; }
        C_WARN=''
        # A real file on PATH, not a shell function: the probe runs env, which execs a binary,
        # so a function is skipped and the host's own uv answers. That is how the first version
        # of this test passed against the wrong cache.
        PATH='$_stub_bin':\"\$PATH\"
        . '$_BOOT'
        _configure_uv_cache >/dev/null 2>&1
        _mode=\"\${_UV_CACHE_MODE:-<none>}\"
        _dir=\"\${UV_CACHE_DIR:-<unset>}\"
        _prepare_studio_uv_cache_for_launch
        printf '%s %s %s' \"\$_mode\" \"\$_dir\" \"\${UV_CACHE_DIR:-<unset>}\"
    "
    rm -rf "$_stub_bin"
}

# Warm means package BYTES, not buckets: the probe skips .msgpack/.http metadata.
_populated="$_TMP/uvdefault"
mkdir -p "$_populated/archive-v0/torch"
: > "$_populated/archive-v0/torch/libtorch.so"
# Every fixture below carries the CACHEDIR.TAG uv writes at the root of a cache it has
# initialized, so a rule that trips over a non-directory entry cannot pass here.
: > "$_populated/CACHEDIR.TAG"
# Metadata only, which must NOT read as warm; that is why the probe looks at file names.
_metadata_only="$_TMP/uvmeta"
mkdir -p "$_metadata_only/wheels-v1"
: > "$_metadata_only/wheels-v1/index.msgpack"
: > "$_metadata_only/CACHEDIR.TAG"
_empty="$_TMP/uvempty"
mkdir -p "$_empty"
# Only the ROOT is closed, so this still reads as warm; uv cannot rewrite CACHEDIR.TAG in it.
_readonly="$_TMP/uvro"
mkdir -p "$_readonly/archive-v0/torch"
: > "$_readonly/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly"
# The root is ours but a bucket is not, which is what a `sudo -E` run leaves behind.
_readonly_bucket="$_TMP/uvrobucket"
mkdir -p "$_readonly_bucket/archive-v0/torch" "$_readonly_bucket/wheels-v6"
: > "$_readonly_bucket/wheels-v6/index.msgpack"
: > "$_readonly_bucket/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_bucket/archive-v0"
# A bucket AFTER the one holding the artifact still has to be probed.
_readonly_late="$_TMP/uvrolate"
mkdir -p "$_readonly_late/archive-v0/torch" "$_readonly_late/sdists-v9"
: > "$_readonly_late/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_late/sdists-v9"
# Unenterable is unwritable: uv would still have to rename into the bucket it cannot open.
_denied_bucket="$_TMP/uvdenied"
mkdir -p "$_denied_bucket/archive-v0" "$_denied_bucket/builds-v0/pkg"
: > "$_denied_bucket/builds-v0/pkg/wheel.whl"
chmod 000 "$_denied_bucket/archive-v0"
# A cache-dir pointed at a mount point carries a root-owned lost+found. uv never writes it,
# so it must not condemn an otherwise usable warm cache.
_alien="$_TMP/uvalien"
mkdir -p "$_alien/archive-v0/torch" "$_alien/lost+found"
: > "$_alien/archive-v0/torch/libtorch.so"
: > "$_alien/CACHEDIR.TAG"
chmod 000 "$_alien/lost+found"
# A bucket NAME needs the whole suffix to be the version: a backup copy or a tarball beside
# the real bucket is not uv's to write, and must not condemn the cache.
_lookalike="$_TMP/uvlookalike"
mkdir -p "$_lookalike/archive-v0/torch"
: > "$_lookalike/archive-v0/torch/libtorch.so"
: > "$_lookalike/CACHEDIR.TAG"
: > "$_lookalike/archive-v0.tar.gz"
: > "$_lookalike/archive-v0.backup"
# A lookalike DIRECTORY full of files is not warmth either: `archive-*` matches
# `archive-v0.backup`, whose bytes uv cannot reuse, so counting it picks a cache that is empty
# in practice.
_lookalike_dir="$_TMP/uvlookalikedir"
mkdir -p "$_lookalike_dir/archive-v0.backup/pkg"
: > "$_lookalike_dir/archive-v0.backup/pkg/payload.so"
: > "$_lookalike_dir/CACHEDIR.TAG"
# Same for a qualifier BEFORE the version: `archive-backup-v0` passes the <kind>-v<N> shape,
# so warmth also has to check that the kind itself is one uv fills.
_lookalike_kind="$_TMP/uvlookalikekind"
mkdir -p "$_lookalike_kind/archive-backup-v0/pkg"
: > "$_lookalike_kind/archive-backup-v0/pkg/payload.so"
: > "$_lookalike_kind/CACHEDIR.TAG"
# `##*-v` strips through the LAST `-v`, so a name ending in one leaves an EMPTY suffix no
# `*[!0-9]*` matches: `archive-v1-v` read as a bucket, and one read-only directory named that
# way condemned a usable warm cache.
_empty_version="$_TMP/uvemptyver"
mkdir -p "$_empty_version/archive-v0/torch" "$_empty_version/archive-v1-v"
: > "$_empty_version/archive-v0/torch/libtorch.so"
: > "$_empty_version/CACHEDIR.TAG"
chmod a-w "$_empty_version/archive-v1-v"
# uv mutates interpreter-v4 too, so the verdict cannot stop at the five artifact families.
_denied_meta="$_TMP/uvmeta2"
mkdir -p "$_denied_meta/archive-v0/torch" "$_denied_meta/interpreter-v4"
: > "$_denied_meta/archive-v0/torch/libtorch.so"
chmod a-w "$_denied_meta/interpreter-v4"
# A dangling bucket link is an existing path to mkdir(2), which answers EEXIST, so uv reports
# `failed to create directory ...: File exists` the first time it needs that bucket.
_dangling="$_TMP/uvdangling"
mkdir -p "$_dangling/builds-v0/pkg"
: > "$_dangling/builds-v0/pkg/wheel.whl"
ln -s "$_TMP/no-such-target" "$_dangling/archive-v0"
# A same-named decoy beside the installer must not be what gets scanned.
mkdir -p "$_TMP/cwd/relcache/archive-v0/decoy"
: > "$_TMP/cwd/relcache/archive-v0/decoy/other.so"
mkdir -p "$_TMP/work/relcache/archive-v0/torch"
: > "$_TMP/work/relcache/archive-v0/torch/libtorch.so"
: > "$_TMP/work/relcache/CACHEDIR.TAG"
# Big enough that `head -n 1` closes the pipe before find is done, as every real cache is.
_big="$_TMP/uvbig"
mkdir -p "$_big/archive-v0/pkg"
: > "$_big/CACHEDIR.TAG"
_i=0
while [ "$_i" -lt 3000 ]; do : > "$_big/archive-v0/pkg/file-$_i.bin"; _i=$((_i + 1)); done

echo "=== the installer's own default does NOT count as a caller override ==="
# The regression: before the fix this was `custom` on every writable machine.
_out=$(_run "$_TMP/a" '' "$_populated")
assert_eq "populated default is reused"  "shared" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and it is uv's own cache"     "$_populated" "$(echo "$_out" | cut -d' ' -f2)"
# _prepare_studio_uv_cache_for_launch is live again, and only in shared mode.
assert_eq "launch repoints to Studio"    "$_TMP/a/cache/uv" "$(echo "$_out" | cut -d' ' -f3)"

_out=$(_run "$_TMP/b" '' "$_empty")
assert_eq "empty default -> studio"      "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "studio cache is used"         "$_TMP/b/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "no launch repoint in studio"  "$_TMP/b/cache/uv" "$(echo "$_out" | cut -d' ' -f3)"

_out=$(_run "$_TMP/e" '' "$_metadata_only")
assert_eq "metadata-only is not warm"    "studio" "$(echo "$_out" | cut -d' ' -f1)"

echo "=== a populated cache we cannot WRITE is not a cache we can use ==="
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-cache case (root writes through the mode bits)"
else
    _out=$(_run "$_TMP/f" '' "$_readonly")
    assert_eq "unwritable root -> studio"     "studio" "$(echo "$_out" | cut -d' ' -f1)"
    assert_eq "and the Studio cache is used"  "$_TMP/f/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
    _out=$(_run "$_TMP/i" '' "$_readonly_bucket")
    assert_eq "unwritable bucket -> studio"   "studio" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/j" '' "$_readonly_late")
    assert_eq "a later bucket is probed too"  "studio" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/m" '' "$_denied_bucket")
    assert_eq "a denied bucket -> studio"     "studio" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/n" '' "$_denied_meta")
    assert_eq "a denied metadata bucket too"  "studio" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/q" '' "$_alien")
    assert_eq "lost+found does not condemn it" "shared" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/v1" '' "$_empty_version")
    assert_eq "an empty -v suffix is not a bucket" "shared" "$(echo "$_out" | cut -d' ' -f1)"
fi
_out=$(_run "$_TMP/r" '' "$_lookalike")
assert_eq "a bucket lookalike is not a bucket" "shared" "$(echo "$_out" | cut -d' ' -f1)"
_out=$(_run "$_TMP/t" '' "$_lookalike_dir")
assert_eq "a lookalike dir is not warmth"      "studio" "$(echo "$_out" | cut -d' ' -f1)"
_out=$(_run "$_TMP/u" '' "$_lookalike_kind")
assert_eq "a lookalike KIND is not warmth"     "studio" "$(echo "$_out" | cut -d' ' -f1)"

echo "=== UV_NO_CACHE stands the selection down, in every spelling uv honours ==="
# uv takes this case-insensitively, so matching a fixed spelling would leave us probing and
# recording a cache uv is not using.
for _nc in 1 y Y true True TRUE tRuE t T yes Yes on On; do
    UV_NO_CACHE="$_nc"; export UV_NO_CACHE
    assert_eq "UV_NO_CACHE=$_nc -> studio" "studio" \
        "$(_run "$_TMP/nc$_nc" '' "$_populated" | cut -d' ' -f1)"
    unset UV_NO_CACHE
done
# ...and a false-ish value must not stand it down.
for _nc in 0 false; do
    UV_NO_CACHE="$_nc"; export UV_NO_CACHE
    assert_eq "UV_NO_CACHE=$_nc -> shared"  "shared" \
        "$(_run "$_TMP/nf$_nc" '' "$_populated" | cut -d' ' -f1)"
    unset UV_NO_CACHE
done
chmod 700 "$_alien/lost+found"
chmod 700 "$_denied_bucket/archive-v0"
chmod u+w "$_readonly" "$_readonly_bucket/archive-v0" "$_readonly_late/sdists-v9" \
    "$_denied_meta/interpreter-v4" "$_empty_version/archive-v1-v"
# The probe writes into a directory uv is about to fill, so it has to leave nothing behind.
_run "$_TMP/g" '' "$_populated" >/dev/null
assert_eq "write probe cleaned up" "" "$(ls -A "$_populated" | grep 'unsloth-write-probe' || true)"

echo "=== a relative cache-dir resolves against UV_WORKING_DIR, not the installer's cwd ==="
_out=$(_run "$_TMP/h" '' "relcache" false "$_TMP/work")
assert_eq "relative default is still warm" "shared" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "resolved against UV_WORKING_DIR" "$_TMP/work/relcache" "$(echo "$_out" | cut -d' ' -f2)"

echo "=== a bucket that is not a directory is not usable either ==="
_out=$(_run "$_TMP/p" '' "$_dangling")
assert_eq "dangling bucket link -> studio" "studio" "$(echo "$_out" | cut -d' ' -f1)"

echo "=== a big warm bucket stays warm under pipefail ==="
assert_eq "big bucket without pipefail" "shared" \
    "$(_run "$_TMP/k" '' "$_big" | cut -d' ' -f1)"
assert_eq "big bucket under pipefail"   "shared" \
    "$(_run "$_TMP/l" '' "$_big" false '' true | cut -d' ' -f1)"

echo "=== the scan still expands its globs under set -f ==="
assert_eq "populated default under set -f" "shared" \
    "$(_run "$_TMP/o" '' "$_populated" false '' false true | cut -d' ' -f1)"

echo "=== a CALLER's UV_CACHE_DIR still outranks the selection, untouched ==="
_out=$(_run "$_TMP/c" "$_TMP/mine" "$_populated")
assert_eq "caller value is custom"       "custom" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "caller value is preserved"    "$_TMP/mine" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "no launch repoint for custom" "$_TMP/mine" "$(echo "$_out" | cut -d' ' -f3)"

echo "=== --isolated-uv-cache still wins over a populated default ==="
_out=$(_run "$_TMP/d" '' "$_populated" true)
assert_eq "isolation is honoured"        "isolated" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and it uses the Studio cache" "$_TMP/d/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"

echo "=== an unwritable STUDIO_HOME still reaches the selection ==="
# The early block unsets UV_CACHE_DIR there, so the choice must not depend on the flag.
: > "$_TMP/blocked"
_out=$(_run "$_TMP/blocked" '' "$_populated")
assert_eq "unwritable home -> shared"    "shared" "$(echo "$_out" | cut -d' ' -f1)"

echo "=== an install that predates the marker keeps the cache it already has ==="
# The installed base from before the marker has a populated $STUDIO_HOME/cache/uv and nothing
# recording it, and abandoning it costs gigabytes of Torch and CUDA. It is the LAST candidate,
# though, behind uv's default: the marker arrived in b66d2a4c8 and the early block only in
# e12963071 the day after, so an install with no marker is old enough that `shared` was
# reachable, and there the launch repoint leaves backend wheels in the Studio cache while the
# real bytes sit in the default. So it wins when the default is cold, and loses when it is warm.
_cold_home="$_TMP/colddefault2"
mkdir -p "$_cold_home"
mkdir -p "$_TMP/pre/cache/uv/archive-v0/torch"
: > "$_TMP/pre/cache/uv/archive-v0/torch/libtorch.so"
: > "$_TMP/pre/cache/uv/CACHEDIR.TAG"
_out=$(_run "$_TMP/pre" '' "$_cold_home")
assert_eq "unmarked warm Studio beats a cold default" "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and it is the one that is used"            "$_TMP/pre/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
# ...and loses to a WARM one, because content cannot say whose the Studio cache is.
assert_eq "a warm default outranks it"                "shared" \
    "$(_run "$_TMP/pre3" '' "$_populated" | cut -d' ' -f1)"
# An unmarked but EMPTY Studio cache is not a reason to skip the shared one.
assert_eq "an unmarked cold Studio cache does not" "shared" \
    "$(_run "$_TMP/pre2" '' "$_populated" | cut -d' ' -f1)"

echo "=== a marker written by the Windows installer is readable here ==="
# install.ps1 writes it with Set-Content -Encoding utf8: a BOM and CRLF under PowerShell 5.1.
# A WSL install shares $STUDIO_HOME with the Windows one, so this file crosses over.
mkdir -p "$_TMP/crlf/cache/uv/archive-v0/torch"
: > "$_TMP/crlf/cache/uv/archive-v0/torch/libtorch.so"
printf '\357\273\277%s\r\n' "$_TMP/crlf/cache/uv" > "$_TMP/crlf/cache/uv-cache-dir"
_out=$(_run "$_TMP/crlf" '' "$_populated")
assert_eq "a BOM+CRLF marker is honoured"  "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and names the right directory"  "$_TMP/crlf/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"

echo "=== a trailing slash in the marker is the same directory ==="
mkdir -p "$_TMP/slash/cache/uv/archive-v0/torch"
: > "$_TMP/slash/cache/uv/archive-v0/torch/libtorch.so"
printf '%s/\n' "$_TMP/slash/cache/uv" > "$_TMP/slash/cache/uv-cache-dir"
# studio, not shared: otherwise the launch repoint the studio branch exists to avoid fires.
assert_eq "a trailing slash still reads as ours" "studio" \
    "$(_run "$_TMP/slash" '' "$_populated" | cut -d' ' -f1)"

echo "=== the launch repoint refuses a Studio cache the backend could not fill ==="
# shared with an unwritable STUDIO_HOME is reachable only because the early block failed its
# own write probe, so repointing would hand the autostarted backend a cache uv aborts on.
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-root launch case (root writes through the mode bits)"
else
    mkdir -p "$_TMP/lockedhome"
    chmod a-w "$_TMP/lockedhome"
    _out=$(_run "$_TMP/lockedhome" '' "$_populated")
    assert_eq "unwritable root still selects shared" "shared" "$(echo "$_out" | cut -d' ' -f1)"
    assert_eq "and the launch keeps that cache"      "$_populated" "$(echo "$_out" | cut -d' ' -f3)"
    chmod u+w "$_TMP/lockedhome"
fi

echo "=== a fallback we cannot write is not a fallback ==="
# The early block has already given up on an unwritable $STUDIO_HOME/cache/uv, which is the
# only reason the selection is running, so landing on that certain failure turns a working
# install into `failed to create cache directory`.
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-fallback case (root writes through the mode bits)"
else
    # A fresh fixture: the shared $_readonly above has had its mode restored by now.
    _refused="$_TMP/uvrefused"
    mkdir -p "$_refused/archive-v0/torch"
    : > "$_refused/archive-v0/torch/libtorch.so"
    : > "$_refused/CACHEDIR.TAG"
    chmod a-w "$_refused"
    mkdir -p "$_TMP/deadhome"
    chmod a-w "$_TMP/deadhome"
    _out=$(_run "$_TMP/deadhome" '' "$_refused")
    assert_eq "a warm cache beats a dead fallback"  "shared" "$(echo "$_out" | cut -d' ' -f1)"
    assert_eq "and it is the one that is used"      "$_refused" "$(echo "$_out" | cut -d' ' -f2)"
    assert_eq "the launch keeps it too"             "$_refused" "$(echo "$_out" | cut -d' ' -f3)"
    chmod u+w "$_TMP/deadhome"
    # With a WRITABLE Studio cache the refused candidate still loses to it, unchanged.
    _out=$(_run "$_TMP/livehome" '' "$_refused")
    assert_eq "a writable fallback still wins"      "studio" "$(echo "$_out" | cut -d' ' -f1)"
    assert_eq "and it is the Studio cache"          "$_TMP/livehome/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
    chmod u+w "$_refused"
fi

echo "=== the same answers under a real /bin/sh, and under the errexit install.sh runs with ==="
# Everything above runs one shell without errexit, which is the one arrangement install.sh
# never uses: it is `#!/bin/sh` with `set -e` on line 5, /bin/sh is dash on Debian and busybox
# ash on Alpine, and a construct whose status leaks aborts the install rather than answering.
# The core verdicts are re-run here across every /bin/sh this box has, errexit both ways.
_SH_SAVED="$_SH"
for _alt in /bin/sh /bin/bash; do
    [ -x "$_alt" ] || continue
    _SH="$_alt"
    _alt_name=$(basename "$_alt")
    for _ee in false true; do
        _tag="$_alt_name errexit=$_ee"
        _out=$(_run "$_TMP/ee.$_alt_name.$_ee.a" '' "$_populated" false '' false false "$_ee")
        assert_eq "$_tag: populated default -> shared" "shared" "$(echo "$_out" | cut -d' ' -f1)"
        assert_eq "$_tag: and the launch repoints"     "$_TMP/ee.$_alt_name.$_ee.a/cache/uv" \
            "$(echo "$_out" | cut -d' ' -f3)"
        assert_eq "$_tag: empty default -> studio"     "studio" \
            "$(_run "$_TMP/ee.$_alt_name.$_ee.b" '' "$_empty" false '' false false "$_ee" | cut -d' ' -f1)"
        assert_eq "$_tag: caller value -> custom"      "custom" \
            "$(_run "$_TMP/ee.$_alt_name.$_ee.c" "$_TMP/mine" "$_populated" false '' false false "$_ee" | cut -d' ' -f1)"
        assert_eq "$_tag: isolation is honoured"       "isolated" \
            "$(_run "$_TMP/ee.$_alt_name.$_ee.d" '' "$_populated" true '' false false "$_ee" | cut -d' ' -f1)"
        assert_eq "$_tag: a lookalike is not warmth"   "studio" \
            "$(_run "$_TMP/ee.$_alt_name.$_ee.e" '' "$_lookalike_dir" false '' false false "$_ee" | cut -d' ' -f1)"
        UV_NO_CACHE=1; export UV_NO_CACHE
        assert_eq "$_tag: UV_NO_CACHE -> studio"       "studio" \
            "$(_run "$_TMP/ee.$_alt_name.$_ee.f" '' "$_populated" false '' false false "$_ee" | cut -d' ' -f1)"
        unset UV_NO_CACHE
    done
done
_SH="$_SH_SAVED"

echo "=== a Studio cache with a bucket uv cannot use is not a fallback either ==="
# A `sudo` run leaves a root-owned bucket and a dangling link is an existing path to mkdir(2),
# so the ROOT can be writable while uv still aborts on the cache. Selecting it anyway, or
# repointing the launch at it, turns an install that reported success into a uv error.
_brokenstudio="$_TMP/brokenstudio"
mkdir -p "$_brokenstudio/cache/uv"
ln -s "$_TMP/no-such-target" "$_brokenstudio/cache/uv/archive-v0"
_cold_default="$_TMP/colddefault"
mkdir -p "$_cold_default"
_out=$(_run "$_brokenstudio" '' "$_cold_default")
assert_eq "broken Studio bucket -> not studio" "shared"        "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and uv's default is used instead"   "$_cold_default" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "and the launch does not repoint"    "$_cold_default" "$(echo "$_out" | cut -d' ' -f3)"
# With a WARM default the selection already preferred it; the repoint is the part that used to
# hand the backend the rejected cache anyway.
_out=$(_run "$_brokenstudio" '' "$_populated")
assert_eq "warm default beside a broken Studio" "shared"     "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and the launch keeps it"             "$_populated" "$(echo "$_out" | cut -d' ' -f3)"
# A Studio cache whose buckets are fine is still the ordinary answer.
_out=$(_run "$_TMP/okstudio" '' "$_cold_default")
assert_eq "an intact Studio cache still wins"   "studio"     "$(echo "$_out" | cut -d' ' -f1)"

echo "=== a marker naming a directory that is gone is stale, not a decision ==="
# The marker named a cache the user deleted. Treating that as a decision handed the install
# uv's default and abandoned a warm Studio cache holding Torch and CUDA, which offline is an
# install that used to succeed and then failed.
_stale=$_TMP/stalemarker
mkdir -p "$_stale/studio/cache/uv/archive-v0/torch"
: > "$_stale/studio/cache/uv/archive-v0/torch/libtorch.so"
: > "$_stale/studio/cache/uv/CACHEDIR.TAG"
printf '%s\n' "$_TMP/deleted-cache" > "$_stale/studio/cache/uv-cache-dir"
_out=$(_run "$_stale/studio" '' "$_cold_home")
assert_eq "stale marker -> keep the Studio cache" "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and not uv's default"                  "$_stale/studio/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
# A stale marker is no more evidence than no marker, so a warm default still outranks it.
assert_eq "a warm default still outranks a stale marker" "shared" \
    "$(_run "$_stale/studio" '' "$_populated" | cut -d' ' -f1)"

echo "=== a root we can create in but not unlink from is not writable ==="
# NTFS carries DELETE as its own ACE and an append-only directory does the same on ext4, so
# "create succeeded" does not mean "uv can rename into this". The bucket probe always failed
# the candidate here; the root probe swallowed it. A failing rm, not an ACL: the mode bits
# cannot express this and running as root would hide it.
_probe_unlink_case() {
    # shellcheck disable=SC2317
    rm() { return 1; }
    if _uv_cache_root_is_writable "$1"; then echo writable; else echo unwritable; fi
}
# A delete that FAILS but leaves nothing behind is a delete: an indexer holding the handle
# makes one rm fail and the next succeed, and a probe another process removed is cleaned up.
# `rm -f` exits 0 on a missing file where Remove-Item -ErrorAction Stop throws, which is the
# divergence this rule closes.
_probe_gone_case() {
    # shellcheck disable=SC2317
    rm() { command rm -f "$@" 2>/dev/null; return 1; }
    if _uv_cache_root_is_writable "$1"; then echo writable; else echo unwritable; fi
}
_out=$(
    . "$_FN" 2>/dev/null || true
    _probe_unlink_case "$_TMP/unlinkdenied"
)
assert_eq "create without unlink -> unwritable" "unwritable" "$_out"
# And the ordinary case still passes, so the check above is not just failing everything.
_out=$(
    . "$_FN" 2>/dev/null || true
    if _uv_cache_root_is_writable "$_TMP/unlinkok"; then echo writable; else echo unwritable; fi
)
assert_eq "an ordinary root is still writable"  "writable"   "$_out"
assert_eq "and the probe is not left behind"    ""           "$(ls -A "$_TMP/unlinkok" 2>/dev/null)"
_out=$(
    . "$_FN" 2>/dev/null || true
    _probe_gone_case "$_TMP/unlinkgone"
)
assert_eq "a failed rm that removed it is fine" "writable"   "$_out"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
