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
#   * --isolated-uv-cache                -> isolated, whatever else is true
#   * unwritable STUDIO_HOME             -> the early block unsets, and the choice still runs
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_TMP=$(mktemp -d)
_EARLY=$(mktemp)
_FN=$(mktemp)
trap 'rm -rf "$_TMP" "$_EARLY" "$_FN"' EXIT

# The real code, anchored on text that PREDATES the fix so this suite also runs against the
# old code and fails on an assertion, not on an empty extraction.
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" > "$_EARLY"
awk '/^_configure_uv_cache\(\) \{$/,/^\}$/' "$INSTALL_SH" > "$_FN"
awk '/^_prepare_studio_uv_cache_for_launch\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
awk '/^_absolutize_uv_cache_dir\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"

if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_EARLY"; then
    echo "FAIL: could not extract the early UV_CACHE_DIR block from install.sh"
    exit 1
fi
if ! grep -q '_UV_CACHE_MODE=shared' "$_FN"; then
    echo "FAIL: could not extract _configure_uv_cache from install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"

# $1 = STUDIO_HOME, $2 = preset UV_CACHE_DIR ("" for unset), $3 = uv's default cache dir,
# $4 = "true" to isolate, $5 = UV_WORKING_DIR (also runs from $_TMP/cwd, so a relative $3
# resolved against the wrong base is visible), $6 = "true" to run under pipefail, $7 = "true"
# to run under set -f. Prints "<mode> <UV_CACHE_DIR> <after-launch-repoint>".
_run() {
    _stub_bin=$(mktemp -d)
    printf '#!/bin/sh\ncase "$1 $2" in "cache dir") printf "%%s\\n" "%s" ;; esac\n' \
        "$3" > "$_stub_bin/uv"
    chmod +x "$_stub_bin/uv"
    "$_SH" -c "
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
        # A real file on PATH, not a shell function: the probe runs
        # \`env -u UV_CACHE_DIR uv cache dir\`, and env execs a binary, so a function would be
        # skipped and the host's own uv would answer -- which is exactly how the first version
        # of this test passed against the wrong cache.
        PATH='$_stub_bin':\"\$PATH\"
        . '$_EARLY'
        . '$_FN'
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
fi
chmod 700 "$_alien/lost+found"
chmod 700 "$_denied_bucket/archive-v0"
chmod u+w "$_readonly" "$_readonly_bucket/archive-v0" "$_readonly_late/sdists-v9" \
    "$_denied_meta/interpreter-v4"
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

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
