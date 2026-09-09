#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards that _configure_uv_cache's adaptive selection is actually REACHABLE.
#
# The defect this pins, found by review and confirmed by execution order rather than by any
# diff:
#
#   * The block near the top of install.sh exports UV_CACHE_DIR="$STUDIO_HOME/cache/uv"
#     whenever the caller left it unset. It has to run there and not later, because uv aborts
#     on a cache it cannot create and several install steps run before _configure_uv_cache.
#   * _configure_uv_cache is called much further down. Its first `case` matched any non-blank
#     ${UV_CACHE_DIR-}, set _UV_CACHE_MODE=custom and returned.
#   * So on EVERY writable install the two together took the `custom` branch. The `uv cache
#     dir` probe never ran, `shared` was never selected, and users re-downloaded multi-gigabyte
#     Torch and CUDA wheels into a second cache while a populated one sat beside it.
#   * _prepare_studio_uv_cache_for_launch begins `[ "${_UV_CACHE_MODE:-}" = shared ] || return
#     0`, so it was dead code for the same reason.
#
# Nothing in the file is wrong to READ. The two halves are each correct and were written apart;
# the defect only exists in the order they run, which is why it needs a test that runs them in
# that order rather than one that inspects either half.
#
# The contract:
#   * caller-set UV_CACHE_DIR            -> custom, left exactly as the caller wrote it
#   * unset, uv's default is populated   -> shared, and the launch repoint becomes live
#   * unset, uv's default is empty       -> studio
#   * populated but not writable, root or bucket -> studio; uv aborts on either
#   * a relative cache-dir               -> resolved against UV_WORKING_DIR before scanning
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

# Both halves lifted from install.sh, so the real code is what runs here.
# Anchored on text that predates the fix, so this suite runs the OLD code too and fails
# against it. Anchoring on a line the fix introduces would make it fail to extract instead,
# which proves nothing about the behaviour.
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" > "$_EARLY"
awk '/^_configure_uv_cache\(\) \{$/,/^\}$/' "$INSTALL_SH" > "$_FN"
awk '/^_prepare_studio_uv_cache_for_launch\(\) \{$/,/^\}$/' "$INSTALL_SH" >> "$_FN"
# Predates the fix too, so the relative-cache-dir case below runs against the old code and
# fails on its assertion rather than on a missing function.
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
# resolved against the wrong base is visible). Prints "<mode> <UV_CACHE_DIR> <after-launch-repoint>".
_run() {
    _stub_bin=$(mktemp -d)
    printf '#!/bin/sh\ncase "$1 $2" in "cache dir") printf "%%s\\n" "%s" ;; esac\n' \
        "$3" > "$_stub_bin/uv"
    chmod +x "$_stub_bin/uv"
    "$_SH" -c "
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

# Warm means package BYTES, not buckets: the probe skips .msgpack/.http metadata, so a real
# artifact file is what makes this cache read as populated.
_populated="$_TMP/uvdefault"
mkdir -p "$_populated/archive-v0/torch"
: > "$_populated/archive-v0/torch/libtorch.so"
# Metadata only, which must NOT read as warm -- that regression is the reason the probe looks
# at file names at all.
_metadata_only="$_TMP/uvmeta"
mkdir -p "$_metadata_only/wheels-v1"
: > "$_metadata_only/wheels-v1/index.msgpack"
_empty="$_TMP/uvempty"
mkdir -p "$_empty"
# Populated and readable, but uv cannot rewrite CACHEDIR.TAG in it. Only the root is closed,
# so the scan still walks the buckets and reads it as warm -- which is the whole point.
_readonly="$_TMP/uvro"
mkdir -p "$_readonly/archive-v0/torch"
: > "$_readonly/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly"
# The other half: the root is ours but a bucket is not, which is what a `sudo -E` run leaves
# behind. uv renames each extracted distribution INTO archive-*, so this fails just as hard.
_readonly_bucket="$_TMP/uvrobucket"
mkdir -p "$_readonly_bucket/archive-v0/torch" "$_readonly_bucket/wheels-v6"
: > "$_readonly_bucket/wheels-v6/index.msgpack"
: > "$_readonly_bucket/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_bucket/archive-v0"
# And the case that must NOT read as blocked: the artifact is found in the first bucket, and a
# LATER writable bucket must not be skipped by an early exit that never probed it.
_readonly_late="$_TMP/uvrolate"
mkdir -p "$_readonly_late/archive-v0/torch" "$_readonly_late/sdists-v9"
: > "$_readonly_late/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_late/sdists-v9"
# uv prints a relative cache-dir from uv.toml verbatim and resolves it against UV_WORKING_DIR,
# so a same-named decoy beside the installer must not be what gets scanned.
mkdir -p "$_TMP/cwd/relcache/archive-v0/decoy"
: > "$_TMP/cwd/relcache/archive-v0/decoy/other.so"
mkdir -p "$_TMP/work/relcache/archive-v0/torch"
: > "$_TMP/work/relcache/archive-v0/torch/libtorch.so"

echo "=== the installer's own default does NOT count as a caller override ==="
# This is the regression. Before the fix the mode here was `custom` and the two lines below
# could not be reached on any writable machine.
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
# Readable was the only thing the scan tested, so this cache got exported and uv then died on
# `Failed to initialize cache ... Permission denied` -- an install that used to work.
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
fi
chmod u+w "$_readonly" "$_readonly_bucket/archive-v0" "$_readonly_late/sdists-v9"
# The probe writes into a directory uv is about to fill, so it has to leave nothing behind.
_run "$_TMP/g" '' "$_populated" >/dev/null
assert_eq "write probe cleaned up" "" "$(ls -A "$_populated" | grep 'unsloth-write-probe' || true)"

echo "=== a relative cache-dir resolves against UV_WORKING_DIR, not the installer's cwd ==="
_out=$(_run "$_TMP/h" '' "relcache" false "$_TMP/work")
assert_eq "relative default is still warm" "shared" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "resolved against UV_WORKING_DIR" "$_TMP/work/relcache" "$(echo "$_out" | cut -d' ' -f2)"

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
# The early block unsets UV_CACHE_DIR there, so nothing was ever going to say `custom`; this
# pins that the fix did not make that path depend on the flag being cleared.
: > "$_TMP/blocked"
_out=$(_run "$_TMP/blocked" '' "$_populated")
assert_eq "unwritable home -> shared"    "shared" "$(echo "$_out" | cut -d' ' -f1)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
