#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Guards that _configure_uv_cache's adaptive selection is reachable: the defect lived in the
# ORDER of the early UV_CACHE_DIR block and _configure_uv_cache, so both run in that order.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_TMP=$(mktemp -d)
_EARLY=$(mktemp)
_BOOT=$(mktemp)
_FN=$(mktemp)
trap 'rm -rf "$_TMP" "$_EARLY" "$_FN" "$_BOOT"' EXIT

# Anchored on text that predates the fix, so old code fails on an assertion, not extraction.
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
# A helper not yet defined exits 127 and `if !` reads that as unwritable, so the suite would
# test nothing. Built once and shared by runner and check so the order lives in one place.
cat "$_FN" "$_EARLY" > "$_BOOT"
_selfcheck=$(
    STUDIO_HOME=$(mktemp -d "$_TMP/selfcheck.XXXXXX")
    export STUDIO_HOME
    unset UV_CACHE_DIR
    . "$_BOOT" 2>/dev/null
    printf '%s' "${UV_CACHE_DIR:-<unset>}"
)
# Do not assert the installer-default flag: this suite must also run against older code.
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

# _FN then _EARLY is install.sh's own order; reversed, the early block calls an undefined helper.

# $1 STUDIO_HOME, $2 preset UV_CACHE_DIR ("" unset), $3 uv default, $4 isolate, $5 UV_WORKING_DIR,
# $6 pipefail, $7 set -f, $8 set -e. Prints "<mode> <UV_CACHE_DIR> <after-launch-repoint>".
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

# Warm means package bytes, not buckets: the probe skips .msgpack/.http metadata.
_populated="$_TMP/uvdefault"
mkdir -p "$_populated/archive-v0/torch"
: > "$_populated/archive-v0/torch/libtorch.so"
: > "$_populated/CACHEDIR.TAG"
_metadata_only="$_TMP/uvmeta"
mkdir -p "$_metadata_only/wheels-v1"
: > "$_metadata_only/wheels-v1/index.msgpack"
: > "$_metadata_only/CACHEDIR.TAG"
_empty="$_TMP/uvempty"
mkdir -p "$_empty"
_readonly="$_TMP/uvro"
mkdir -p "$_readonly/archive-v0/torch"
: > "$_readonly/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly"
# The root is ours but a bucket is not, as a `sudo -E` run leaves behind.
_readonly_bucket="$_TMP/uvrobucket"
mkdir -p "$_readonly_bucket/archive-v0/torch" "$_readonly_bucket/wheels-v6"
: > "$_readonly_bucket/wheels-v6/index.msgpack"
: > "$_readonly_bucket/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_bucket/archive-v0"
_readonly_late="$_TMP/uvrolate"
mkdir -p "$_readonly_late/archive-v0/torch" "$_readonly_late/sdists-v9"
: > "$_readonly_late/archive-v0/torch/libtorch.so"
chmod a-w "$_readonly_late/sdists-v9"
_denied_bucket="$_TMP/uvdenied"
mkdir -p "$_denied_bucket/archive-v0" "$_denied_bucket/builds-v0/pkg"
: > "$_denied_bucket/builds-v0/pkg/wheel.whl"
chmod 000 "$_denied_bucket/archive-v0"
# A mount point's root-owned lost+found must not condemn a usable warm cache.
_alien="$_TMP/uvalien"
mkdir -p "$_alien/archive-v0/torch" "$_alien/lost+found"
: > "$_alien/archive-v0/torch/libtorch.so"
: > "$_alien/CACHEDIR.TAG"
chmod 000 "$_alien/lost+found"
_alien_bucket="$_TMP/uvalienbucket"
mkdir -p "$_alien_bucket/archive-v0/torch" "$_alien_bucket/unused-v999"
: > "$_alien_bucket/archive-v0/torch/libtorch.so"
: > "$_alien_bucket/CACHEDIR.TAG"
chmod 000 "$_alien_bucket/unused-v999"
# python-v0 is uv's since 0.8.16.
_managed_python="$_TMP/uvpython"
mkdir -p "$_managed_python/archive-v0/torch" "$_managed_python/python-v0"
: > "$_managed_python/archive-v0/torch/libtorch.so"
: > "$_managed_python/CACHEDIR.TAG"
chmod a-w "$_managed_python/python-v0"
_cased_bucket="$_TMP/uvcased"
mkdir -p "$_cased_bucket/archive-v0/torch" "$_cased_bucket/Python-V0" "$_cased_bucket/python-v0"
: > "$_cased_bucket/archive-v0/torch/libtorch.so"
: > "$_cased_bucket/CACHEDIR.TAG"
chmod a-w "$_cased_bucket/Python-V0"
_cased_only="$_TMP/uvcasedonly"
mkdir -p "$_cased_only/archive-v0/torch" "$_cased_only/Python-V0"
: > "$_cased_only/archive-v0/torch/libtorch.so"
: > "$_cased_only/CACHEDIR.TAG"
chmod a-w "$_cased_only/Python-V0"
_cased_file="$_TMP/uvcasedfile"
mkdir -p "$_cased_file/archive-v0/torch"
: > "$_cased_file/archive-v0/torch/libtorch.so"
: > "$_cased_file/CACHEDIR.TAG"
: > "$_cased_file/Python-V0"
_FOLDS=false
mkdir -p "$_TMP/.foldcheck-A" && [ -d "$_TMP/.foldcheck-a" ] && _FOLDS=true
_lookalike="$_TMP/uvlookalike"
mkdir -p "$_lookalike/archive-v0/torch"
: > "$_lookalike/archive-v0/torch/libtorch.so"
: > "$_lookalike/CACHEDIR.TAG"
: > "$_lookalike/archive-v0.tar.gz"
: > "$_lookalike/archive-v0.backup"
_lookalike_dir="$_TMP/uvlookalikedir"
mkdir -p "$_lookalike_dir/archive-v0.backup/pkg"
: > "$_lookalike_dir/archive-v0.backup/pkg/payload.so"
: > "$_lookalike_dir/CACHEDIR.TAG"
_lookalike_kind="$_TMP/uvlookalikekind"
mkdir -p "$_lookalike_kind/archive-backup-v0/pkg"
: > "$_lookalike_kind/archive-backup-v0/pkg/payload.so"
: > "$_lookalike_kind/CACHEDIR.TAG"
# `##*-v` strips through the last `-v`, so a name ending in one leaves an empty version.
_empty_version="$_TMP/uvemptyver"
mkdir -p "$_empty_version/archive-v0/torch" "$_empty_version/archive-v1-v"
: > "$_empty_version/archive-v0/torch/libtorch.so"
: > "$_empty_version/CACHEDIR.TAG"
chmod a-w "$_empty_version/archive-v1-v"
_denied_meta="$_TMP/uvmeta2"
mkdir -p "$_denied_meta/archive-v0/torch" "$_denied_meta/interpreter-v4"
: > "$_denied_meta/archive-v0/torch/libtorch.so"
chmod a-w "$_denied_meta/interpreter-v4"
# A dangling bucket link makes mkdir(2) answer EEXIST, so uv fails on it.
_dangling="$_TMP/uvdangling"
mkdir -p "$_dangling/builds-v0/pkg"
: > "$_dangling/builds-v0/pkg/wheel.whl"
ln -s "$_TMP/no-such-target" "$_dangling/archive-v0"
mkdir -p "$_TMP/cwd/relcache/archive-v0/decoy"
: > "$_TMP/cwd/relcache/archive-v0/decoy/other.so"
mkdir -p "$_TMP/work/relcache/archive-v0/torch"
: > "$_TMP/work/relcache/archive-v0/torch/libtorch.so"
: > "$_TMP/work/relcache/CACHEDIR.TAG"
# Big enough that `head -n 1` closes the pipe before find is done.
_big="$_TMP/uvbig"
mkdir -p "$_big/archive-v0/pkg"
: > "$_big/CACHEDIR.TAG"
_i=0
while [ "$_i" -lt 3000 ]; do : > "$_big/archive-v0/pkg/file-$_i.bin"; _i=$((_i + 1)); done

echo "=== the installer's own default does NOT count as a caller override ==="
_out=$(_run "$_TMP/a" '' "$_populated")
assert_eq "populated default is reused"  "shared" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and it is uv's own cache"     "$_populated" "$(echo "$_out" | cut -d' ' -f2)"
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
    _out=$(_run "$_TMP/q2" '' "$_alien_bucket")
    assert_eq "an unknown KIND does not either" "shared" "$(echo "$_out" | cut -d' ' -f1)"
    _out=$(_run "$_TMP/q3" '' "$_managed_python")
    assert_eq "a read-only python-v0 still does" "studio" "$(echo "$_out" | cut -d' ' -f1)"
    if [ "$_FOLDS" = true ]; then
        _out=$(_run "$_TMP/q4" '' "$_cased_bucket")
        assert_eq "a cased bucket condemns when folded" "studio" "$(echo "$_out" | cut -d' ' -f1)"
        _out=$(_run "$_TMP/q5" '' "$_cased_only")
        assert_eq "so does a lone cased bucket"         "studio" "$(echo "$_out" | cut -d' ' -f1)"
        _out=$(_run "$_TMP/q6" '' "$_cased_file")
        assert_eq "and a cased FILE does too"           "studio" "$(echo "$_out" | cut -d' ' -f1)"
    else
        _out=$(_run "$_TMP/q4" '' "$_cased_bucket")
        assert_eq "a cased sibling does not condemn" "shared" "$(echo "$_out" | cut -d' ' -f1)"
        _out=$(_run "$_TMP/q5" '' "$_cased_only")
        assert_eq "nor a lone cased directory"       "shared" "$(echo "$_out" | cut -d' ' -f1)"
        _out=$(_run "$_TMP/q6" '' "$_cased_file")
        assert_eq "nor a cased file"                 "shared" "$(echo "$_out" | cut -d' ' -f1)"
    fi
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
# uv parses UV_NO_CACHE case-insensitively.
for _nc in 1 y Y true True TRUE tRuE t T yes Yes on On; do
    UV_NO_CACHE="$_nc"; export UV_NO_CACHE
    assert_eq "UV_NO_CACHE=$_nc -> studio" "studio" \
        "$(_run "$_TMP/nc$_nc" '' "$_populated" | cut -d' ' -f1)"
    unset UV_NO_CACHE
done
for _nc in 0 false; do
    UV_NO_CACHE="$_nc"; export UV_NO_CACHE
    assert_eq "UV_NO_CACHE=$_nc -> shared"  "shared" \
        "$(_run "$_TMP/nf$_nc" '' "$_populated" | cut -d' ' -f1)"
    unset UV_NO_CACHE
done
chmod 700 "$_alien/lost+found"
chmod 700 "$_alien_bucket/unused-v999"
chmod u+w "$_managed_python/python-v0" "$_cased_bucket/Python-V0" "$_cased_only/Python-V0"
chmod 700 "$_denied_bucket/archive-v0"
chmod u+w "$_readonly" "$_readonly_bucket/archive-v0" "$_readonly_late/sdists-v9" \
    "$_denied_meta/interpreter-v4" "$_empty_version/archive-v1-v"
_run "$_TMP/g" '' "$_populated" >/dev/null
assert_eq "write probe cleaned up" "" "$(ls -A "$_populated" | grep 'unsloth-write-probe' || true)"
assert_eq "case probe cleaned up"  "" "$(ls -A "$_populated" | grep 'unsloth-case-probe' || true)"

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
: > "$_TMP/blocked"
_out=$(_run "$_TMP/blocked" '' "$_populated")
assert_eq "unwritable home -> shared"    "shared" "$(echo "$_out" | cut -d' ' -f1)"

echo "=== an install that predates the marker keeps the cache it already has ==="
# Pre-marker installs have an unmarked warm Studio cache; it wins over a cold default but
# loses to a warm one, since `shared` was reachable on those installs.
_cold_home="$_TMP/colddefault2"
mkdir -p "$_cold_home"
mkdir -p "$_TMP/pre/cache/uv/archive-v0/torch"
: > "$_TMP/pre/cache/uv/archive-v0/torch/libtorch.so"
: > "$_TMP/pre/cache/uv/CACHEDIR.TAG"
_out=$(_run "$_TMP/pre" '' "$_cold_home")
assert_eq "unmarked warm Studio beats a cold default" "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and it is the one that is used"            "$_TMP/pre/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "a warm default outranks it"                "shared" \
    "$(_run "$_TMP/pre3" '' "$_populated" | cut -d' ' -f1)"
assert_eq "an unmarked cold Studio cache does not" "shared" \
    "$(_run "$_TMP/pre2" '' "$_populated" | cut -d' ' -f1)"

echo "=== a marker written by the Windows installer is readable here ==="
# install.ps1 writes the marker with a BOM and CRLF, and WSL shares $STUDIO_HOME with Windows.
mkdir -p "$_TMP/crlf/cache/uv/archive-v0/torch"
: > "$_TMP/crlf/cache/uv/archive-v0/torch/libtorch.so"
printf '\357\273\277%s\r\n' "$_TMP/crlf/cache/uv" > "$_TMP/crlf/cache/uv-cache-dir"
_out=$(_run "$_TMP/crlf" '' "$_populated")
assert_eq "a BOM+CRLF marker is honoured"  "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and names the right directory"  "$_TMP/crlf/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"

_cr=$(printf '\r')
mkdir -p "$_TMP/embedcr/cache" "$_TMP/sha${_cr}red/archive-v0/torch"
: > "$_TMP/sha${_cr}red/archive-v0/torch/libtorch.so"
printf '%s\r\n' "$_TMP/sha${_cr}red" > "$_TMP/embedcr/cache/uv-cache-dir"
_out=$(_run "$_TMP/embedcr" '' "$_empty")
assert_eq "a CR inside the path survives" "$_TMP/sha${_cr}red" "$(echo "$_out" | cut -d' ' -f2)"
unset _cr

echo "=== a trailing slash in the marker is the same directory ==="
mkdir -p "$_TMP/slash/cache/uv/archive-v0/torch"
: > "$_TMP/slash/cache/uv/archive-v0/torch/libtorch.so"
printf '%s/\n' "$_TMP/slash/cache/uv" > "$_TMP/slash/cache/uv-cache-dir"
assert_eq "a trailing slash still reads as ours" "studio" \
    "$(_run "$_TMP/slash" '' "$_populated" | cut -d' ' -f1)"

echo "=== the launch repoint refuses a Studio cache the backend could not fill ==="
# shared with an unwritable STUDIO_HOME: repointing would hand the backend a cache uv aborts on.
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
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-fallback case (root writes through the mode bits)"
else
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
    _out=$(_run "$_TMP/livehome" '' "$_refused")
    assert_eq "a writable fallback still wins"      "studio" "$(echo "$_out" | cut -d' ' -f1)"
    assert_eq "and it is the Studio cache"          "$_TMP/livehome/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
    chmod u+w "$_refused"
fi

echo "=== the same answers under a real /bin/sh, and under the errexit install.sh runs with ==="
# install.sh runs under /bin/sh (dash, busybox ash) with set -e, so re-run the core verdicts
# across every available /bin/sh with errexit both ways.
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
# A root-owned bucket or dangling link leaves the root writable while uv still aborts on it.
_brokenstudio="$_TMP/brokenstudio"
mkdir -p "$_brokenstudio/cache/uv"
ln -s "$_TMP/no-such-target" "$_brokenstudio/cache/uv/archive-v0"
_cold_default="$_TMP/colddefault"
mkdir -p "$_cold_default"
_out=$(_run "$_brokenstudio" '' "$_cold_default")
assert_eq "broken Studio bucket -> not studio" "shared"        "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and uv's default is used instead"   "$_cold_default" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "and the launch does not repoint"    "$_cold_default" "$(echo "$_out" | cut -d' ' -f3)"
_out=$(_run "$_brokenstudio" '' "$_populated")
assert_eq "warm default beside a broken Studio" "shared"     "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and the launch keeps it"             "$_populated" "$(echo "$_out" | cut -d' ' -f3)"
_out=$(_run "$_TMP/okstudio" '' "$_cold_default")
assert_eq "an intact Studio cache still wins"   "studio"     "$(echo "$_out" | cut -d' ' -f1)"

echo "=== a marker naming a directory that is gone is stale, not a decision ==="
# A marker naming a deleted cache must not abandon a warm Studio cache.
_stale=$_TMP/stalemarker
mkdir -p "$_stale/studio/cache/uv/archive-v0/torch"
: > "$_stale/studio/cache/uv/archive-v0/torch/libtorch.so"
: > "$_stale/studio/cache/uv/CACHEDIR.TAG"
printf '%s\n' "$_TMP/deleted-cache" > "$_stale/studio/cache/uv-cache-dir"
_out=$(_run "$_stale/studio" '' "$_cold_home")
assert_eq "stale marker -> keep the Studio cache" "studio" "$(echo "$_out" | cut -d' ' -f1)"
assert_eq "and not uv's default"                  "$_stale/studio/cache/uv" "$(echo "$_out" | cut -d' ' -f2)"
assert_eq "a warm default still outranks a stale marker" "shared" \
    "$(_run "$_stale/studio" '' "$_populated" | cut -d' ' -f1)"

echo "=== a root we can create in but not unlink from is not writable ==="
# Create succeeding does not mean uv can rename into it (NTFS DELETE ACE, append-only dirs).
# Simulated with a failing rm, since mode bits cannot express it.
_probe_unlink_case() {
    # shellcheck disable=SC2317
    rm() { return 1; }
    if _uv_cache_root_is_writable "$1"; then echo writable; else echo unwritable; fi
}
# A failed delete that leaves nothing behind counts as a delete (indexer holding the handle).
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
