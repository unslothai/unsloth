#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# studio/setup.sh is the standalone entry point `unsloth studio update` runs, so it
# picks its own uv cache. It used to pick $STUDIO_HOME/cache/uv unconditionally, which
# on a shared-mode install is the EMPTY one -- the wheels are in uv's own cache, and the
# update refetched every one of them.
#
# The marker install.sh recorded is the only thing that can tell those apart, so this
# reads it, on the same terms the CLI does (unsloth_cli/commands/studio.py):
#   * an explicit UV_CACHE_DIR wins and is never recorded
#   * UV_NO_CACHE leaves it unset entirely
#   * the marker wins only while its cache still holds packages
#   * otherwise the Studio path, and only if it is writable
# setup.sh never WRITES the marker: it infers, and an inference written down as a
# decision is how a stale marker outlives the install that justified it.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

# The real helpers plus the real selector, lifted out of setup.sh so nothing here is a
# paraphrase. The selector is not a function, so it is sliced by its own anchors.
HELPERS=$(awk '
    /^_uv_no_cache_requested\(\) \{/ { grab = 1 }
    /^_uv_cache_probe_writable\(\) \{/ { grab = 1 }
    /^_uv_cache_warm\(\) \{/ { grab = 1 }
    /^_recorded_uv_cache\(\) \{/ { grab = 1 }
    /^_UV_MARKER_BOM=/ { print; next }
    /^_UV_MARKER_CR=/ { print; next }
    /^_UV_MARKER_LF=/ { print; next }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")
SELECTOR=$(awk '
    /^if \[ -n "\$\{UV_CACHE_DIR:-\}" \]; then$/ { grab = 1 }
    grab { print }
    grab && /^fi$/ { exit }
' "$SETUP_SH")

for _need in _uv_no_cache_requested _uv_cache_probe_writable _uv_cache_warm _recorded_uv_cache; do
    if ! printf '%s\n' "$HELPERS" | grep -q "^${_need}() {"; then
        echo "FATAL: could not extract $_need from setup.sh" >&2
        exit 1
    fi
done
printf '%s\n' "$HELPERS" | grep -q '^_UV_MARKER_BOM=' || {
    echo "FATAL: the BOM constant is gone" >&2; exit 1; }
printf '%s\n' "$SELECTOR" | grep -q '_uv_cache_warm "\$_uv_recorded"' || {
    echo "FATAL: could not extract the selector from setup.sh" >&2; exit 1; }

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

# The same probe under the options setup.sh actually sets on line 5. Every case here runs
# the selector under a bare `sh`/`bash`, which is a weaker shell than the one that ships:
# `set -o pipefail` turns a SIGPIPE inside a command substitution into a failed pipeline,
# and that is the difference between reading a real cache as warm and refetching it.
run_strict() {  # run_strict <state> <input> <no-cache-state> <no-cache> <studio home>
    bash -e -u -o pipefail "$PROBE" "$1" "$2" "$3" "$4" "$5"
}

# A bucket with more file names in it than a 64K pipe can hold. Built once, outside the
# per-shell loop, because `touch`ing it twice is the only slow thing in this file.
# `find ... -print | head -n 1` reads THIS cache as cold under pipefail: head has its
# answer after one line and exits, find is killed writing the rest, and the pipeline's
# non-zero status sends the match to `|| _uvw_hit=""`. Every other fixture here is a
# handful of files, which is exactly why the suite stayed green while `unsloth studio
# update` walked away from warm caches on real machines.
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

    # A recorded cache that has gone read-only since the install (a share remounted
    # read-only): warm, and useless to uv, which aborts on a cache it cannot write.
    RO="$CASE/read-only recorded/uv"
    warm "$RO"
    record "$HOME_DIR" "$RO\\n"
    if [ "$(id -u 2>/dev/null || echo 0)" != 0 ] && chmod 555 "$RO" 2>/dev/null; then
        assert_eq "$shell: a warm recorded cache that is not writable loses to the Studio cache" \
            "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
        chmod 755 "$RO" 2>/dev/null || true
    fi

    # A cache the user cleared, or one recorded by an install whose cache has since
    # been deleted: content, not the record, has the last word on emptiness.
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

    for bucket in archive-v0 builds-v0 built-wheels-v3 sdists-v9; do
        BUCKET_CACHE="$CASE/$bucket/uv"
        warm "$BUCKET_CACHE" "$bucket"
        record "$HOME_DIR" "$BUCKET_CACHE\\n"
        assert_eq "$shell: $bucket counts as package data" \
            "$BUCKET_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    done

    # ...and one that is warm by thousands of files rather than by one. Under the shell
    # setup.sh really runs, the old `-print | head -n 1` scan answered "cold" for this and
    # sent the update to an empty Studio cache to refetch every wheel it already had.
    record "$HOME_DIR" "$BIG\\n"
    assert_eq "$shell: a cache too big for one pipe buffer is still warm" \
        "$BIG" "$(run "$shell" unset "" unset "" "$HOME_DIR")"
    if [ "$shell" = bash ]; then
        assert_eq "$shell: ...under setup.sh's own set -euo pipefail too" \
            "$BIG" "$(run_strict unset "" unset "" "$HOME_DIR")"
    fi

    # A relative record names a different directory in each phase and there is nothing
    # here to resolve it against, so it is declined rather than guessed at.
    record "$HOME_DIR" "relative/cache\\n"
    assert_eq "$shell: a relative record is declined" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    record "$HOME_DIR" "\\n"
    assert_eq "$shell: an empty record is declined" \
        "$STUDIO_CACHE" "$(run "$shell" unset "" unset "" "$HOME_DIR")"

    # A pathname that itself ends in a newline. The CLI writes it plus one delimiter and
    # reads it back by removing exactly one; a reader that let command substitution strip
    # every trailing newline checked a different directory and called the cache cold.
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

    for truthy in 1 true TRUE yes ON " on "; do
        assert_eq "$shell: UV_NO_CACHE=[$truthy] leaves the cache unset" \
            "<unset>" "$(run "$shell" unset "" value "$truthy" "$HOME_DIR")"
    done
    # An exported EMPTY UV_CACHE_DIR is not a caller value, and uv parses it as an empty
    # --cache-dir and fails; no-cache mode has to unset it rather than leave it.
    assert_eq "$shell: an empty UV_CACHE_DIR under UV_NO_CACHE is unset, not kept" \
        "<unset>" "$(run "$shell" value "" value 1 "$HOME_DIR")"
    for falsy in 0 false "" maybe; do
        assert_eq "$shell: UV_NO_CACHE=[$falsy] changes nothing" \
            "$SHARED" "$(run "$shell" unset "" value "$falsy" "$HOME_DIR")"
    done

    # uv aborts on a cache it cannot create, so an unwritable Studio path has to unset
    # rather than export: uv's own default still works.
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
