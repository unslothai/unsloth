#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards assert_portable_appdir in build-portable-appimage.sh.
#
# Context: the thin AppImage takes the desktop stack from the host, which cannot work on
# an immutable distro that has no WebKitGTK and no apt -- SteamOS has libgtk-3 and
# libsoup-3 but not libwebkit2gtk-4.1, so the thin AppRun prints an apt command the user
# cannot run. The portable build bundles the stack instead.
#
# The thin script's comment names the real hazard in bundling: a PARTIAL GTK/WebKit
# closure mixes a bundled GLib with the host's newer GIO modules. assert_portable_appdir
# is the answer to that -- it resolves the executable and every bundled object against the
# bundle alone and fails on one "not found", so a partial closure cannot ship. These tests
# exist because that gate is the only thing standing between "bundled" and "bundled
# correctly".
#
# The contract:
#   * complete closure            -> passes
#   * any unresolved DT_NEEDED    -> fails
#   * no bundled WebKit           -> fails (that is a thin bundle under the wrong name)
#   * missing WebKit helpers      -> fails (window opens, page never renders)
#   * no library directory at all -> fails
#
# And for the WebKit path rewrite, which is layout-sensitive:
#   * every distro layout patched -> Debian multiarch, Fedora lib64, Arch, nix
#   * nothing matched             -> fails the BUILD (a no-op must never read as success)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_SH="$SCRIPT_DIR/../../studio/src-tauri/linux/build-portable-appimage.sh"
PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

[ -f "$BUILD_SH" ] || { echo "FAIL: $BUILD_SH not found"; exit 1; }
# gcc and patchelf are only needed to build the ELF closure fixtures. The WebKit
# path tests below need neither, and skipping the whole file when a runner lacks a
# compiler would reproduce the very failure mode these tests exist to catch: a
# silent no-op that reads as a pass. So gate only the sections that need them.
HAVE_TOOLCHAIN=yes
for _tool in gcc patchelf; do
    command -v "$_tool" >/dev/null 2>&1 || {
        echo "SKIP: $_tool missing -- ELF closure fixtures will not run"
        HAVE_TOOLCHAIN=no
    }
done

_TMP=$(mktemp -d)
trap 'rm -rf "$_TMP"' EXIT

# Build a tiny fixture: an executable that needs a bundled library, standing in for
# unsloth-studio needing libwebkit2gtk-4.1.
make_appdir() {  # $1 = dir, $2 = "complete"|"broken"|"nowebkit"|"nohelpers"|"nolibdir"
    _d="$1"; _mode="$2"
    mkdir -p "$_d/usr/bin"
    [ "$_mode" = "nolibdir" ] && return 0
    _lib="$_d/usr/lib/unsloth"; _exec="$_d/usr/libexec/unsloth-webkit"
    mkdir -p "$_lib" "$_exec"

    printf 'int webkit_symbol(void){return 42;}\n' > "$_TMP/w.c"
    gcc -shared -fPIC -o "$_lib/libwebkit2gtk-4.1.so.0" "$_TMP/w.c" \
        -Wl,-soname,libwebkit2gtk-4.1.so.0 2>/dev/null
    printf 'int webkit_symbol(void); int main(void){return webkit_symbol();}\n' > "$_TMP/m.c"
    gcc -o "$_d/usr/bin/unsloth-studio" "$_TMP/m.c" \
        -L"$_lib" -l:libwebkit2gtk-4.1.so.0 2>/dev/null
    patchelf --set-rpath '$ORIGIN/../lib/unsloth' "$_d/usr/bin/unsloth-studio" 2>/dev/null || true

    if [ "$_mode" != "nohelpers" ]; then
        for h in WebKitNetworkProcess WebKitWebProcess; do
            printf 'int main(void){return 0;}\n' > "$_TMP/h.c"
            gcc -o "$_exec/$h" "$_TMP/h.c" 2>/dev/null
        done
    fi
    # "broken": the executable needs a library nothing provides -> partial closure.
    if [ "$_mode" = "broken" ]; then
        printf 'int missing_symbol(void){return 1;}\n' > "$_TMP/x.c"
        gcc -shared -fPIC -o "$_TMP/libphantom.so.1" "$_TMP/x.c" \
            -Wl,-soname,libphantom.so.1 2>/dev/null
        printf 'int missing_symbol(void); int webkit_symbol(void);\nint main(void){return missing_symbol()+webkit_symbol();}\n' > "$_TMP/m2.c"
        gcc -o "$_d/usr/bin/unsloth-studio" "$_TMP/m2.c" \
            -L"$_lib" -l:libwebkit2gtk-4.1.so.0 -L"$_TMP" -l:libphantom.so.1 2>/dev/null
        patchelf --set-rpath '$ORIGIN/../lib/unsloth' "$_d/usr/bin/unsloth-studio" 2>/dev/null || true
        rm -f "$_TMP/libphantom.so.1"   # now genuinely unresolvable
    fi
    [ "$_mode" = "nowebkit" ] && rm -f "$_lib/libwebkit2gtk-4.1.so.0"
    return 0
}

_verify() {  # dir -> "ok"/"rejected"
    if bash "$BUILD_SH" --verify-appdir "$1" >/dev/null 2>&1; then echo ok; else echo rejected; fi
}

if [ "$HAVE_TOOLCHAIN" = yes ]; then
    echo "=== a complete closure passes ==="
    make_appdir "$_TMP/good" complete
    assert_eq "complete closure accepted" "ok" "$(_verify "$_TMP/good")"

    echo "=== a partial closure is rejected (the failure mode thin bundling feared) ==="
    make_appdir "$_TMP/broken" broken
    assert_eq "unresolved dependency rejected" "rejected" "$(_verify "$_TMP/broken")"

    echo "=== a bundle without WebKit is rejected ==="
    make_appdir "$_TMP/nowebkit" nowebkit
    assert_eq "missing libwebkit2gtk rejected" "rejected" "$(_verify "$_TMP/nowebkit")"

    echo "=== a bundle without WebKit helper processes is rejected ==="
    make_appdir "$_TMP/nohelpers" nohelpers
    assert_eq "missing WebKit helpers rejected" "rejected" "$(_verify "$_TMP/nohelpers")"

    echo "=== a bundle with no library directory is rejected ==="
    make_appdir "$_TMP/nolibdir" nolibdir
    assert_eq "missing library dir rejected" "rejected" "$(_verify "$_TMP/nolibdir")"

    echo "=== an ABSOLUTE DT_NEEDED is rejected ==="
    # The fault that shipped: libsoup/libtinysparql/libwebkit2gtk each recorded sqlite as a
    # full build-host path rather than a soname. The loader ignores RUNPATH for such an
    # entry and opens the absolute path, which does not exist on the target -- and ldd on
    # the BUILD host resolves it happily, so the closure check alone cannot see it.
    make_appdir "$_TMP/absneeded" complete
    _absdir="$_TMP/absneeded/usr/lib/unsloth"
    # Deliberately NO -Wl,-soname: with a SONAME the linker records that instead, and the
    # absolute path never appears. A library built without one -- which is how the real
    # sqlite dependency arose -- makes ld record the path exactly as it was given.
    printf 'int extra_symbol(void){return 7;}\n' > "$_TMP/e.c"
    gcc -shared -fPIC -o "$_absdir/libextra.so.1" "$_TMP/e.c" 2>/dev/null
    # Link against the FULL PATH so the recorded DT_NEEDED is absolute.
    printf 'int extra_symbol(void); int webkit_symbol(void);\nint main(void){return extra_symbol()+webkit_symbol();}\n' > "$_TMP/ma.c"
    gcc -o "$_TMP/absneeded/usr/bin/unsloth-studio" "$_TMP/ma.c" \
        "$_absdir/libextra.so.1" -L"$_absdir" -l:libwebkit2gtk-4.1.so.0 2>/dev/null
    patchelf --set-rpath '$ORIGIN/../lib/unsloth' "$_TMP/absneeded/usr/bin/unsloth-studio" 2>/dev/null || true
    _has_abs=$(patchelf --print-needed "$_TMP/absneeded/usr/bin/unsloth-studio" 2>/dev/null | grep -c "/" || true)
    if [ "${_has_abs:-0}" -ge 1 ]; then
        assert_eq "absolute DT_NEEDED rejected" "rejected" "$(_verify "$_TMP/absneeded")"
    else
        echo "  SKIP: toolchain did not record an absolute DT_NEEDED"
    fi
fi

echo "=== structural: the ldd parser sees BOTH ldd output forms ==="
# ldd prints `name => /path` normally but `/path (0x..)` for an absolute DT_NEEDED.
# Matching only the arrow form is why the absolute sqlite was never copied.
assert_eq "parser handles the arrow form"    "yes" \
    "$(grep -q '=>\[\[:space:\]\]\*' "$BUILD_SH" && echo yes || echo no)"
assert_eq "parser handles the bare-path form" "yes" \
    "$(grep -q '(0x' "$BUILD_SH" && echo yes || echo no)"

echo "=== behavioural: WebKit's compiled-in paths, on EVERY distro layout ==="
# WebKitGTK bakes two absolute paths into libwebkit2gtk -- the helper directory and
# the injected bundle -- and their layout is distro-specific:
#
#   Debian/Ubuntu  /usr/lib/x86_64-linux-gnu/webkit2gtk-4.1[/injected-bundle]
#   Fedora         /usr/lib64/webkit2gtk-4.1[/injected-bundle]
#   nix, Arch      $prefix/libexec/webkit2gtk-4.1 + $prefix/lib/.../injected-bundle
#
# The first version of this patch matched '/libexec/...' and '/lib/webkit2gtk-4.X/...'
# literally. Those score ZERO hits on Ubuntu -- which is what release-desktop.yml
# builds on -- and a zero-hit run was indistinguishable from success, so CI would have
# shipped an AppImage carrying the build host's paths. These tests run the real
# patcher against each layout, because only a behavioural test can catch a no-op.
_PATCHER="$_TMP/patcher.py"
awk "/<<'PYEOF'/{f=1;next} f&&/^PYEOF\$/{exit} f" "$BUILD_SH" > "$_PATCHER"
assert_eq "patcher extracted from the build script" "yes" \
    "$([ -s "$_PATCHER" ] && echo yes || echo no)"

_LINK="/tmp/.unsloth-wk-1000"
_mk_wk() {  # $1 = dir, $2.. = NUL-terminated strings to embed
    mkdir -p "$1"; _f="$1/libwebkit2gtk-4.1.so.0"; : > "$_f"; shift
    for _s in "$@"; do printf '%s\0' "$_s" >> "$_f"; done
}
_patch() {  # $1 = dir, $2 = helper src, $3 = injected src -> ok/failed
    if python3 "$_PATCHER" "$1" "$_LINK" "$2" "$3" >/dev/null 2>&1; then
        echo ok; else echo failed; fi
}
_leftover() {  # any build-host webkit path still present?
    strings -a "$1/libwebkit2gtk-4.1.so.0" 2>/dev/null \
        | grep -cE '/webkit2gtk-4\.[01]' || true
}

for _layout in \
    "debian:/usr/lib/x86_64-linux-gnu/webkit2gtk-4.1" \
    "fedora:/usr/lib64/webkit2gtk-4.1" \
    "arch:/usr/lib/webkit2gtk-4.1" \
    "nix:/nix/store/0000000000000000000000000000000-webkitgtk-2.52.4+abi=4.1/libexec/webkit2gtk-4.1"; do
    _name="${_layout%%:*}"; _hdir="${_layout#*:}"
    # Ubuntu's injected bundle carries a trailing slash; keep that in the fixture.
    _idir="${_hdir%/libexec/webkit2gtk-4.1}"
    [ "$_idir" = "$_hdir" ] && _idir="$_hdir/injected-bundle" \
                            || _idir="$_idir/lib/webkit2gtk-4.1/injected-bundle"
    _d="$_TMP/wk-$_name"
    _mk_wk "$_d" "$_hdir" "$_idir/"
    assert_eq "$_name layout patched"  "ok" "$(_patch "$_d" "$_hdir" "$_idir")"
    assert_eq "$_name leaves no build-host path" "0" "$(_leftover "$_d")"
done

echo "=== behavioural: a patch that matches NOTHING fails the build ==="
# The property that matters most: silence is not success.
_mk_wk "$_TMP/wk-nomatch" "no webkit paths in here"
assert_eq "unmatched layout rejected" "failed" \
    "$(_patch "$_TMP/wk-nomatch" /bogus/helpers /bogus/injected)"

echo "=== behavioural: a replacement longer than the original fails ==="
# The rewrite is in-place and NUL-padded, so it can only ever shrink a string.
_mk_wk "$_TMP/wk-short" "/l/webkit2gtk-4.1" "/l/webkit2gtk-4.1/injected-bundle"
assert_eq "over-long replacement rejected" "failed" \
    "$(_patch "$_TMP/wk-short" /l/webkit2gtk-4.1 /l/webkit2gtk-4.1/injected-bundle)"

echo "=== structural: the tauri bundle-type stamp ==="
assert_eq "tauri bundle-type stamped" "yes" \
    "$(grep -q 'stamp_appimage_bundle_type "\$binary_file"' "$BUILD_SH" && echo yes || echo no)"

echo "=== structural: the host/bundle boundary ==="
# glibc and the GPU stack must stay on the host; the desktop stack must not.
for _host in 'libc' 'libGL' 'libEGL' 'libdrm' 'libX11' 'libwayland-'; do
    assert_eq "$_host is host-provided" "yes" \
        "$(grep -q "$_host" "$BUILD_SH" && echo yes || echo no)"
done
assert_eq "AppRun puts the bundle first" "yes" \
    "$(grep -q 'export LD_LIBRARY_PATH="\$libdir' "$BUILD_SH" && echo yes || echo no)"
assert_eq "WebKit helper path exported" "yes" \
    "$(grep -q 'WEBKIT_EXEC_PATH' "$BUILD_SH" && echo yes || echo no)"
assert_eq "GIO modules redirected" "yes" \
    "$(grep -q 'GIO_MODULE_DIR' "$BUILD_SH" && echo yes || echo no)"
assert_eq "pixbuf loaders redirected" "yes" \
    "$(grep -q 'GDK_PIXBUF_MODULE_FILE' "$BUILD_SH" && echo yes || echo no)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
