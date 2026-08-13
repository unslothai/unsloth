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

# Same, with $2 on the loader's search path. LD_LIBRARY_PATH stands in for the build
# host's own /usr/lib: a test cannot install into it, and what the gate has to notice
# is only that a name resolved somewhere OUTSIDE the AppDir -- which is the place that
# will not exist on the user's machine.
_verify_with_hostpath() {  # dir, hostdir -> "ok"/"rejected"
    if LD_LIBRARY_PATH="$2" bash "$BUILD_SH" --verify-appdir "$1" >/dev/null 2>&1
    then echo ok; else echo rejected; fi
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

    echo "=== a dependency the BUILD HOST happens to provide is still rejected ==="
    # "not found" is the easy half. The dangerous half is a library the closure walk
    # failed to copy that RESOLVES here anyway, because the loader searches the host
    # after RUNPATH -- the bundle then reads as complete and fails on a target that
    # does not carry it. Measured against --verify-appdir before this was tightened:
    # an AppDir needing libz.so.1 and bundling only WebKit was accepted, and the CI
    # step that re-verifies the SHIPPED squashfs inherited the same blind spot.
    _HOSTLIBS="$_TMP/hostlibs"; mkdir -p "$_HOSTLIBS"
    make_appdir "$_TMP/hostdep" complete
    printf 'int host_symbol(void){return 3;}\n' > "$_TMP/hl.c"
    gcc -shared -fPIC -o "$_HOSTLIBS/libhostonly.so.1" "$_TMP/hl.c" \
        -Wl,-soname,libhostonly.so.1 2>/dev/null
    printf 'int host_symbol(void); int webkit_symbol(void);\nint main(void){return host_symbol()+webkit_symbol();}\n' > "$_TMP/mh.c"
    gcc -o "$_TMP/hostdep/usr/bin/unsloth-studio" "$_TMP/mh.c" \
        -L"$_TMP/hostdep/usr/lib/unsloth" -l:libwebkit2gtk-4.1.so.0 \
        -L"$_HOSTLIBS" -l:libhostonly.so.1 2>/dev/null
    patchelf --set-rpath '$ORIGIN/../lib/unsloth' \
        "$_TMP/hostdep/usr/bin/unsloth-studio" 2>/dev/null || true
    assert_eq "a host-resolved dependency is rejected" "rejected" \
        "$(_verify_with_hostpath "$_TMP/hostdep" "$_HOSTLIBS")"

    echo "=== but a HOST-BOUNDARY library, and its own tail, still passes ==="
    # The boundary libraries are supposed to resolve on the host, and each drags its
    # own dependencies with it: libX11 pulls libxcb, libXdmcp, libbsd and libmd, none
    # of which this bundle ships or should. Judging everything ldd prints -- rather
    # than each object's own DT_NEEDED -- would reject every correct build on that
    # tail alone, so this is the case that keeps the gate from being unusable.
    make_appdir "$_TMP/hostok" complete
    printf 'int tail_symbol(void){return 5;}\n' > "$_TMP/tl.c"
    gcc -shared -fPIC -o "$_HOSTLIBS/libtail.so.1" "$_TMP/tl.c" \
        -Wl,-soname,libtail.so.1 2>/dev/null
    printf 'int tail_symbol(void); int egl_symbol(void){return tail_symbol();}\n' > "$_TMP/eg.c"
    gcc -shared -fPIC -o "$_HOSTLIBS/libEGL.so.1" "$_TMP/eg.c" \
        -Wl,-soname,libEGL.so.1 -L"$_HOSTLIBS" -l:libtail.so.1 2>/dev/null
    printf 'int egl_symbol(void); int webkit_symbol(void);\nint main(void){return egl_symbol()+webkit_symbol();}\n' > "$_TMP/me.c"
    # -rpath-link only tells the LINKER where libEGL's own libtail is; it records no
    # RUNPATH, so at verify time libtail is still reached through the host path alone.
    gcc -o "$_TMP/hostok/usr/bin/unsloth-studio" "$_TMP/me.c" \
        -L"$_TMP/hostok/usr/lib/unsloth" -l:libwebkit2gtk-4.1.so.0 \
        -L"$_HOSTLIBS" -Wl,-rpath-link,"$_HOSTLIBS" -l:libEGL.so.1 2>/dev/null
    patchelf --set-rpath '$ORIGIN/../lib/unsloth' \
        "$_TMP/hostok/usr/bin/unsloth-studio" 2>/dev/null || true
    assert_eq "a host-boundary library is accepted" "ok" \
        "$(_verify_with_hostpath "$_TMP/hostok" "$_HOSTLIBS")"
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

_LINK="/tmp/.unsloth-webkit"
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

echo "=== behavioural: the WebKit RUNPATH sweep reaches non-executable .so files ==="
# What shipped: the sweep matched `-perm -u+x` only. A shared object needs no execute
# bit -- Debian installs libwebkit2gtkinjectedbundle.so as 0644 -- so patchelf never ran
# on it and it kept an EMPTY RUNPATH. Nix marks its .so files executable, so this was
# invisible on the build host and only failed on Ubuntu. The injected bundle also sits
# one directory deeper than the helpers, so a fixed '../../' points one level short.
_wk="$_TMP/rpath/AppDir/usr/libexec/unsloth-webkit"
_ld="$_TMP/rpath/AppDir/usr/lib/unsloth"
mkdir -p "$_wk/injected-bundle" "$_ld"
: > "$_wk/WebKitNetworkProcess"; chmod 755 "$_wk/WebKitNetworkProcess"
: > "$_wk/injected-bundle/libwebkit2gtkinjectedbundle.so"
chmod 644 "$_wk/injected-bundle/libwebkit2gtkinjectedbundle.so"

# Lift the real find expression out of the script so this cannot drift from it.
# Match on the sweep wherever it lives: it was three per-directory loops, then one
# unified loop over $libdir and $webkit_exec. What must not change is that it selects
# .so files by NAME and not by mode alone.
_sweep=$(grep -E 'find .*\$webkit_exec.* -type f' "$BUILD_SH" | head -1)
assert_eq "sweep matches .so by name, not just by mode" "yes" \
    "$(printf '%s' "$_sweep" | grep -q "name '\*\.so\*'" && echo yes || echo no)"

_matched=$(find "$_wk" -type f \( -perm -u+x -o -name '*.so*' \) | wc -l | tr -d ' ')
assert_eq "both helper and injected bundle are swept" "2" "$_matched"

# Each object must get its OWN way back to libdir, not a fixed depth.
_ok=yes
while IFS= read -r _obj; do
    _rel=$(realpath --relative-to="$(dirname "$_obj")" "$_ld")
    _resolved=$(cd "$(dirname "$_obj")" && cd "$_rel" && pwd)
    [ "$_resolved" = "$(cd "$_ld" && pwd)" ] || _ok=no
done < <(find "$_wk" -type f \( -perm -u+x -o -name '*.so*' \))
assert_eq "every swept object's RUNPATH resolves to libdir" "yes" "$_ok"
assert_eq "depth is computed, not hardcoded" "yes" \
    "$(grep -q 'realpath --relative-to' "$BUILD_SH" && echo yes || echo no)"

echo "=== behavioural: the pixbuf loader cache survives relocation ==="
# Two faults in one file. gdk-pixbuf resolves a RELATIVE module path against the
# process CWD -- not the cache, not GDK_PIXBUF_MODULEDIR -- and AppRun keeps whatever
# directory the user launched from, so './libpixbufloader-svg.so' was found only by
# luck. And gdk-pixbuf-query-loaders writes an absolute LoaderDir header, so rewriting
# only the module lines left the build host's path in the shipped bundle, which the
# CI leak check caught.
_pb="$_TMP/pbcache"; mkdir -p "$_pb"
_fake_appdir="/build/AppDir"
cat > "$_pb/in" <<EOF
# LoaderDir = $_fake_appdir/usr/lib/unsloth/gdk-pixbuf
#
"$_fake_appdir/usr/lib/unsloth/gdk-pixbuf/libpixbufloader-svg.so"
EOF
sed "s|$_fake_appdir|@APPDIR@|g" "$_pb/in" > "$_pb/cache"
assert_eq "no build-host path survives the build" "0" \
    "$(grep -c "$_fake_appdir" "$_pb/cache" || true)"
sed "s|@APPDIR@|/tmp/.mount_ABC|g" "$_pb/cache" > "$_pb/expanded"
_rel=0
while IFS= read -r _m; do
    case "$_m" in /*) ;; *) _rel=$((_rel + 1)) ;; esac
done < <(grep -oE '^"[^"]+\.so"' "$_pb/expanded" | tr -d '"')
assert_eq "every module path is absolute after expansion" "0" "$_rel"
assert_eq "the LoaderDir header is relocated too" "yes" \
    "$(grep -q 'LoaderDir = /tmp/.mount_ABC' "$_pb/expanded" && echo yes || echo no)"

assert_eq "build writes the token, not a relative path" "yes" \
    "$(grep -q 's|\$app_dir|@APPDIR@|g' "$BUILD_SH" && echo yes || echo no)"
assert_eq "build fails if a build path survives" "yes" \
    "$(grep -q 'still contains build-host paths' "$BUILD_SH" && echo yes || echo no)"
assert_eq "AppRun expands the token at launch" "yes" \
    "$(grep -q 's|@APPDIR@|\$appdir|g' "$BUILD_SH" && echo yes || echo no)"

echo "=== structural: GLES comes from the host, and the fallback has two triggers ==="
# glvnd's libGLESv2 is a dispatch shim that binds a vendor only when the host glvnd is
# the one it was built against. A container matrix over seven hosts found the packaged
# copy renders on exactly one -- its build host -- and fails even where there is no
# host copy to shadow, so packaging it buys nothing the DMA-BUF fallback does not.
assert_eq "libGLESv2 is not force-bundled" "yes" \
    "$(sed -n '/^DLOPEN_LIBS=(/,/^)/p' "$BUILD_SH" | grep -q 'libGLESv2' && echo no || echo yes)"
assert_eq "the probe asks the host, not the bundle" "yes" \
    "$(grep -q 'ldconfig -p .*libGLESv2' "$BUILD_SH" && echo yes || echo no)"
# Two independent faults produce the same blank window: a packaged shim (above), and
# WebKit's DMA-BUF renderer on Wayland, which fails on a Steam Deck even with the
# host's own libGLESv2 in use. The second does not reproduce under Xvfb/llvmpipe.
assert_eq "fallback fires when the host cannot serve GLES" "yes" \
    "$(grep -q 'this system has no libGLESv2' "$BUILD_SH" && echo yes || echo no)"
assert_eq "fallback also fires on a Wayland session" "yes" \
    "$(grep -q 'XDG_SESSION_TYPE:-}" = "wayland"' "$BUILD_SH" && echo yes || echo no)"
assert_eq "it degrades rather than refusing to start" "yes" \
    "$(grep -q 'WEBKIT_DISABLE_DMABUF_RENDERER=1' "$BUILD_SH" && echo yes || echo no)"

echo "=== structural: the tauri bundle-type stamp ==="
assert_eq "tauri bundle-type stamped" "yes" \
    "$(grep -q 'stamp_appimage_bundle_type "\$binary_file"' "$BUILD_SH" && echo yes || echo no)"

echo "=== behavioural: the guard checks the LINK's owner, not the payload's ==="
# The regression this encodes: appimagetool normalises the payload to root:root, and
# `test -O` DEREFERENCES, so the guard asked "is the bundle owned by me?" -- false for
# every non-root user. Every launch of a CI-built AppImage died with exit 126. The
# fixture Daniel's tests use is owned by the test user, so it could not see this.
# /usr stands in for the root-owned payload: it is not ours on any normal system.
_root_owned=""
for _cand in /usr /usr/lib /lib; do
    if [ -d "$_cand" ] && [ ! -O "$_cand" ]; then _root_owned="$_cand"; break; fi
done
if [ -n "$_root_owned" ]; then
    ln -sfn "$_root_owned" "$_TMP/foreign-target-link"
    # Old check: -O follows the link and sees the foreign owner -> would refuse.
    assert_eq "dereferencing owner check rejects a root-owned payload" "refused" \
        "$([ -O "$_TMP/foreign-target-link" ] && echo accepted || echo refused)"
    # New check: stat without -L reads the link itself, which we created -> accepted.
    assert_eq "link-owner check accepts our own link to it" "$(id -u)" \
        "$(stat -c %u "$_TMP/foreign-target-link" 2>/dev/null)"
else
    echo "  SKIP: no foreign-owned directory available to stand in for the payload"
fi
assert_eq "guard reads the link, not the target" "yes" \
    "$(grep -q 'stat -c %u "\$WEBKIT_LINK"' "$BUILD_SH" && echo yes || echo no)"
assert_eq "guard no longer uses the dereferencing -O test" "yes" \
    "$(grep -q '\-O "\$WEBKIT_LINK"' "$BUILD_SH" && echo no || echo yes)"

echo "=== behavioural: libstdc++ is host-provided, not bundled ==="
# Measured on a Steam Deck: WebKit pulls the bundled libstdc++ in through its own
# $ORIGIN RUNPATH -- no LD_LIBRARY_PATH -- and the loader then reuses that SONAME for
# every host library opened later, so host Mesa died on GLIBCXX_3.4.32 (required by
# libSPIRV-Tools) while loading fine on its own. A bundled C++ runtime OLDER than the
# host's poisons the host graphics stack, which is the same argument that keeps glibc
# out of the bundle.
grep '^HOST_LIBS_RE=' "$BUILD_SH" > "$_TMP/hostlibs.sh"
_classify() {
    bash -c '. "$1"; if [[ "$2" =~ $HOST_LIBS_RE ]]; then echo HOST; else echo bundled; fi' \
        _ "$_TMP/hostlibs.sh" "$1"
}
for _h in libstdc++.so.6 libgcc_s.so.1 libc.so.6 libnghttp2.so.14; do
    assert_eq "$_h is host-provided" "HOST" "$(_classify "$_h")"
done
# The desktop stack must still travel with us, or this is a thin bundle again.
for _b in libwebkit2gtk-4.1.so.0 libsoup-3.0.so.0 libsqlite3.so.0; do
    assert_eq "$_b is bundled" "bundled" "$(_classify "$_b")"
done

echo "=== structural: the host/bundle boundary ==="
# glibc and the GPU stack must stay on the host; the desktop stack must not.
for _host in 'libc' 'libGL' 'libEGL' 'libdrm' 'libX11' 'libwayland-'; do
    assert_eq "$_host is host-provided" "yes" \
        "$(grep -q "$_host" "$BUILD_SH" && echo yes || echo no)"
done
# AppRun must NOT export LD_LIBRARY_PATH. It is global, so it outranks the default
# search path for every library the process opens -- including the host GL/EGL/curl
# stack this bundle deliberately does not ship, which then resolves ITS dependencies
# out of the 22.04 bundle. Measured on Ubuntu 24.04 (Linux Mint 22 Wilma's base) while
# the export was in place: host libcurl-gnutls hit the #7953 undefined nghttp2 symbol
# and host libGLX_mesa hit a missing GLIBCXX_3.4.32. The bundle resolves through
# $ORIGIN RUNPATHs instead, which the build asserts on every object.
assert_eq "AppRun does not export LD_LIBRARY_PATH" "yes" \
    "$(grep -q '^export LD_LIBRARY_PATH=' "$BUILD_SH" && echo no || echo yes)"
assert_eq "the build asserts \$ORIGIN RUNPATHs" "yes" \
    "$(grep -q 'ORIGIN-relative RUNPATH' "$BUILD_SH" && echo yes || echo no)"
assert_eq "an empty RUNPATH is rejected" "yes" \
    "$(grep -q 'RUNPATH empty (patchelf did nothing)' "$BUILD_SH" && echo yes || echo no)"
assert_eq "the closure check runs without LD_LIBRARY_PATH" "yes" \
    "$(grep -q 'LD_LIBRARY_PATH="\$libdir" ldd' "$BUILD_SH" && echo no || echo yes)"
assert_eq "WebKit helper path exported" "yes" \
    "$(grep -q 'WEBKIT_EXEC_PATH' "$BUILD_SH" && echo yes || echo no)"

assert_eq "GIO modules redirected" "yes" \
    "$(grep -q 'GIO_MODULE_DIR' "$BUILD_SH" && echo yes || echo no)"
assert_eq "pixbuf loaders redirected" "yes" \
    "$(grep -q 'GDK_PIXBUF_MODULE_FILE' "$BUILD_SH" && echo yes || echo no)"

echo "=== behavioural: AppRun refuses a WebKit link it does not control ==="
# The link path is fixed and /tmp is world-writable, so another user can pre-create it;
# the sticky bit then blocks our replace. Swallowing that leaves WebKit spawning THEIR
# helpers, so the guard must abort instead. The foreign-owner case needs a second uid,
# but it fails through the same arm as a target with no helpers in it.
_GUARD="$_TMP/guard.sh"
awk '/^WEBKIT_LINK="__WEBKIT_LINK_PATH__"$/,/^fi$/' "$BUILD_SH" > "$_GUARD"
assert_eq "guard extracted from AppRun" "yes" \
    "$([ -s "$_GUARD" ] && echo yes || echo no)"
_run_guard() {  # $1 = appdir -> ok/refused
    _l="$_TMP/link-$2"
    sed "s|__WEBKIT_LINK_PATH__|$_l|" "$_GUARD" > "$_TMP/g.sh"
    if (appdir="$1"; . "$_TMP/g.sh") 2>/dev/null; then echo ok; else echo refused; fi
}
_good="$_TMP/good/usr/libexec/unsloth-webkit"
mkdir -p "$_good"; : > "$_good/WebKitNetworkProcess"; chmod +x "$_good/WebKitNetworkProcess"
assert_eq "a real bundle is accepted" "ok" "$(_run_guard "$_TMP/good" good)"
mkdir -p "$_TMP/empty/usr/libexec/unsloth-webkit"
assert_eq "a target with no helpers is refused" "refused" "$(_run_guard "$_TMP/empty" empty)"
assert_eq "the failure is never swallowed" "no" \
    "$(grep -q 'ln -sfn .*unsloth-webkit.* || true' "$BUILD_SH" && echo yes || echo no)"

echo "=== behavioural: a second launch must not steal a running instance's link ==="
# The link is UID-global, each mount is per-process. Re-creating it unconditionally
# meant a second launch repointed it at its own /tmp/.mount_XXXXXX, single-instance
# then rejected that launch, and the AppImage runtime unmounted as it exited --
# leaving the FIRST, still-running process pointed at nothing. WebKit reads the
# compiled-in path on every helper spawn and g_error()s when the spawn fails, so the
# running app aborts. A link naming a mount that is GONE must still be replaced,
# otherwise the next launch inherits a dangling one.
_run_guard_at() {  # appdir, link -> ok/refused
    sed "s|__WEBKIT_LINK_PATH__|$2|" "$_GUARD" > "$_TMP/gshared.sh"
    if (appdir="$1"; . "$_TMP/gshared.sh") 2>/dev/null; then echo ok; else echo refused; fi
}
_LINK="$_TMP/link-shared"
for _inst in A B; do
    mkdir -p "$_TMP/inst$_inst/usr/libexec/unsloth-webkit"
    : > "$_TMP/inst$_inst/usr/libexec/unsloth-webkit/WebKitNetworkProcess"
    chmod +x "$_TMP/inst$_inst/usr/libexec/unsloth-webkit/WebKitNetworkProcess"
done
assert_eq "first launch takes the link" "ok" "$(_run_guard_at "$_TMP/instA" "$_LINK")"
assert_eq "the link names the first instance" "$_TMP/instA/usr/libexec/unsloth-webkit" \
    "$(readlink "$_LINK")"
assert_eq "a second launch is still accepted" "ok" "$(_run_guard_at "$_TMP/instB" "$_LINK")"
assert_eq "the second launch left the link alone" "$_TMP/instA/usr/libexec/unsloth-webkit" \
    "$(readlink "$_LINK")"
rm -rf "$_TMP/instA"   # the first instance exited; its mount is gone
assert_eq "a stale link is replaced" "ok" "$(_run_guard_at "$_TMP/instB" "$_LINK")"
assert_eq "the link now names the live instance" "$_TMP/instB/usr/libexec/unsloth-webkit" \
    "$(readlink "$_LINK")"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
