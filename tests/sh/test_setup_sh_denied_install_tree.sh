#!/bin/bash
# studio/setup.sh must tell an unreadable install tree apart from somebody else's,
# and must not abort with a raw rm error when it cannot replace one. setup.ps1
# learned both in #7735/#7757; this is the POSIX side.
#
# Pins: a tree that IS ours goes unsearchable, every probe inside reports "absent",
# and the ownership guard blamed ownership ("Move it aside or choose an empty
# UNSLOTH_STUDIO_HOME") when the fix is a permission change. The marker probes need
# search (+x), and a directory can be readable but unsearchable (444) or the reverse
# (111), so probe search, not read.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
assert_contains() {
    if grep -qF -- "$3" "$2"; then ok "$1"; else bad "$1 (expected '$3')"; fi
}

echo ""
echo "=== setup.sh: the guards exist and probe the right permission ==="

assert_contains "defines the unsearchable-directory probe" \
    "$SETUP_SH" "_studio_dir_unsearchable() {"
# cd needs +x, exactly what the marker probes need; ls needs +r, which is neither
# sufficient nor necessary. Pin the cd form.
assert_contains "the probe tests search (cd), not read (ls)" \
    "$SETUP_SH" '( cd "$1" ) 2>/dev/null && return 1'
assert_contains "defines the denial reporter" \
    "$SETUP_SH" "_path_access_denied() {"
assert_contains "the ownership guard checks readability before blaming ownership" \
    "$SETUP_SH" '_path_access_denied "$_aso_dir" "$_aso_label" owner-unverified'
assert_contains "an unverifiable tree is never described as not ours" \
    "$SETUP_SH" "Unsloth cannot confirm this folder is its own install while it is unreadable"

# The custom-home ownership guard does not cover the default cache, so verify
# the prebuilt path rejects an unreadable default cache before Python runs.
PREBUILT_BLOCK="$(awk '/substep "installing prebuilt llama.cpp\.\.\."/,/_PREBUILT_CMD=\(/' "$SETUP_SH")"
if printf '%s' "$PREBUILT_BLOCK" | grep -qF '_studio_dir_unreadable "$LLAMA_CPP_DIR"' &&
   printf '%s' "$PREBUILT_BLOCK" | grep -qF '_path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"'; then
    ok "the default-home prebuilt install stops on an unreadable tree"
else
    bad "the default-home prebuilt install stops on an unreadable tree"
fi
# The source build only reaches its own guards after building, so the access check
# has to run before the prebuilt/source branch, and skip the local-link paths.
PRE_BRANCH="$(awk '/^if \[ "\$_LOCAL_LLAMA_CPP_LINKED" != true \]; then/,/^fi$/' "$SETUP_SH")"
if printf '%s' "$PRE_BRANCH" | grep -qF '_studio_dir_unreadable "$LLAMA_CPP_DIR"' &&
   printf '%s' "$PRE_BRANCH" | grep -qF '_assert_studio_owned_or_absent "$LLAMA_CPP_DIR"'; then
    ok "the access guard runs before the prebuilt/source branch"
else
    bad "the access guard runs before the prebuilt/source branch"
fi
PRE_AT="$(grep -n '^if \[ "\$_LOCAL_LLAMA_CPP_LINKED" != true \]; then' "$SETUP_SH" | cut -d: -f1 | head -1)"
BRANCH_AT="$(grep -n '^if \[ "\$_LOCAL_LLAMA_CPP_LINKED" = true \]; then' "$SETUP_SH" | cut -d: -f1 | head -1)"
if [ -n "$PRE_AT" ] && [ -n "$BRANCH_AT" ] && [ "$PRE_AT" -lt "$BRANCH_AT" ]; then
    ok "the early guard precedes the branch it protects"
else
    bad "the early guard precedes the branch it protects"
fi

# Run the custom-home ownership guard first to preserve its cautious wording.
# Default the line numbers so a vanished grep fails loudly instead of erroring out.
OWNED_LINE="$(printf '%s' "$PREBUILT_BLOCK" | grep -n '_assert_studio_owned_or_absent' | cut -d: -f1 | head -1)"
UNREADABLE_LINE="$(printf '%s' "$PREBUILT_BLOCK" | grep -n '_studio_dir_unreadable' | cut -d: -f1 | head -1)"
if [ -n "$OWNED_LINE" ] && [ -n "$UNREADABLE_LINE" ] && [ "$OWNED_LINE" -lt "$UNREADABLE_LINE" ]; then
    ok "the ownership guard still reports a custom home first"
else
    bad "the ownership guard still reports a custom home first"
fi

echo ""
echo "=== setup.sh: neither destructive replace runs blind ==="

# A failing rm -rf under errexit aborts with no [TAURI:ERROR], so the desktop app
# shows a bare exit code.
if [ "$(grep -c 'rm -rf "$LLAMA_CPP_DIR" || true' "$SETUP_SH")" -ge 2 ]; then
    ok "both replace sites tolerate a failing rm instead of aborting raw"
else
    bad "both replace sites tolerate a failing rm instead of aborting raw"
fi
# rm names the exact failing subpath; our messages can only name the install root.
if grep -q 'rm -rf "$LLAMA_CPP_DIR" 2>/dev/null' "$SETUP_SH"; then
    bad "the failing rm keeps its stderr"
else
    ok "the failing rm keeps its stderr"
fi
# Mode 111 defeats the rm but stays searchable, so a replace postcondition on the
# search-only probe falls through to the generic "could not be replaced" message.
if [ "$(grep -c '_studio_dir_unsearchable "\$LLAMA_CPP_DIR"' "$SETUP_SH")" = 0 ]; then
    ok "no replace site still uses the search-only probe"
else
    bad "no replace site still uses the search-only probe"
fi
if [ "$(grep -c 'if \[ -e "$LLAMA_CPP_DIR" \]; then' "$SETUP_SH")" -ge 2 ]; then
    ok "both replace sites check the postcondition"
else
    bad "both replace sites check the postcondition"
fi
assert_contains "a stranded build reports where it was left" \
    "$SETUP_SH" 'The new build is at $_BUILD_TMP.'
assert_contains "a bare rm -rf of the install dir is gone" \
    "$SETUP_SH" 'rm -rf "$LLAMA_CPP_DIR" || true'
if grep -qE '^\s*rm -rf "\$LLAMA_CPP_DIR"\s*$' "$SETUP_SH"; then
    bad "no unguarded rm -rf of the install dir remains"
else
    ok "no unguarded rm -rf of the install dir remains"
fi

echo ""
echo "=== behaviour against a genuinely unsearchable tree ==="

WORK="$(mktemp -d)"
trap 'chmod -R u+rwX "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

# setup.sh runs install steps at load, so extract the real functions instead.
python3 - "$SETUP_SH" "$WORK/helpers.sh" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = []
for name in ("_studio_owned_adoptable", "_studio_dir_unsearchable",
             "_studio_dir_unreadable",
             "_path_access_denied", "_assert_studio_owned_or_absent"):
    i = src.index(name + "() {")
    out.append(src[i:src.index("\n}\n", i) + 3])
pathlib.Path(sys.argv[2]).write_text("\n".join(out))
PY

cat > "$WORK/drive.sh" <<'EOF'
set -uo pipefail
C_ERR= C_WARN= C_DIM= C_OK= C_RST=
step()    { printf 'STEP|%s|%s\n' "$1" "$2"; }
substep() { printf 'SUBSTEP|%s\n' "$1"; }
setup_fail() { printf 'FAIL|%s|%s\n' "$1" "$2"; exit "$1"; }
_STUDIO_OWNED_MARKER=".unsloth-studio-owned"
_STUDIO_HOME_IS_CUSTOM=true
. "$1"
_assert_studio_owned_or_absent "$2" "llama.cpp install"
echo "ACCEPTED"
EOF

OURS="$WORK/ours"; mkdir -p "$OURS"; : > "$OURS/.unsloth-studio-owned"
THEIRS="$WORK/theirs"; mkdir -p "$THEIRS"; : > "$THEIRS/someone-elses.txt"

out=$(bash "$WORK/drive.sh" "$WORK/helpers.sh" "$OURS" 2>&1)
case "$out" in
    *ACCEPTED*) ok "a readable tree of ours is still accepted" ;;
    *) bad "a readable tree of ours is still accepted (got: $out)" ;;
esac

out=$(bash "$WORK/drive.sh" "$WORK/helpers.sh" "$THEIRS" 2>&1)
case "$out" in
    *"not marked as an Unsloth-owned"*) ok "an unowned readable tree still stops on ownership" ;;
    *) bad "an unowned readable tree still stops on ownership (got: $out)" ;;
esac

chmod 000 "$OURS"
# Gate and negative control in one: as root the checks below pass vacuously.
if [ -f "$OURS/.unsloth-studio-owned" ]; then
    echo "  SKIP: this host cannot make a directory unsearchable (running as root?)"
else
    ok "the host really cannot search the tree (negative control)"
    out=$(bash "$WORK/drive.sh" "$WORK/helpers.sh" "$OURS" 2>&1)
    case "$out" in
        *"cannot be read: permission denied"*) ok "an unreadable tree reports permissions" ;;
        *) bad "an unreadable tree reports permissions (got: $out)" ;;
    esac
    case "$out" in
        *"not marked as an Unsloth-owned"*) bad "it must not also blame ownership" ;;
        *) ok "it does not blame ownership" ;;
    esac
    case "$out" in
        *"delete or rename"*|*"Delete or rename"*) bad "it must not advise deleting an unverified tree" ;;
        *) ok "it does not advise deleting an unverified tree" ;;
    esac
fi
chmod 755 "$OURS"

# setup_fail gates [TAURI:ERROR] on two variables. Test them separately: one joined
# case subject would let a comma inside either value alias the other's arm.
python3 - "$SETUP_SH" "$WORK/gate.sh" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
i = src.index("setup_fail() {")
pathlib.Path(sys.argv[2]).write_text(src[i:src.index("\n}\n", i) + 3])
PY
gate_marker() {  # $1=MODE ("-" for unset), $2=UPDATE ("-" for unset)
    env -u UNSLOTH_TAURI_MODE -u UNSLOTH_TAURI_UPDATE \
        ${1:+$([ "$1" = - ] || echo UNSLOTH_TAURI_MODE="$1")} \
        ${2:+$([ "$2" = - ] || echo UNSLOTH_TAURI_UPDATE="$2")} \
        bash -c '. "$1"; setup_fail 1 boom' _ "$WORK/gate.sh" 2>/dev/null | grep -c 'TAURI:ERROR' || true
}
[ "$(gate_marker 1 -)" = 1 ] && ok "the Tauri marker prints for UNSLOTH_TAURI_MODE=1" \
                             || bad "the Tauri marker prints for UNSLOTH_TAURI_MODE=1"
[ "$(gate_marker - 1)" = 1 ] && ok "the Tauri marker prints for UNSLOTH_TAURI_UPDATE=1" \
                             || bad "the Tauri marker prints for UNSLOTH_TAURI_UPDATE=1"
[ "$(gate_marker - -)" = 0 ] && ok "a plain CLI run prints no Tauri marker" \
                             || bad "a plain CLI run prints no Tauri marker"
[ "$(gate_marker 0 'a,1')" = 0 ] && ok "a comma in UNSLOTH_TAURI_UPDATE does not alias the marker" \
                                 || bad "a comma in UNSLOTH_TAURI_UPDATE does not alias the marker"
[ "$(gate_marker '1,2' -)" = 0 ] && ok "a comma in UNSLOTH_TAURI_MODE does not alias the marker" \
                                 || bad "a comma in UNSLOTH_TAURI_MODE does not alias the marker"

# Mode 111 is searchable but not listable, and still breaks install_llama_prebuilt.py.
NOLIST="$WORK/nolist"; mkdir -p "$NOLIST"; : > "$NOLIST/UNSLOTH_PREBUILT_INFO.json"
chmod 111 "$NOLIST"
if ls -A "$NOLIST" >/dev/null 2>&1; then
    echo "  SKIP: this host cannot make a directory unlistable (running as root?)"
else
    ok "the host really cannot list the tree (negative control)"
    probe=$(bash -c '. "$1"; _studio_dir_unsearchable "$2" && echo SEARCH_CAUGHT
        _studio_dir_unreadable "$2" && echo READ_CAUGHT' _ "$WORK/helpers.sh" "$NOLIST")
    case "$probe" in
        *SEARCH_CAUGHT*) bad "mode 111 is searchable, so the search probe must not fire" ;;
        *) ok "mode 111 is searchable, so the search probe does not fire" ;;
    esac
    case "$probe" in
        *READ_CAUGHT*) ok "the read probe catches a searchable but unlistable tree" ;;
        *) bad "the read probe catches a searchable but unlistable tree (got: $probe)" ;;
    esac
fi
chmod 755 "$NOLIST"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then echo "FAILED"; exit 1; fi
echo "ALL PASSED"
