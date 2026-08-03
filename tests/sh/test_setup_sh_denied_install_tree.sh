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

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then echo "FAILED"; exit 1; fi
echo "ALL PASSED"
