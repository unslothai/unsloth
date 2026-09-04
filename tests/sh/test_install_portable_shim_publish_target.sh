#!/usr/bin/env bash
# Regression test: the portable shim is published with `mv -f`, and `mv -f` replaces
# its destination only while that destination is not a directory. POSIX gives the
# move-INTO form for a final operand that is a directory "or referenced if target_dir
# is a symbolic link referring to an existing directory", so a symlink-to-directory at
# <root>/bin/unsloth makes the publish deposit .unsloth.shim.$$ INSIDE the linked
# directory: bytes written outside the portable root, a bin/unsloth that is still not a
# launcher, and an exit 0 saying it is. `mv -T` states it directly and is GNU-only, and
# BSD mv has no equivalent, so the portable answer is to refuse.
#
# The directory guard above the block deliberately lets a symlink through, because the
# normal-mode branch replaces one with `ln -sfn`. What keeps the portable branch safe is
# that it moves whatever is on the path aside FIRST, which leaves the path empty. That
# only holds while the move is checked: a rename this run could not perform (a sticky
# <root>/bin holding an entry owned by somebody else is the reachable one) used to be
# discarded, and the publish then followed the link out of the root.
#
# The denial below is an `mv` on PATH that refuses exactly the rename a sticky directory
# refuses -- removing the bin/unsloth entry -- and passes every other rename, the publish
# included, to the real mv. A sticky directory owned by a second account cannot be built
# by a test that runs as one user.
#
# Case [2] and case [3] are the pin the refusal must not swallow: a first install, and a
# reinstall replacing this installer's own wrapper on the same path, both still publish.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# Same lift tests/sh/test_install_portable_shim_conversion.sh uses.
blockSHIM="$(awk '
  /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ { if (!seen) grab = 1 }
  /^# why: -sfn is atomic/ { if (grab) exit }
  grab { print }
  /_shim_tmp=/ { seen = 1 }
' "$INSTALL")
fi"

echo
echo "[1] the lift really picked up the shim block, and the refusal is inside it"
# The opening anchor is NOT unique -- the closing summary opens the same way -- so the
# lift leans on the `# why: -sfn` terminator, which must be. A second one would end the
# block early and every case below would test a truncated snippet.
check "1 the terminator anchor is unique" 1 \
    "$(grep -c '^# why: -sfn is atomic' "$INSTALL")"
check "1 the opening anchor is the one the awk seen-flag expects" 2 \
    "$(grep -c '^if \[ "\$_PORTABLE_MODE" = true \]; then$' "$INSTALL")"
case "$blockSHIM" in *'_shim_tmp='*) : ;; *) echo "FAIL: shim block extraction broke"; exit 1 ;; esac
case "$blockSHIM" in
    *'mv -f "$_shim_tmp" "$_shim_path"'*) : ;;
    *) echo "FAIL: the lift lost the publishing rename"; exit 1 ;;
esac
# The closing summary opens with the same line; landing on it would lift the wrong block.
case "$blockSHIM" in
    *'portable install; everything lives in:'*)
        echo "FAIL: the lift ran on into the closing summary"; exit 1 ;;
    *) : ;;
esac
# A guard defined outside the lifted range would exit 127 here, read as false inside the
# `if`, and go silently inert while cases [2] and [3] stayed green.
case "$blockSHIM" in
    *'could not move the existing $_shim_path aside'*) : ;;
    *) echo "FAIL: the publish-target refusal is not inside the lifted block"; exit 1 ;;
esac

T="$(mktemp -d)"
trap 'chmod -R u+rwX "$T" 2>/dev/null; rm -rf "$T"' EXIT
mkdir -p "$T/home" "$T/fakebin"

# An mv that models a sticky <root>/bin: renaming AWAY the bin/unsloth entry is denied,
# everything else is the real mv. Written once, aimed by $DENY_SRC at run time.
cat > "$T/fakebin/mv" <<'DENY_EOF'
#!/bin/sh
_src=""
for _a in "$@"; do
    case "$_a" in (-*) continue ;; esac
    [ -z "$_src" ] && _src="$_a"
done
if [ -n "${DENY_SRC:-}" ] && [ "$_src" = "$DENY_SRC" ]; then
    echo "mv: cannot move '$_src': Operation not permitted" >&2
    exit 1
fi
for _real in /usr/bin/mv /bin/mv; do
    [ -x "$_real" ] && exec "$_real" "$@"
done
echo "mv: no real mv found" >&2
exit 1
DENY_EOF
chmod +x "$T/fakebin/mv"

new_root() { # -> a fresh portable root with a venv entry point and an empty bin
    _nr="$(mktemp -d "$T/root.XXXXXX")"
    mkdir -p "$_nr/bin" "$_nr/unsloth_studio/bin" "$_nr/outside"
    printf '#!/bin/sh\nprintf "%%s\\n" "${UNSLOTH_PORTABLE:-unset}"\n' \
        > "$_nr/unsloth_studio/bin/unsloth"
    chmod +x "$_nr/unsloth_studio/bin/unsloth"
    printf '%s' "$_nr"
}

run() { # root [deny]
    _rr="$1"
    _rpath="$PATH"
    [ "${2:-}" = deny ] && _rpath="$T/fakebin:$PATH"
    env PATH="$_rpath" HOME="$T/home" DENY_SRC="${2:+$_rr/bin/unsloth}" \
        _PORTABLE_MODE=true UNSLOTH_ROOT="$_rr" STUDIO_HOME="$_rr" \
        VENV_DIR="$_rr/unsloth_studio" _LOCAL_BIN="$_rr/bin" \
        _shim_path="$_rr/bin/unsloth" \
        bash -c "C_WARN=''; substep() { printf '  . %s\n' \"\$1\"; }; $blockSHIM" \
        > "$T/out" 2>"$T/err"
    printf '%s' "$?"
}

state() { # root  -> what sits at bin/unsloth
    if [ -L "$1/bin/unsloth" ]; then printf symlink
    elif [ -f "$1/bin/unsloth" ]; then printf file
    else printf gone; fi
}
escaped() { # root -> how many temp shims landed outside the root
    set -- "$1"/outside/.unsloth.shim.*
    if [ -e "$1" ]; then printf '%s' "$#"; else printf 0; fi
}
strays() { # root -> how many temp shims were left inside bin
    set -- "$1"/bin/.unsloth.shim.*
    if [ -e "$1" ]; then printf '%s' "$#"; else printf 0; fi
}
backups() { # root -> how many moved-aside copies are held for the rollback
    set -- "$1"/bin/.unsloth-portable-shim.*
    if [ -e "$1" ]; then printf '%s' "$#"; else printf 0; fi
}
said() { # substring
    if grep -qF "$1" "$T/out"; then printf yes; else printf no; fi
}
is_portable_shim() { # path
    if grep -qxF "export UNSLOTH_PORTABLE=1" "$1" 2>/dev/null; then printf yes; else printf no; fi
}

echo
echo "[2] a first install publishes onto an empty path"
R2="$(new_root)"
check "2 exits 0" 0 "$(run "$R2")"
check "2 leaves a wrapper on the path" file "$(state "$R2")"
check "2 that is our portable shim" yes "$(is_portable_shim "$R2/bin/unsloth")"
check "2 and says so" yes "$(said "portable shim at")"
check "2 with no temp file left behind" 0 "$(strays "$R2")"

echo
echo "[3] a reinstall replaces its OWN wrapper on the same path"
# The legitimate self-replacement every portable reinstall performs. If the refusal ever
# widens into "anything on the path is fatal", this is what breaks.
R3="$(new_root)"
check "3 the first run publishes" 0 "$(run "$R3")"
_before="$(cat "$R3/bin/unsloth")"
check "3 the reinstall exits 0" 0 "$(run "$R3")"
check "3 a wrapper is still on the path" file "$(state "$R3")"
check "3 it is our portable shim" yes "$(is_portable_shim "$R3/bin/unsloth")"
check "3 byte-identical to the one it replaced" "$_before" "$(cat "$R3/bin/unsloth")"
check "3 and it moved the old one aside for the rollback" 1 "$(backups "$R3")"
check "3 with nothing outside the root" 0 "$(escaped "$R3")"

echo
echo "[4] a symlink to a directory is moved aside, not published through"
# The directory guard lets a symlink through on purpose. While the move aside works,
# the path is empty by publish time and the link is preserved for the rollback.
R4="$(new_root)"
ln -s "$R4/outside" "$R4/bin/unsloth"
check "4 exits 0" 0 "$(run "$R4")"
check "4 the wrapper replaced the link" file "$(state "$R4")"
check "4 the link was kept for the rollback" 1 "$(backups "$R4")"
check "4 and nothing was written through it" 0 "$(escaped "$R4")"

echo
echo "[5] a symlink to a directory that CANNOT be moved aside is refused"
R5="$(new_root)"
ln -s "$R5/outside" "$R5/bin/unsloth"
check "5 exits nonzero" 1 "$(run "$R5" deny)"
check "5 nothing was written outside the root" 0 "$(escaped "$R5")"
check "5 no temp file was left in bin" 0 "$(strays "$R5")"
check "5 the user's link is untouched" symlink "$(state "$R5")"
check "5 and the run never claimed a shim" no "$(said "portable shim at")"
check "5 it named the path instead" yes \
    "$(grep -qF "could not move the existing $R5/bin/unsloth aside" "$T/err" && printf yes || printf no)"

echo
echo "[6] the same refusal for an ordinary file that cannot be moved aside"
# Not symlink-specific: a discarded backup failure means the rollback has no copy of
# whatever the publish is about to destroy, whichever shape it was.
R6="$(new_root)"
printf '#!/bin/sh\necho mine\n' > "$R6/bin/unsloth"
chmod +x "$R6/bin/unsloth"
check "6 exits nonzero" 1 "$(run "$R6" deny)"
check "6 the user's file is untouched" file "$(state "$R6")"
check "6 with its own contents" "mine" "$("$R6/bin/unsloth")"
check "6 and no temp file was left in bin" 0 "$(strays "$R6")"

echo
if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
