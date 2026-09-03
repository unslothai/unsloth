#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Regression test: a FLAT root converts in place, in both directions, so whichever launcher
# is on <root>/bin/unsloth is destroyed by the conversion and has to roll back with it.
#
# tests/sh/test_install_portable_shim_conversion.sh covers the NESTED portable -> normal
# case, where the old wrapper is retired and put back on failure. The flat root is the case
# that arm's `case "$_spm_leaf" in */studio)` deliberately skips, because there is nothing to
# retire: the new shim lands on the same path. Both directions are covered here, plus the run
# where the nested and flat retirements fire together, which is why the launcher has two
# rollback slots rather than one.
#
# tests/sh/test_install_portable_shim_conversion.sh covers portable -> normal, where
# the old wrapper is retired and put back on failure. This is the mirror. A user who
# installed with UNSLOTH_STUDIO_HOME=DIR has a symlink at DIR/bin/unsloth, and README
# tells that user to add portable mode to pull the caches in too. --portable over the
# same DIR is a FLAT portable install -- DIR/unsloth_studio already exists -- so
# _LOCAL_BIN resolves to DIR/bin and the shim block renames its wrapper straight over
# that symlink. The shim block runs hundreds of lines BEFORE the setup.sh gate, so a
# conversion that setup.sh failed used to restore the previous environment and drop
# the marker it published while leaving that wrapper in front of the restored install,
# exporting UNSLOTH_HOME and UNSLOTH_PORTABLE=1 on every later launch, with no
# supported way back.
#
# The fixtures and the conversion are install.sh's own blocks, in install.sh's own
# order, so a change to either half fails here rather than going quietly inert.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blk() { awk "$1" "$INSTALL"; }
blockA="$(blk '/^# ── Parse flags ──$/ {grab=1} grab {print} /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen=1} seen && /^fi$/ {exit}')"
blockB="$(blk '/^_resolve_studio_destinations\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockM="$(blk '/^_PORTABLE_MARKER_PATH_1=""$/ {grab=1} grab {print} /^_PORTABLE_SHIM_BACKUP=""$/ {exit}')"
blockE="$(blk '/^_export_portable_roots\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockD="$(blk '/^_clear_stale_portable_marker\(\) \{$/ {grab=1} grab {print} grab && /^\}$/ {exit}')"
blockR="$(blk '/^_VENV_ROLLBACK_DIR=""$/ {grab=1} grab {print} /^trap ._on_install_signal 143. TERM$/ {exit}')"
blockSHIM="$(awk '
  /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ { if (!seen) grab = 1 }
  /^# why: -sfn is atomic/ { if (grab) exit }
  grab { print }
  /_shim_tmp=/ { seen = 1 }
' "$INSTALL")
fi"
# The WHOLE chain, portable branch and the `elif ln -sfn` normal branch together, because
# the flat conversion back to normal runs through the elif. blockSHIM stays extracted above
# only to assert the standalone-extraction rules the other test's fixture generator needs.
blockFULL="$(awk '
  /^# A wrapper, not a symlink: a symlink carries no environment, so uv and$/ {grab=1}
  grab {print}
  grab && /^fi$/ {exit}
' "$INSTALL")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockB" in *'_PORTABLE_FLAT=true'*) : ;; *) echo "FAIL: blockB lost the flat-layout branch"; exit 1 ;; esac
case "$blockM" in *'_PORTABLE_SHIM_BACKUP'*) : ;; *) echo "FAIL: blockM lost the launcher slot"; exit 1 ;; esac
case "$blockE" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockE extraction broke"; exit 1 ;; esac
case "$blockD" in *'.unsloth-portable-root'*) : ;; *) echo "FAIL: blockD extraction broke"; exit 1 ;; esac
case "$blockR" in *'_restore_portable_shim() {'*) : ;; *) echo "FAIL: the launcher restore is defined outside the rollback block"; exit 1 ;; esac
case "$blockSHIM" in *_shim_tmp*) : ;; *) echo "FAIL: shim block extraction broke"; exit 1 ;; esac
# The whole point: the shim block has to preserve what its rename displaces.
case "$blockSHIM" in *'_PORTABLE_SHIM_BACKUP'*) : ;; *) echo "FAIL: the shim block does not preserve what it renames over"; exit 1 ;; esac
# ...and it has to do it inline. tests/sh/test_install_portable_shim_conversion.sh runs
# this block standalone to generate its fixtures, with only substep defined, so a call
# to a helper defined elsewhere dies with `command not found` in that generator. Comments
# stripped first: naming the restore in a comment is how the two halves stay findable.
_shim_code="$(printf '%s\n' "$blockSHIM" | grep -v '^[[:space:]]*#')"
case "$_shim_code" in *'_restore_portable_shim'*) echo "FAIL: the shim block calls the restore helper"; exit 1 ;; *) : ;; esac
case "$_shim_code" in *rollback_substep*) echo "FAIL: the shim block calls rollback_substep, which its standalone extraction does not define"; exit 1 ;; *) : ;; esac
case "$_shim_code" in *'_PORTABLE_SHIM_BACKUP'*) : ;; *) echo "FAIL: the preserve is a comment and nothing else"; exit 1 ;; esac
# One slot for both directions, so they must not be able to fire in the same run.
case "$blockD" in *'if [ "$_PORTABLE_MODE" = true ]; then return 0; fi'*) : ;; *) echo "FAIL: the portable->normal retirement no longer returns early in portable mode"; exit 1 ;; esac
# The restore recognizes a generated wrapper by a line the generator writes. Both halves
# must carry the same literal, or the mirror rollback goes inert while nothing complains.
_GEN_LINE='# Generated by install.sh --portable. Keeps every Unsloth path inside'
case "$blockSHIM" in *"$_GEN_LINE"*) : ;; *) echo "FAIL: the generated wrapper no longer carries the line the restore matches"; exit 1 ;; esac
case "$blockR" in *"$_GEN_LINE"*) : ;; *) echo "FAIL: the restore no longer matches the line the generator writes"; exit 1 ;; esac
# The normal-mode shape at that path is a symlink into the venv. The fixtures below build
# exactly that by hand, so pin the line they are copying, and make sure blockFULL carries it.
grep -qF 'elif ! ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path" 2>/dev/null; then' "$INSTALL" \
    || { echo "FAIL: a normal install no longer symlinks the shim to the venv"; exit 1; }
case "$blockFULL" in *'elif ! ln -sfn'*) : ;; *) echo "FAIL: blockFULL lost the normal-mode branch"; exit 1 ;; esac
case "$blockFULL" in *_shim_tmp*) : ;; *) echo "FAIL: blockFULL lost the portable branch"; exit 1 ;; esac
# blockM is delimited by the LAST slot declaration. If that line is renamed or reordered the
# awk terminator stops matching and every blockM in tests/sh silently swallows the rest of
# install.sh, while the substring self-checks around it still pass.
case "$blockM" in *_export_portable_roots*) echo "FAIL: blockM ran past the slot declarations"; exit 1 ;; *) : ;; esac
# Two launcher pairs, because the nested and flat retirements can both fire in one run.
case "$blockM" in *'_PORTABLE_FLAT_SHIM_BACKUP'*) : ;; *) echo "FAIL: blockM lost the flat launcher slot"; exit 1 ;; esac
case "$blockD" in *'_PORTABLE_FLAT_SHIM_PATH'*) : ;; *) echo "FAIL: the flat retirement no longer preserves its launcher"; exit 1 ;; esac
# ...and the flat save has to be behavioural, not "is there a file called unsloth".
case "$blockD" in *'grep -qxF "export UNSLOTH_PORTABLE=1" "$_spm_flat_shim"'*) : ;; *) echo "FAIL: the flat save lost its ownership probe"; exit 1 ;; esac
# And pin the ordering that makes the rollback necessary in the first place: the wrapper is
# renamed into place well before the run can still exit on setup.sh's status.
_shim_line=$(grep -n '^    _shim_tmp="\$_LOCAL_BIN/\.unsloth\.shim\.\$\$"$' "$INSTALL" | head -n1 | cut -d: -f1)
_setup_gate=$(grep -n '^    exit "\$_SETUP_EXIT"$' "$INSTALL" | head -n1 | cut -d: -f1)
[ -n "$_shim_line" ] || { echo "FAIL: could not locate the portable shim block"; exit 1; }
[ -n "$_setup_gate" ] || { echo "FAIL: could not locate the setup.sh failure gate"; exit 1; }
[ "$_shim_line" -lt "$_setup_gate" ] || { echo "FAIL: the shim now follows the setup.sh gate; this test no longer describes install.sh"; exit 1; }

# install.sh's own order: destinations, marker, venv replacement, setup.sh, THEN the shim,
# and only after that the gate that exits on setup.sh's status.
SNIP='set -e
C_WARN=""
substep() { printf "  . %s\n" "$1"; }
rollback_substep() { printf "  R %s\n" "$1"; }
'"$blockA"'
'"$blockB"'
_resolve_studio_destinations
UNSLOTH_ROOT="${UNSLOTH_ROOT:-}"
'"$blockM"'
'"$blockE"'
_export_portable_roots
'"$blockD"'
_clear_stale_portable_marker
_UNSLOTH_LOGIN_PATH="$PATH"
VENV_DIR="$STUDIO_HOME/unsloth_studio"
'"$blockR"'
# ── everything above is install.sh verbatim ──
printf "reached|%s|%s\n" "$STUDIO_HOME" "$_LOCAL_BIN"
if [ -d "$VENV_DIR" ]; then _start_studio_venv_replacement "$VENV_DIR"; fi
mkdir -p "$VENV_DIR/bin"
printf "new\n" > "$VENV_DIR/tag"
cat > "$VENV_DIR/bin/unsloth" <<"VENVEOF"
#!/bin/sh
printf "%s\n" "${UNSLOTH_PORTABLE:-unset}"
VENVEOF
chmod +x "$VENV_DIR/bin/unsloth"
_SETUP_EXIT=${SETUP_EXIT:-7}
if [ "$_SETUP_EXIT" -eq 0 ]; then _commit_studio_venv_replacement; fi
mkdir -p "$_LOCAL_BIN"
_shim_path="$_LOCAL_BIN/unsloth"
'"$blockFULL"'
if [ "${FAIL_MODE:-}" = signal ]; then kill -INT $$; sleep 5; fi
if [ "$_SETUP_EXIT" -ne 0 ]; then exit "$_SETUP_EXIT"; fi
_commit_portable_marker'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home"
new_root() { mktemp -d "$T/root.XXXXXX"; }

# A flat custom NORMAL install: venv at <dir>/unsloth_studio, the shim at <dir>/bin the
# way `ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"` writes it, and no marker.
build_normal() { # dir
    _bd="$1"
    mkdir -p "$_bd/unsloth_studio/bin" "$_bd/bin"
    printf 'normal\n' > "$_bd/unsloth_studio/tag"
    cat > "$_bd/unsloth_studio/bin/unsloth" <<"VEOF"
#!/bin/sh
printf "%s\n" "${UNSLOTH_PORTABLE:-unset}"
VEOF
    chmod +x "$_bd/unsloth_studio/bin/unsloth"
    ln -sfn "$_bd/unsloth_studio/bin/unsloth" "$_bd/bin/unsloth"
}

convert() { # dir [env assignments...]
    _cd="$1"; shift
    # shellcheck disable=SC2086
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
        UNSLOTH_STUDIO_HOME="$_cd" "$@" bash -c "$SNIP" _ --portable \
        > "$T/out" 2>"$T/err"
    _crc=$?
    if ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet never reached the install (rc=%s)\n%s\n%s\n' \
            "$_crc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
    printf '%s' "$_crc"
}

shim_state() { # dir
    if [ -L "$1/bin/unsloth" ]; then printf symlink
    elif [ -f "$1/bin/unsloth" ]; then printf wrapper
    elif [ -e "$1/bin/unsloth" ]; then printf other
    else printf gone; fi
}
shim_target() { # dir
    ls -l "$1/bin/unsloth" 2>/dev/null | sed 's/.* -> //'
}
launch() { # dir -- what the user actually gets when they run the documented command
    "$1/bin/unsloth" studio 2>/dev/null || printf 'broken'
}
marker_state() { # dir
    if [ -f "$1/.unsloth-portable-root" ]; then printf present; else printf gone; fi
}
venv_tag() { # dir
    cat "$1/unsloth_studio/tag" 2>/dev/null || printf '(none)'
}
convert_normal() { # dir [env assignments...] -- a NORMAL reinstall over the same tree
    _cn="$1"; shift
    # shellcheck disable=SC2086
    env -i HOME="$T/home" PATH="$PATH" USER="${USER:-tester}" \
        UNSLOTH_STUDIO_HOME="$_cn" "$@" bash -c "$SNIP" _ \
        > "$T/out" 2>"$T/err"
    _nrc=$?
    if ! grep -q '^reached|' "$T/out"; then
        printf '  FAIL  installer snippet never reached the install (rc=%s)\n%s\n%s\n' \
            "$_nrc" "$(cat "$T/out")" "$(cat "$T/err")"
        fails=$((fails + 1))
    fi
    printf '%s' "$_nrc"
}

strays() { # dir
    set -- "$1"/bin/.unsloth-portable-shim.* "$1"/bin/.unsloth-flat-shim.* "$1"/bin/.unsloth.shim.*
    _n=0
    for _s in "$@"; do [ -e "$_s" ] || [ -L "$_s" ] || continue; _n=$((_n + 1)); done
    printf '%s' "$_n"
}
# Without this the rollback cases also pass on an installer that never touched the
# launcher: "restored" is vacuously true when nothing was ever moved.
said() { # substring
    if grep -qF "$1" "$T/out"; then printf yes; else printf no; fi
}

echo
echo "[0] the fixture really is the flat conversion this is about"
D0="$(new_root)"; build_normal "$D0"
check "0 the normal install launches uncontained" unset "$(launch "$D0")"
rc="$(convert "$D0" SETUP_EXIT=0)"
check "0 the conversion resolves the flat layout onto the same bin" "$D0/bin" \
    "$(sed -n 's/^reached|.*|//p' "$T/out")"

echo
echo "[1] a successful conversion replaces the symlink with the wrapper"
check "1 the conversion exits 0" 0 "$rc"
check "1 the shim is now a generated wrapper" wrapper "$(shim_state "$D0")"
check "1 and launches contained" 1 "$(launch "$D0")"
check "1 the marker is published" present "$(marker_state "$D0")"
check "1 and no copy of the old launcher is left behind" 0 "$(strays "$D0")"

echo
echo "[2] a failed conversion puts the symlink back"
D1="$(new_root)"; build_normal "$D1"
before_target="$(shim_target "$D1")"
rc="$(convert "$D1")"
check "2 the failed conversion exits nonzero" 7 "$rc"
check "2 the previous environment is back" normal "$(venv_tag "$D1")"
check "2 the marker it published is gone" gone "$(marker_state "$D1")"
check "2 the shim is a symlink again" symlink "$(shim_state "$D1")"
check "2 pointing where it did before" "$before_target" "$(shim_target "$D1")"
check "2 and the documented command launches uncontained again" unset "$(launch "$D1")"
check "2 leaving no copy behind" 0 "$(strays "$D1")"
check "2 and the run reported the restore" yes "$(said "restored the previous launcher at")"

echo
echo "[3] an interrupted conversion restores it too"
D2="$(new_root)"; build_normal "$D2"
rc="$(convert "$D2" FAIL_MODE=signal SETUP_EXIT=0)"
check "3 the interrupted conversion exits 130" 130 "$rc"
check "3 the shim is a symlink again" symlink "$(shim_state "$D2")"
check "3 and launches uncontained" unset "$(launch "$D2")"
check "3 no copy is left behind" 0 "$(strays "$D2")"
check "3 and the run reported the restore" yes "$(said "restored the previous launcher at")"

echo
echo "[4] a fresh portable root has nothing to displace"
D3="$(new_root)"
mkdir -p "$D3/unsloth_studio"
rc="$(convert "$D3")"
check "4 the failed fresh install exits nonzero" 7 "$rc"
# The wrapper the run wrote stays: there was no launcher on that path to put back, and the
# half-built root it sits in is the run's own. What must not happen is a phantom restore.
check "4 the wrapper it wrote is what is there" wrapper "$(shim_state "$D3")"
check "4 and nothing is claimed about restoring one" no "$(said "restored the previous launcher at")"
check "4 no copy is left behind" 0 "$(strays "$D3")"

echo
echo "[5] a failed portable REINSTALL restores the previous wrapper byte for byte"
# The same rename, over a wrapper this time: the root was already portable, and the
# failure has to leave the launcher its summary printed exactly as it was.
D4="$(new_root)"; build_normal "$D4"
rc="$(convert "$D4" SETUP_EXIT=0)"
check "5 the first conversion succeeded" 0 "$rc"
before_bytes="$(cat "$D4/bin/unsloth")"
printf '%s\n' "$D4" > "$D4/.unsloth-portable-root"
rc="$(convert "$D4")"
check "5 the failed reinstall exits nonzero" 7 "$rc"
check "5 the wrapper is still there" wrapper "$(shim_state "$D4")"
check "5 with its original contents" "$before_bytes" "$(cat "$D4/bin/unsloth")"
check "5 still executable" yes "$([ -x "$D4/bin/unsloth" ] && printf yes || printf no)"
check "5 and still launches contained" 1 "$(launch "$D4")"
check "5 the marker that was already there survives" present "$(marker_state "$D4")"
check "5 no copy is left behind" 0 "$(strays "$D4")"

echo
echo "[6] the restore refuses a launcher it does not recognize"
# A concurrent run that published some other shape at the path owns it now, exactly as
# a marker that is back already is left alone.
# The conversion snippet has no hook for a second writer, so drive the real restore
# helper directly against a path something else has taken over.
D6="$(new_root)"; build_normal "$D6"
{
    printf '%s\n' 'set -e'
    printf '%s\n' 'rollback_substep() { printf "  R %s\n" "$1"; }'
    printf '%s\n' 'substep() { :; }'
    printf '%s\n' 'C_WARN=""'
    printf "STUDIO_HOME='%s'\n" "$D6"
    printf "VENV_DIR='%s/unsloth_studio'\n" "$D6"
    printf '%s\n' "$blockR"
    printf "_PORTABLE_SHIM_PATH='%s/bin/unsloth'\n" "$D6"
    printf "_PORTABLE_SHIM_BACKUP='%s/bin/.unsloth-portable-shim.test'\n" "$D6"
    printf '%s\n' 'mv "$_PORTABLE_SHIM_PATH" "$_PORTABLE_SHIM_BACKUP"'
    printf '%s\n' 'printf "#!/bin/sh\necho hijacked\n" > "$_PORTABLE_SHIM_PATH"'
    printf '%s\n' 'chmod +x "$_PORTABLE_SHIM_PATH"'
    printf '%s\n' '_restore_portable_shim'
} > "$T/refuse.sh"
sh "$T/refuse.sh" > "$T/refuse.out" 2>&1
check "6 an unrecognized launcher is left alone" "#!/bin/sh
echo hijacked" "$(cat "$D6/bin/unsloth")"
check "6 and the run claims no restore" no \
    "$(grep -qF 'restored the previous launcher at' "$T/refuse.out" && printf yes || printf no)"
check "6 the copy is still on disk rather than silently dropped" yes \
    "$([ -e "$D6/bin/.unsloth-portable-shim.test" ] && printf yes || printf no)"

echo
echo "[7] the same path, converting the OTHER way: flat portable -> normal"
# <root>/bin IS the bin a normal reinstall writes into, so `ln -sfn` lands on the portable
# wrapper itself. Nothing to retire (the nested arm's case does not match a flat root), but
# the marker DOES roll back, so without a save the restored tree reads as portable through a
# launcher that exports none of it.
D7="$(new_root)"; build_normal "$D7"
rc="$(convert "$D7" SETUP_EXIT=0)"
check "7 the tree starts as a real flat portable install" wrapper "$(shim_state "$D7")"
check "7 with its marker" present "$(marker_state "$D7")"
printf 'portable\n' > "$D7/unsloth_studio/tag"
wrapper_bytes="$(cat "$D7/bin/unsloth")"
rc="$(convert_normal "$D7")"
check "7 the failed normal reinstall exits nonzero" 7 "$rc"
check "7 it really moved the launcher aside" yes "$(said "replacing the portable launcher at")"
check "7 the portable environment is back" portable "$(venv_tag "$D7")"
check "7 and its marker is restored" present "$(marker_state "$D7")"
check "7 the portable launcher is back" wrapper "$(shim_state "$D7")"
check "7 with its original contents" "$wrapper_bytes" "$(cat "$D7/bin/unsloth")"
check "7 still executable" yes "$([ -x "$D7/bin/unsloth" ] && printf yes || printf no)"
check "7 and still launches contained" 1 "$(launch "$D7")"
check "7 leaving no copy behind" 0 "$(strays "$D7")"
check "7 and the run reported the restore" yes "$(said "restored the previous launcher at")"

echo
echo "[8] a SUCCESSFUL flat conversion back to normal keeps the conversion"
D8="$(new_root)"; build_normal "$D8"
rc="$(convert "$D8" SETUP_EXIT=0)"
rc="$(convert_normal "$D8" SETUP_EXIT=0)"
check "8 the conversion exits 0" 0 "$rc"
check "8 the marker stays removed" gone "$(marker_state "$D8")"
check "8 the launcher is the normal symlink now" symlink "$(shim_state "$D8")"
check "8 and launches uncontained" unset "$(launch "$D8")"
check "8 with the saved copy dropped" 0 "$(strays "$D8")"

echo
echo "[9] the save cannot fire on an ordinary normal install"
# No portable marker, so _clear_stale_portable_marker never reaches the save at all. This is
# the whole safety argument for putting it there rather than beside ln -sfn.
D9="$(new_root)"; build_normal "$D9"
rc="$(convert_normal "$D9")"
check "9 the failed ordinary reinstall exits nonzero" 7 "$rc"
check "9 and nothing was moved aside" no "$(said "replacing the portable launcher at")"
check "9 nothing was restored either" no "$(said "restored the previous launcher at")"
check "9 no copy is left behind" 0 "$(strays "$D9")"

# A launcher that is not ours keeps its place: same tree, same path, no ownership.
DA="$(new_root)"; build_normal "$DA"
rc="$(convert "$DA" SETUP_EXIT=0)"
printf '#!/bin/sh\nexec "%s/unsloth_studio/bin/unsloth" "$@"\n' "$DA" > "$DA/bin/unsloth"
chmod +x "$DA/bin/unsloth"
own_bytes="$(cat "$DA/bin/unsloth")"
rc="$(convert_normal "$DA")"
check "9 a launcher without the portable exports is not moved" no "$(said "replacing the portable launcher at")"
check "9 and a failed reinstall leaves whatever ln -sfn wrote" symlink "$(shim_state "$DA")"
check "9 no copy is left behind" 0 "$(strays "$DA")"

# A launcher for a DIFFERENT install's venv is not ours either.
DB="$(new_root)"; OTHER="$(new_root)"; build_normal "$DB"
rc="$(convert "$DB" SETUP_EXIT=0)"
sed "s|$DB/unsloth_studio|$OTHER/unsloth_studio|" "$DB/bin/unsloth" > "$DB/bin/unsloth.new"
mv "$DB/bin/unsloth.new" "$DB/bin/unsloth"; chmod +x "$DB/bin/unsloth"
rc="$(convert_normal "$DB")"
check "9 another install's launcher is not moved" no "$(said "replacing the portable launcher at")"

echo "[10] both retirements in ONE run, which is why the flat one has its own slot"
# <X> is a nested portable root and <X>/studio is a flat one, the same double-marker shape
# tests/sh/test_install_portable_marker_rollback.sh calls D3. A normal reinstall of
# <X>/studio removes BOTH markers and displaces BOTH launchers: <X>/bin/unsloth is retired
# by the nested arm, <X>/studio/bin/unsloth is renamed over by ln -sfn. Two launchers, so
# two slots; one pair would silently drop whichever went second.
gen_wrapper() { # local_bin root studio_home venv_dir
    mkdir -p "$1"
    env _PORTABLE_MODE=true UNSLOTH_ROOT="$2" STUDIO_HOME="$3" VENV_DIR="$4" \
        _LOCAL_BIN="$1" _shim_path="$1/unsloth" HOME="$T/home" \
        bash -c "substep(){ :; }; $blockSHIM" >/dev/null 2>&1
    [ -f "$1/unsloth" ] || { echo "FAIL: fixture launcher was not generated in $1"; exit 1; }
}
X="$(new_root)"
mkdir -p "$X/studio/unsloth_studio/bin"
printf 'portable\n' > "$X/studio/unsloth_studio/tag"
cat > "$X/studio/unsloth_studio/bin/unsloth" <<"VEOF"
#!/bin/sh
printf "%s\n" "${UNSLOTH_PORTABLE:-unset}"
VEOF
chmod +x "$X/studio/unsloth_studio/bin/unsloth"
printf '%s\n' "$X" > "$X/.unsloth-portable-root"
printf '%s\n' "$X/studio" > "$X/studio/.unsloth-portable-root"
gen_wrapper "$X/bin"        "$X"        "$X/studio" "$X/studio/unsloth_studio"
gen_wrapper "$X/studio/bin" "$X/studio" "$X/studio/unsloth_studio" "$X/studio/unsloth_studio"
nested_bytes="$(cat "$X/bin/unsloth")"
flat_bytes="$(cat "$X/studio/bin/unsloth")"
check "10 both launchers start as portable wrappers" "wrapper wrapper" \
    "$(shim_state "$X") $(shim_state "$X/studio")"
check "10 and both launch contained" "1 1" "$(launch "$X") $(launch "$X/studio")"
rc="$(convert_normal "$X/studio")"
check "10 the failed reinstall exits nonzero" 7 "$rc"
check "10 the nested launcher was retired" yes "$(said "removed the portable launcher at")"
check "10 and the flat one was moved aside" yes "$(said "replacing the portable launcher at")"
check "10 both markers are restored" "present present" \
    "$(marker_state "$X") $(marker_state "$X/studio")"
check "10 the nested launcher is back, byte for byte" "$nested_bytes" "$(cat "$X/bin/unsloth")"
check "10 the flat launcher is back, byte for byte" "$flat_bytes" "$(cat "$X/studio/bin/unsloth")"
check "10 both launch contained again" "1 1" "$(launch "$X") $(launch "$X/studio")"
check "10 no copies left in either bin" "0 0" "$(strays "$X") $(strays "$X/studio")"

echo
if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
