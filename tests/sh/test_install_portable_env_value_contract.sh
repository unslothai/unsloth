#!/usr/bin/env bash
# Regression test: UNSLOTH_PORTABLE means the same thing to the installer and the runtime.
#
# storage_roots.portable_mode() reads every nonblank value except 0/false/off/no as ON, while
# install.sh used a truthy allowlist (1/true/yes/on) and read everything else as OFF. So
# UNSLOTH_PORTABLE=enabled -- or a typo like `flase` -- installed the normal roots with no
# portable marker while the backend running in that same environment considered itself
# portable and redirected the HF caches, TORCH_HOME and the projects root, reverting again on
# the next launch that carried no such variable. install.ps1 already fails the install for
# exactly those values.
#
# Neither silent reading is safe on its own: guessing ON relocates the tree of a user who
# spelled "off" as `disabled`, guessing OFF is the split above. So the installer refuses a
# value that is on neither list. The Python probe at the end pins WHY: it fails if the runtime
# ever stops disagreeing, which is the day this guard could be relaxed.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
PS1_FILE="$ROOT/install.ps1"
BACKEND="$ROOT/studio/backend"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blockA="$(awk '
    /^# ── Parse flags ──$/ {grab = 1}
    grab {print}
    /^        _UNSLOTH_ROOT="\$HOME\/\.unsloth"$/ {seen = 1}
    seen && /^fi$/ {exit}
' "$INSTALL")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockA" in *"--portable) _PORTABLE_MODE=true ;;"*) : ;; *) echo "FAIL: blockA extraction broke"; exit 1 ;; esac
case "$blockA" in *'UNSLOTH_PORTABLE'*) : ;; *) echo "FAIL: blockA lost the UNSLOTH_PORTABLE branch"; exit 1 ;; esac

# set -e is on: a bare `[ cond ] && action` as the last statement of a block would kill the run.
SNIP='set -e
'"$blockA"'
printf "reached|%s|%s\n" "$_PORTABLE_MODE" "$_UNSLOTH_ROOT"'

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
# mktemp per case, not a counter: an increment inside `$( )` happens in a subshell.
new_home() { mktemp -d "$T/home.XXXXXX"; }

# Returns "rc|portable" for one value of UNSLOTH_PORTABLE.
parse() { # value-or-UNSET
    _h="$(new_home)"
    if [ "$1" = UNSET ]; then
        env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" \
            bash -c "$SNIP" _ > "$T/out" 2>"$T/err"
    else
        env -i HOME="$_h" PATH="$PATH" USER="${USER:-tester}" UNSLOTH_PORTABLE="$1" \
            bash -c "$SNIP" _ > "$T/out" 2>"$T/err"
    fi
    _rc=$?
    _mode=$(sed -n 's/^reached|\([^|]*\)|.*$/\1/p' "$T/out")
    printf '%s|%s' "$_rc" "${_mode:-none}"
}

# ── 1. The values the installer accepts as ON. Case and surrounding whitespace are stripped
# the way storage_roots.portable_mode() and install.ps1's guard strip them.
for v in 1 true yes on True " TRUE " Yes " ON " " true "; do
    check "UNSLOTH_PORTABLE='$v' installs portable" "0|true" "$(parse "$v")"
done

# ── 2. The values it accepts as OFF. These are exactly portable_mode()'s off-list, so a
# normal install with any of them set is normal at both ends.
for v in 0 false off no FALSE Off " no " " 0 " ""; do
    check "UNSLOTH_PORTABLE='$v' installs normally" "0|false" "$(parse "$v")"
done
check "an unset UNSLOTH_PORTABLE installs normally" "0|false" "$(parse UNSET)"

# ── 3. Everything else is refused rather than guessed, and the refusal names the variable
# and both spellings so a piped install can be fixed from the message alone.
for v in enabled flase 2 bogus disabled ENABLED " enabled " "true false" "-1" "yes please"; do
    check "UNSLOTH_PORTABLE='$v' is refused" "1|none" "$(parse "$v")"
done
# Re-run one refused value so $T/out and $T/err hold its output, then read the message.
parse enabled > /dev/null
check "the error names the variable" 1 "$(grep -c 'UNSLOTH_PORTABLE' "$T/err")"
case "$(cat "$T/err")" in
    *"1, true, yes or on"*) check "the error names the ON spellings" ok ok ;;
    *) check "the error names the ON spellings" ok "missing: $(cat "$T/err")" ;;
esac
case "$(cat "$T/err")" in
    *"0, false, off, no"*) check "the error names the OFF spellings" ok ok ;;
    *) check "the error names the OFF spellings" ok "missing: $(cat "$T/err")" ;;
esac
check "the refusal goes to stderr, not stdout" "" "$(cat "$T/out")"

# ── 4. Cross-file: install.ps1 refuses the same set. Its guard is an off-list, so the values
# it lets through must be exactly the ones accepted as OFF above.
# [^)]*, not .*: the trailing `)) {` on that line makes a greedy match swallow the paren.
_ps1_off="$(sed -n 's/.*\$env:UNSLOTH_PORTABLE\.Trim() -notin @(\([^)]*\)).*/\1/p' "$PS1_FILE" \
    | head -n1 | tr -d '" ' | tr ',' '\n' | sort | tr '\n' ' ')"
check "install.ps1 off-list found" "0 false no off " "$_ps1_off"
_sh_off="$(printf '%s\n' "$blockA" | sed -n "s/^    ''|\(.*\)) ;;$/\1/p" | head -n1 \
    | tr '|' '\n' | sort | tr '\n' ' ')"
check "install.sh accepts the same off-list" "$_ps1_off" "$_sh_off"

# ── 5. The runtime half: the values refused above are exactly the ones the backend would
# have read as portable. Delete this section only when portable_mode() stops guessing too.
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
sr._setup_cache_env()
print("__JSON__" + json.dumps({
    "portable": sr.portable_mode(),
    "torch_home": os.environ.get("TORCH_HOME"),
}))
PYEOF
    probe() { # value field
        _h="$(new_home)"
        mkdir -p "$_h/.unsloth/studio/unsloth_studio/bin"
        _pout=$(env -i HOME="$_h" PATH="$PATH" _BACKEND="$BACKEND" UNSLOTH_PORTABLE="$1" \
            python3 "$PROBE" 2>"$T/perr")
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            printf '  FAIL  storage_roots probe produced no output\n%s\n' "$(cat "$T/perr")"
            fails=$((fails + 1))
            printf 'probe-failed'
            return 0
        fi
        printf '%s' "$_pjson" | _FIELD="$2" python3 -c \
            'import json,os,sys; v=json.load(sys.stdin)[os.environ["_FIELD"]]; print("null" if v is None else str(v).lower() if isinstance(v,bool) else v)'
    }
    for v in enabled flase 2 bogus disabled; do
        check "the runtime really reads '$v' as portable" true "$(probe "$v" portable)"
    done
    for v in 0 false off no; do
        check "the runtime reads '$v' as normal" false "$(probe "$v" portable)"
    done
    # Not just a boolean: this is the redirection the installer would not have matched.
    case "$(probe enabled torch_home)" in
        */.unsloth/studio/cache/torch) check "and redirects TORCH_HOME under the studio root" ok ok ;;
        *) check "and redirects TORCH_HOME under the studio root" ok "$(probe enabled torch_home)" ;;
    esac
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
