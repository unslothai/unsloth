#!/usr/bin/env bash
# Regression test: UNSLOTH_PORTABLE means the same thing to the installer and the runtime.
#
# storage_roots.portable_mode() used to read every nonblank value except 0/false/off/no as ON,
# while install.sh used a truthy allowlist (1/true/yes/on) and read everything else as OFF. So
# UNSLOTH_PORTABLE=enabled -- or a typo like `flase` -- installed the normal roots with no
# portable marker while the backend running in that same environment considered itself
# portable and redirected the HF caches, TORCH_HOME and the projects root, reverting again on
# the next launch that carried no such variable. install.ps1 already failed the install for
# exactly those values.
#
# Neither silent reading is safe on its own: guessing ON relocates the tree of a user who
# spelled "off" as `disabled`, guessing OFF is the split above. So the installer refuses a
# value on neither list, and the runtime now reads through the same two allowlists and treats
# anything else as no opinion. Both halves are tested here because the bug was the DISAGREEMENT
# between them, not either one alone: sections 1-4 pin the installer's three answers, and the
# Python probe in section 5 pins that the runtime gives the same three for the same values.
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

# ── 5. The runtime half: the backend reads the same three answers out of the same value.
# The refused set and the runtime's no-opinion set have to stay the SAME set, so this section
# drives the real resolver over the very values sections 1-3 drove install.sh over.
if command -v python3 > /dev/null 2>&1; then
    PROBE="$T/probe.py"
    cat > "$PROBE" <<'PYEOF'
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
sr._setup_cache_env()
print("__JSON__" + json.dumps({
    "portable": sr.portable_mode(),
    # The portable-only redirections: set from _portable_cache_defaults and from
    # nowhere else, so "null" here is proof the tree was left alone rather than
    # proof the variable is unpopular.
    "torch_home": os.environ.get("TORCH_HOME"),
    "projects_home": os.environ.get("UNSLOTH_STUDIO_PROJECTS_HOME"),
    "datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
}))
PYEOF
    # A genuine portable tree: a master root, which is what actually makes an install
    # portable. Kept next to the normal-tree HOMEs so both shapes see the same value.
    PROOT="$T/portable root"
    mkdir -p "$PROOT/studio"

    # One python3 launch per environment, read out field by field: a probe costs an
    # interpreter start plus the backend import, and this section needs several fields
    # from most of them.
    probe_json() { # value [master-root]
        _h="$(new_home)"
        mkdir -p "$_h/.unsloth/studio/unsloth_studio/bin"
        if [ -n "${2:-}" ]; then
            _pout=$(env -i HOME="$_h" PATH="$PATH" _BACKEND="$BACKEND" \
                UNSLOTH_PORTABLE="$1" UNSLOTH_HOME="$2" python3 "$PROBE" 2>"$T/perr")
        else
            _pout=$(env -i HOME="$_h" PATH="$PATH" _BACKEND="$BACKEND" \
                UNSLOTH_PORTABLE="$1" python3 "$PROBE" 2>"$T/perr")
        fi
        # Kept for 5f: the backend's logger writes to stdout, so the notice arrives
        # interleaved with the payload line rather than on perr.
        printf '%s\n' "$_pout" > "$T/pout"
        _pjson=$(printf '%s\n' "$_pout" | sed -n 's/^__JSON__//p')
        if [ -z "$_pjson" ]; then
            # fails++ here would land in the $( ) subshell, so report and hand back a
            # sentinel that cannot equal any expected value.
            printf '  FAIL  storage_roots probe produced no output for UNSLOTH_PORTABLE=%s\n%s\n' \
                "$1" "$(cat "$T/perr")" >&2
            printf 'probe-failed'
            return 0
        fi
        printf '%s' "$_pjson"
    }
    field() { # json field
        case "$1" in probe-failed) printf 'probe-failed'; return 0 ;; esac
        printf '%s' "$1" | _FIELD="$2" python3 -c \
            'import json,os,sys; v=json.load(sys.stdin)[os.environ["_FIELD"]]; print("null" if v is None else str(v).lower() if isinstance(v,bool) else v)'
    }

    # 5a. The ON list means the same at both ends: these are the values that turn a tree
    # the installer set up normally into a portable one.
    for v in 1 true yes on " TRUE " " on "; do
        check "the runtime reads '$v' as portable" true "$(field "$(probe_json "$v")" portable)"
    done

    # 5b. The OFF list, likewise.
    for v in 0 false off no FALSE " no "; do
        check "the runtime reads '$v' as normal" false "$(field "$(probe_json "$v")" portable)"
    done

    # 5c. And the refused set is now exactly the runtime's no-opinion set. This is the
    # assertion the whole guard rests on: for every value install.sh refuses in section 3,
    # a normal tree stays normal, and none of the portable-only variables get pinned. If
    # portable_mode() ever goes back to guessing, the install refused in section 3 and the
    # relocation seen here are the same split this file exists to prevent.
    for v in enabled flase 2 bogus disabled ENABLED " enabled " "-1"; do
        _j="$(probe_json "$v")"
        check "the runtime does not read '$v' as portable" false "$(field "$_j" portable)"
        check "and pins no TORCH_HOME for '$v'" null "$(field "$_j" torch_home)"
        check "and pins no projects root for '$v'" null "$(field "$_j" projects_home)"
        check "and pins no datasets cache for '$v'" null "$(field "$_j" datasets_cache)"
    done

    # 5d. No opinion is a fall-through, not a veto. A real portable install carries its
    # root, and the root is what makes it portable, so the same unrecognized value that
    # cannot opt a normal tree IN must not strand a portable one either. Getting this
    # wrong would scatter a running install's caches back across the host.
    for v in enabled flase 2 bogus disabled; do
        check "a portable root survives '$v'" true "$(field "$(probe_json "$v" "$PROOT")" portable)"
    done
    # Not just a boolean: the redirection itself still has to follow the root.
    _pj="$(probe_json enabled "$PROOT")"
    check "and TORCH_HOME follows the portable root" "$PROOT/studio/cache/torch" \
        "$(field "$_pj" torch_home)"
    check "and the projects root follows it too" "$PROOT/studio/projects" \
        "$(field "$_pj" projects_home)"

    # 5e. An explicit OFF is the same fall-through, and always was: it declines to opt a
    # normal install in rather than vetoing a root. Pinned here because 5c leans on the
    # unrecognized values behaving exactly like these.
    for v in 0 false off no; do
        check "an explicit '$v' does not veto a portable root" true \
            "$(field "$(probe_json "$v" "$PROOT")" portable)"
    done

    # 5f. Silently ignoring the value is what made the old split invisible, so the runtime
    # says so, names the value as typed, and stays quiet for a value it understands.
    probe_json Enabled > /dev/null
    case "$(cat "$T/pout")" in
        *UNSLOTH_PORTABLE*Enabled*) check "the runtime warns, quoting the value as typed" ok ok ;;
        *) check "the runtime warns, quoting the value as typed" ok "missing: $(cat "$T/pout")" ;;
    esac
    probe_json 1 > /dev/null
    # grep -c exits 1 on no match, so read the count, not the status.
    check "and says nothing for a value it understands" 0 "$(grep -c UNSLOTH_PORTABLE "$T/pout")"
else
    printf '  SKIP  storage_roots probe (no python3)\n'
fi

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
