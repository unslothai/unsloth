#!/usr/bin/env bash
# Regression test: the portable summary prints a launch command you can paste.
#
# The summary block names three commands for the same install: the launch, the `rm -rf` that
# removes it, and (further down) the deferred manual-launch hint. The removal and the hint
# single-quote the path; the launch line interpolated it bare, so a root the installer
# explicitly supports -- `--root '/Volumes/My Drive/unsloth'` -- printed a command the shell
# splits into `/Volumes/My`, and one holding a glob character printed one the shell expands.
# The same summary then printed the quoted spelling, so the two instructions contradicted
# each other.
#
# Not a grep for quotes: each rendered command is re-parsed by a shell and the resulting
# argv is compared against the path the installer actually installed to.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
ROOT="$HERE/../.."
INSTALL="$ROOT/install.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

# The portable arm of the summary, and the deferred hint's quoting, both verbatim.
blockS="$(awk '
    /^printf "  \$\{C_TITLE\}%s\$\{C_RST\}\\n" "Unsloth Studio installed!"$/ {seen = 1}
    seen && /^if \[ "\$_PORTABLE_MODE" = true \]; then$/ {grab = 1}
    grab {print}
    grab && /^fi$/ {exit}
' "$INSTALL")"
blockQ="$(grep -n '^_li_shim_q=' "$INSTALL" | head -n1 | cut -d: -f2-)"

case "$blockS" in *'launch it with'*) : ;; *) echo "FAIL: blockS extraction broke"; exit 1 ;; esac
case "$blockS" in *'rm -rf'*) : ;; *) echo "FAIL: blockS lost the removal command"; exit 1 ;; esac
case "$blockQ" in *'_LOCAL_BIN'*) : ;; *) echo "FAIL: could not extract the deferred launch hint"; exit 1 ;; esac

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
# mktemp per case, not a counter: an increment inside `$( )` happens in a subshell.
new_dir() { mktemp -d "$T/case.XXXXXX"; }

# Render the summary for one root and return the marked lines.
render() { # root
    _d="$(new_dir)"
    env -i HOME="$_d" PATH="$PATH" _R="$1" bash -c '
        set -e
        C_WARN=""
        substep() { printf "%s\n" "$1"; }
        _portable_escapes() { :; }
        _PORTABLE_MODE=true
        UNSLOTH_ROOT="$_R"
        STUDIO_HOME="$UNSLOTH_ROOT/studio"
        VENV_DIR="$STUDIO_HOME/unsloth_studio"
        _LOCAL_BIN="$UNSLOTH_ROOT/bin"
        '"$blockS"'
        '"$blockQ"'
        printf "HINT|%s studio -p 8888\n" "$_li_shim_q"
    ' 2>&1
}

# Re-parse a printed command the way a user pasting it would, and report "argc|argv0|argv1".
reparse() { # command-text
    env -i PATH="$PATH" _CMD="$1" bash -c '
        eval "set -- $_CMD"
        printf "%s|%s|%s" "$#" "${1:-}" "${2:-}"
    ' 2>/dev/null || printf 'unparseable'
}

run_case() { # root label
    _root="$1"; _label="$2"
    _out="$(render "$_root")"
    _launch="$(printf '%s\n' "$_out" | sed -n 's/^launch it with //p')"
    _remove="$(printf '%s\n' "$_out" | sed -n 's/^  rm -rf //p')"
    _hint="$(printf '%s\n' "$_out" | sed -n 's/^HINT|//p')"
    if [ -z "$_launch" ]; then
        printf '  FAIL  %s : no launch line rendered\n%s\n' "$_label" "$_out"
        fails=$((fails + 1))
        return 0
    fi
    # The pasted launch command must be exactly two words: the shim, then `studio`.
    check "$_label: the summary command pastes as one program plus 'studio'" \
        "2|$_root/bin/unsloth|studio" "$(reparse "$_launch")"
    # ...and it must name the same program as the deferred hint, which already quoted it.
    check "$_label: it agrees with the deferred manual-launch hint" \
        "$_root/bin/unsloth" "$(printf '%s' "$(reparse "$_hint")" | cut -d'|' -f2)"
    # The neighbouring removal command was already quoted; keep it that way.
    check "$_label: the removal command still pastes as one path" \
        "1|$_root|" "$(reparse "$_remove")"
}

run_case "/plain/unsloth"                 "plain path"
run_case "/Volumes/My Drive/unsloth"      "spaces (the bug)"
run_case "/home/star*glob/unsloth"        "glob star"
run_case "/home/quest?ion/unsloth"        "glob question mark"
run_case "/home/brack[et]s/unsloth"       "glob bracket"
run_case "/home/o'brien/unsloth"          "apostrophe"
run_case '/home/dollar$var/unsloth'       "dollar sign"
run_case '/home/tick`cmd`/unsloth'        "backtick"
run_case '/home/semi;colon/unsloth'       "semicolon"
run_case '/home/back\slash/unsloth'       "backslash"
run_case '/home/quote"dq/unsloth'         "double quote"
run_case '/home/paren(s)/unsloth'         "parentheses"
run_case '/home/amp&sand/unsloth'         "ampersand"
run_case "/home/ünïcodé/unsloth"          "unicode"
run_case "/home/$(printf 'tab\tchar')/un" "tab"

if [ "$fails" -eq 0 ]; then
    printf 'ALL PASS\n'
else
    printf '%s FAILURES\n' "$fails"
    exit 1
fi
