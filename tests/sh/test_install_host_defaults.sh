#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Installers and README must not show a wildcard bind (0.0.0.0) as the DEFAULT launch command.
# Nothing here keys on README headings or comment prose; flag spellings are read from the
# typer.Option in unsloth_cli/commands/studio.py, and every window asserts it closed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"
INSTALL_PS1="$REPO_ROOT/install.ps1"
SETUP_SH="$REPO_ROOT/studio/setup.sh"
README="$REPO_ROOT/README.md"
STUDIO_CLI="$REPO_ROOT/unsloth_cli/commands/studio.py"
assert_ge() {
    _label="$1"; _actual="$2"; _min="$3"
    if [ "$_actual" -ge "$_min" ] 2> /dev/null; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected at least $_min, got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

# One shared probe for the README and installer halves so they cannot drift.
PROBE="$(mktemp)"
trap 'rm -f "$PROBE"' EXIT INT TERM
cat > "$PROBE" << 'PROBE_PY'
"""Structural probe for the host-defaults guard. See tests/sh/test_install_host_defaults.sh.

Usage:
    probe.py flags   <studio.py>            -> one host flag spelling per line
    probe.py readme  <README.md> <studio.py> -> `key<TAB>value` facts
    probe.py scan    <studio.py>            -> reads text on stdin, prints the
                                               number of wildcard-binding studio
                                               commands in it
"""

import ast
import re
import sys

WILDCARDS = ("0.0.0.0", "[::]", "::")

# A command token boundary: what may sit immediately before the `studio`
# subcommand word. Bare `studio` rather than `unsloth studio`, because
# install.sh's generated launcher runs `"$UNSLOTH_EXE" studio` and install.ps1
# has an `unsloth.cmd studio` variant, and both are launch commands a user ends
# up running. The trailing boundary is whitespace or end of string, which is
# what keeps studio.conf, studio.log, studio-<port>.pid, unsloth-studio-launcher
# and shutdown_studio from being read as invocations.
_BEFORE = " \t\"'/"
_WORD = "studio"

def host_flags(source):
    """Every CLI spelling of the host option, read off the typer.Option that declares it.

    Anchored on the Python parameter NAME (`host`), so the surface spellings
    (`-H`, `--host`) are outputs of this function rather than inputs to it.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = node.args.posonlyargs + node.args.args
        defaults = node.args.defaults
        pairs = list(zip(positional[len(positional) - len(defaults):], defaults))
        pairs += [
            (a, d)
            for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if d is not None
        ]
        for arg, default in pairs:
            if arg.arg != "host" or not isinstance(default, ast.Call):
                continue
            func = default.func
            if not (isinstance(func, ast.Attribute) and func.attr == "Option"):
                continue
            for a in default.args[1:]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    if a.value.startswith("-"):
                        found.add(a.value)
    return sorted(found)

def wildcard_pattern(flags):
    """A host flag bound to a wildcard address, in any of the flag's spellings.

    The trailing lookahead is a NON-ADDRESS character rather than whitespace,
    because these commands are usually quoted inside the shell or PowerShell that
    prints them: `"unsloth studio -p 8888 -H 0.0.0.0"` ends the address with a
    quote, and a whitespace-only boundary reads that as no match at all. It still
    has to be a boundary, so `-H 0.0.0.0.5` and the `::1` loopback do not count as
    wildcards, while `-H 0.0.0.0:8888` does.
    """
    alternation = "|".join(re.escape(f) for f in sorted(flags, key = len, reverse = True))
    values = "|".join(re.escape(w) for w in sorted(WILDCARDS, key = len, reverse = True))
    return re.compile(
        r"(?:^|\s)(?:%s)(?:\s+|=)(?:%s)(?=$|[^0-9A-Za-z._-])" % (alternation, values)
    )

def code_blocks(text):
    """Fenced code blocks, CommonMark-style. Returns (blocks, unterminated).

    The opening fence is 0-3 spaces then three or more backticks or tildes; the
    close is the same character, at least as long, with nothing after it. The
    line-start anchor is what stops the inline ```unsloth/unsloth``` span in the
    Docker paragraph from opening a block and swallowing the rest of the file.
    """
    blocks, current, char, length = [], None, None, 0
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        head = stripped[:1]
        run = 0
        if indent <= 3 and head in ("`", "~"):
            while run < len(stripped) and stripped[run] == head:
                run += 1
        if current is None:
            if run >= 3 and not (head == "`" and "`" in stripped[run:]):
                current, char, length = [], head, run
            continue
        if run >= 3 and head == char and run >= length and not stripped[run:].strip():
            blocks.append("\n".join(current))
            current = None
            continue
        current.append(line)
    if current is not None:
        blocks.append("\n".join(current))
        return blocks, True
    return blocks, False

_COMMENT = re.compile(r"(?:^|\s)#.*$")

def studio_commands(text):
    """Every `... studio <args>` command in *text*, as the tail from `studio` onwards.

    A trailing comment is dropped so prose such as `unsloth studio  # add -H 0.0.0.0
    for LAN access` is read as the loopback command it is. The comment marker has to
    start a token, or a `#` inside a URL or a printf format would truncate a real
    command and hide a wildcard bind sitting after it.
    """
    out = []
    for raw in text.splitlines():
        line = _COMMENT.sub("", raw)
        cursor = 0
        while True:
            at = line.find(_WORD, cursor)
            if at < 0:
                break
            cursor = at + len(_WORD)
            before = line[at - 1] if at else ""
            after = line[cursor:cursor + 1]
            if (at == 0 or before in _BEFORE) and after in ("", " ", "\t"):
                out.append(line[at:].strip())
    return out

def main():
    mode = sys.argv[1]
    if mode == "flags":
        source = open(sys.argv[2], encoding = "utf-8").read()
        print("\n".join(host_flags(source)))
        return 0
    if mode == "scan":
        source = open(sys.argv[2], encoding = "utf-8").read()
        pattern = wildcard_pattern(host_flags(source))
        text = sys.stdin.read()
        print(sum(1 for c in studio_commands(text) if pattern.search(c)))
        return 0
    if mode == "readme":
        readme = open(sys.argv[2], encoding = "utf-8").read()
        source = open(sys.argv[3], encoding = "utf-8").read()
        flags = host_flags(source)
        pattern = wildcard_pattern(flags)
        blocks, unterminated = code_blocks(readme)
        commands = [c for b in blocks for c in studio_commands(b)]
        facts = {
            "flags": " ".join(flags),
            "long_flags": str(sum(1 for f in flags if f.startswith("--"))),
            "fences": "unterminated" if unterminated else "balanced",
            "blocks": str(len(blocks)),
            "commands": str(len(commands)),
            "primary": commands[0] if commands else "",
            "primary_binds_wildcard": (
                "yes" if commands and pattern.search(commands[0]) else "no"
            ),
            "wildcard_opt_ins": str(sum(1 for c in commands if pattern.search(c))),
        }
        for key, value in facts.items():
            print("%s\t%s" % (key, value))
        return 0
    raise SystemExit("unknown mode: %s" % mode)

if __name__ == "__main__":
    raise SystemExit(main())
PROBE_PY

probe_fact() { printf '%s\n' "$_readme_facts" | awk -F '\t' -v k="$1" '$1 == k {print $2; found = 1} END {if (!found) exit 1}'; }

assert_no_wildcard_bind() {
    _label="$1"; _haystack="$2"
    _hits=$(printf '%s\n' "$_haystack" | python3 "$PROBE" scan "$STUDIO_CLI")
    if [ "$_hits" = "0" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label ($_hits launch command(s) bind a wildcard host)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=== host option spellings (derived from the CLI) ==="

_host_flags=$(python3 "$PROBE" flags "$STUDIO_CLI")
# Canary: a renamed option would otherwise leave every negative assertion matching nothing.
assert_contains \
    "host flags: the studio CLI declares a long --host option" \
    "$_host_flags" "--"
echo "  (derived: $(printf '%s' "$_host_flags" | tr '\n' ' '))"

echo ""
echo "=== the detector, against fixtures ==="

# Every negative assertion passes when the detector sees nothing, so pin both directions.
assert_detects() {
    _label="$1"; _line="$2"; _want="$3"
    _hits=$(printf '%s\n' "$_line" | python3 "$PROBE" scan "$STUDIO_CLI")
    if [ "$_hits" -ge 1 ] 2> /dev/null; then _got="detected"; else _got="ignored"; fi
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: detector: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: detector: $_label (wanted $_want, got $_got, on: $_line)"
        FAIL=$((FAIL + 1))
    fi
}

assert_detects "bare wildcard bind"                 'unsloth studio -H 0.0.0.0'                    detected
assert_detects "long flag"                          'unsloth studio --host 0.0.0.0'                detected
assert_detects "long flag with ="                   'unsloth studio --host=0.0.0.0'                detected
assert_detects "flags between studio and the bind"  'unsloth studio -p 8888 -H 0.0.0.0'            detected
assert_detects "trailing flags after the bind"      'unsloth studio -H 0.0.0.0 -p 8888'            detected
assert_detects "shell-prompt prefix"                '$ unsloth studio -H 0.0.0.0'                  detected
assert_detects "inline env assignment prefix"       "PW='x' unsloth studio -H 0.0.0.0"             detected
assert_detects "the prefixed shape from #9654"      "UNSLOTH_STUDIO_PASSWORD='x' unsloth studio --host=0.0.0.0" detected
assert_detects "indented inside a fenced block"     '    unsloth studio -H 0.0.0.0'                detected
assert_detects "quoted inside a printf"             'printf "%s" "unsloth studio -p 1 -H 0.0.0.0"' detected
# The fixture is the literal text a launcher script CONTAINS, so the `$` must not expand.
# shellcheck disable=SC2016
assert_detects "launched through a variable"        'exec "$UNSLOTH_EXE" studio -H 0.0.0.0'        detected
assert_detects "wildcard with a port suffix"        'unsloth studio -H 0.0.0.0:8888'               detected
assert_detects "the IPv6 wildcard"                  'unsloth studio -H [::]'                       detected
assert_detects "loopback default"                   'unsloth studio'                               ignored
assert_detects "an explicit loopback bind"          'unsloth studio -H 127.0.0.1'                  ignored
assert_detects "IPv6 loopback is not the wildcard"  'unsloth studio -H ::1'                        ignored
assert_detects "a longer address starting 0.0.0.0"  'unsloth studio -H 0.0.0.0.5'                  ignored
assert_detects "an address that merely begins 0.0.0.0" 'unsloth studio -H 0.0.0.01'                ignored
assert_detects "another program's wildcard bind"    'llama-server --host 0.0.0.0'                  ignored
assert_detects "another program, another flag"      'jupyter lab --ip 0.0.0.0'                     ignored
assert_detects "prose naming the opt-in"            'add -H 0.0.0.0 for LAN / cloud access'        ignored
assert_detects "a trailing comment naming it"       'unsloth studio  # add -H 0.0.0.0 for LAN'     ignored
assert_detects "a path, not the subcommand"         'unsloth-studio-launcher -H 0.0.0.0'           ignored
assert_detects "an empty window"                    ''                                             ignored

# Windows are cut from the file's own structure, never from comments or prose.

# Heredoc redirecting into $2, opener through terminator; want=delim returns the terminator.
# Both come from the same matched line. Continuations are joined only until the heredoc opens.
_heredoc_window() {
    awk -v q="\"'" -v target="$2" -v want="${3:-body}" '
    !delim {
        line = $0
        while (line ~ /\\[ \t]*$/ && (getline nxt) > 0) {
            sub(/\\[ \t]*$/, "", line)
            sub(/^[ \t]+/, " ", nxt)
            line = line nxt
        }
        if (!index(line, target) || !index(line, "<<")) next
        rest = substr(line, index(line, "<<") + 2)
        if (substr(rest, 1, 1) == "-") { dash = 1; rest = substr(rest, 2) }
        sub(/^[ \t]+/, "", rest)
        if (index(q, substr(rest, 1, 1))) rest = substr(rest, 2)
        if (!match(rest, /^[A-Za-z_][A-Za-z0-9_]*/)) next
        delim = substr(rest, 1, RLENGTH)
        if (want == "delim") { print delim; exit }
        print line
        next
    }
    {
        print
        line = $0
        if (dash) sub(/^[ \t]+/, "", line)
        if (line == delim) exit
    }
    ' "$1"
}

# Top-level `if` matching $2 through the column-zero `fi` that closes it.
_shell_if_block() {
    awk -v pat="$2" '
    !found && $0 ~ pat { found = 1; print; next }
    found { print; if ($0 == "fi") exit }
    ' "$1"
}

# `[{]` so the brace is a literal, not an ERE interval.
_ps_brace_block() {
    awk -v pat="$2" '
    !found && $0 ~ pat { found = 1 }
    found { print; depth += gsub(/[{]/, "&") - gsub(/[}]/, "&"); if (depth <= 0) exit }
    ' "$1"
}

# A window that ran off the end of the file does not end on its own closing token.
_window_close() { printf '%s\n' "$1" | tail -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

echo ""
echo "=== install.sh launcher template ==="

# One helper for start and delimiter: a separate delimiter grep once matched nothing on a
# split redirect and collapsed the window to a few lines.
_launcher_delim=$(_heredoc_window "$INSTALL_SH" '_css_launcher' delim)
_launcher=$(_heredoc_window "$INSTALL_SH" '_css_launcher' body)
assert_ge \
    "launcher template: a heredoc terminator was derived" \
    "${#_launcher_delim}" 1
assert_contains \
    "launcher template: extraction found the heredoc content" \
    "$_launcher" "#!/usr/bin/env bash"
assert_eq \
    "launcher template: the window closes on the heredoc terminator" \
    "$_launcher_delim" "$(_window_close "$_launcher")"
assert_no_wildcard_bind \
    "launcher template: the generated launcher binds no wildcard host" \
    "$_launcher"

echo ""
echo "=== install.sh end-of-install block ==="

# Anchored on the `if` itself; `_SKIP_AUTOSTART` renames are caught by the prompt canary.
_end=$(_shell_if_block "$INSTALL_SH" '^if .*_SKIP_AUTOSTART.*; then$')
# "read" alone also matches "readable" and "_can_read_tty".
assert_contains \
    "install.sh: interactive block prompts user (read)" \
    "$_end" "read -r _reply"
assert_eq \
    "install.sh: the end-of-install window closes on its own fi" \
    "fi" "$(_window_close "$_end")"
assert_no_wildcard_bind \
    "install.sh: end-of-install commands bind no wildcard host" \
    "$_end"

echo ""
echo "=== install.ps1 end-of-install block ==="

_ps1_end=$(_ps_brace_block "$INSTALL_PS1" '^[ \t]*if [(][$]IsInteractive[)] [{][ \t]*$')
assert_contains \
    "install.ps1: interactive block prompts user (Read-Host)" \
    "$_ps1_end" "Read-Host"
assert_eq \
    "install.ps1: the end-of-install window closes on its own brace" \
    "}" "$(_window_close "$_ps1_end")"
assert_no_wildcard_bind \
    "install.ps1: end-of-install commands bind no wildcard host" \
    "$_ps1_end"

echo ""
echo "=== studio/setup.sh launch hint ==="

# Deliberately not windowed: the `_LLAMA_ONLY` header opens three blocks in setup.sh. Its
# opt-in text is prose without `studio`, so reading the whole file does not trip.
_setup_all=$(cat "$SETUP_SH")
# Canary: a file that stopped mentioning studio would satisfy the assertion below.
assert_contains \
    "studio/setup.sh: the footer still prints a launch hint" \
    "$_setup_all" "unsloth studio -p 8888"
assert_no_wildcard_bind \
    "studio/setup.sh: no launch command in setup.sh binds a wildcard host" \
    "$_setup_all"

echo ""
echo "=== the installers, whole file ==="

# Whole-file checks cannot be narrowed by an edit. If they fire, rephrase the opt-in prose
# rather than deleting them.
assert_no_wildcard_bind \
    "install.sh: no launch command anywhere in the file binds a wildcard host" \
    "$(cat "$INSTALL_SH")"
assert_no_wildcard_bind \
    "install.ps1: no launch command anywhere in the file binds a wildcard host" \
    "$(cat "$INSTALL_PS1")"

echo ""
echo "=== README.md launch commands ==="

_readme_facts=$(python3 "$PROBE" readme "$README" "$STUDIO_CLI")

assert_contains \
    "README: the structural probe reported its facts" \
    "$_readme_facts" "commands"

# Canaries: an empty parse would let the negative assertion pass vacuously.
assert_eq \
    "README: every fenced code block is terminated" \
    "balanced" "$(probe_fact fences)"
assert_ge \
    "README: fenced code blocks were parsed" \
    "$(probe_fact blocks)" 1
assert_ge \
    "README: the README shows at least two studio launch commands" \
    "$(probe_fact commands)" 2

# The first command in document order is the primary one, wherever it moved.
assert_eq \
    "README: the primary launch command binds no wildcard host" \
    "no" "$(probe_fact primary_binds_wildcard)"
assert_ge \
    "README: a wildcard-host bind is documented as an explicit opt-in" \
    "$(probe_fact wildcard_opt_ins)" 1
echo "  (primary: $(probe_fact primary))"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
