#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Installer scripts and README must not put a wildcard bind in a user-visible
# DEFAULT launch command. Binding 0.0.0.0 exposes the raw port to the LAN, so it
# has to be something the reader opts into, never what they are handed first.
# Provenance: #5267 (default the Studio host to 127.0.0.1) and #7774 (anchor the
# host-defaults assertions).
#
# The property, stated without reference to any heading:
#
#   1. The FIRST `unsloth studio` command the README shows must not bind a
#      wildcard host.
#   2. A wildcard bind must still be documented SOMEWHERE, as a deliberate
#      opt-in, so the capability is not simply undocumented.
#   3. No launch command the installers print or generate binds a wildcard host.
#
# Two things this file used to do, and no longer does, because both went red on
# an edit that was correct:
#
#   * It sliced README.md by heading -- `#### Launch` up to the next heading of
#     any level -- and asserted BOTH halves inside that one window. Deleting
#     `#### Update` silently extended the window to end-of-file (fixed once by
#     adding the stop-at-any-heading rule), and then moving the opt-in under a
#     sibling `#### Remote HTTPS & LAN Access` moved it out of the window and
#     broke the positive half. The README's heading structure is edited directly
#     and often; it is not a stable index and must not be load-bearing. The
#     assertions below survive a section being renamed, reordered, merged or
#     split, because they never look at a heading.
#
#   * It hard-coded the needle `studio -H 0.0.0.0`. That misses
#     `studio -p 8888 -H 0.0.0.0`, misses `--host 0.0.0.0`, misses `-H=0.0.0.0`,
#     and -- the reason it matters most -- would go permanently, silently vacuous
#     the day `-H` is renamed. The flag spellings are now read off the
#     `typer.Option` in unsloth_cli/commands/studio.py that declares them, so the
#     needle is derived from the code it protects rather than re-spelled here.
#
# `unsloth studio`'s own default (`--host 127.0.0.1`) is asserted separately and
# structurally in tests/studio/test_cli_studio_defaults.py; this file is about
# what the docs and installers SHOW.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"
INSTALL_PS1="$REPO_ROOT/install.ps1"
SETUP_SH="$REPO_ROOT/studio/setup.sh"
README="$REPO_ROOT/README.md"
STUDIO_CLI="$REPO_ROOT/unsloth_cli/commands/studio.py"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

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

# ── the probe ────────────────────────────────────────────────────────────────
# One definition of "a studio launch command that binds a wildcard host", shared
# by the README half and the installer half below, so the two cannot drift into
# meaning different things. Written to a temp file rather than inlined at each
# call site for the same reason.
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

# A studio launch command that binds a wildcard host, anywhere in $2. Used for
# the plain-text installer windows; the README goes through `readme` mode above,
# which adds the fenced-block parse but shares this same command matcher.
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
# Canary. If the parameter is renamed or the option removed, this reports it
# rather than leaving every negative assertion below matching nothing forever.
assert_contains \
    "host flags: the studio CLI declares a long --host option" \
    "$_host_flags" "--"
echo "  (derived: $(printf '%s' "$_host_flags" | tr '\n' ' '))"

echo ""
echo "=== the detector, against fixtures ==="

# Every negative assertion in this file passes when the detector sees NOTHING,
# so a detector that quietly stops matching turns the whole file green while
# guarding nothing. These pin both directions of it directly, so that failure
# reports itself here instead of hiding behind five green lines below.
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
assert_detects "shell-prompt prefix"                '$ unsloth studio -H 0.0.0.0'                  detected
assert_detects "inline env assignment prefix"       "PW='x' unsloth studio -H 0.0.0.0"             detected
assert_detects "indented inside a fenced block"     '    unsloth studio -H 0.0.0.0'                detected
assert_detects "quoted inside a printf"             'printf "%s" "unsloth studio -p 1 -H 0.0.0.0"' detected
assert_detects "launched through a variable"        'exec "$UNSLOTH_EXE" studio -H 0.0.0.0'        detected
assert_detects "wildcard with a port suffix"        'unsloth studio -H 0.0.0.0:8888'               detected
assert_detects "the IPv6 wildcard"                  'unsloth studio -H [::]'                       detected
assert_detects "loopback default"                   'unsloth studio'                               ignored
assert_detects "an explicit loopback bind"          'unsloth studio -H 127.0.0.1'                  ignored
assert_detects "IPv6 loopback is not the wildcard"  'unsloth studio -H ::1'                        ignored
assert_detects "a longer address starting 0.0.0.0"  'unsloth studio -H 0.0.0.0.5'                  ignored
assert_detects "another program's wildcard bind"    'llama-server --host 0.0.0.0'                  ignored
assert_detects "prose naming the opt-in"            'add -H 0.0.0.0 for LAN / cloud access'        ignored
assert_detects "a trailing comment naming it"       'unsloth studio  # add -H 0.0.0.0 for LAN'     ignored
assert_detects "a path, not the subcommand"         'unsloth-studio-launcher -H 0.0.0.0'           ignored
assert_detects "an empty window"                    ''                                             ignored

echo ""
echo "=== install.sh launcher template ==="

# Extract the heredoc that generates ~/.local/share/unsloth/launch-studio.sh.
# The terminator is read off the `<<` that opens it, so renaming the delimiter
# cannot silently truncate the window to nothing.
_launcher_delim=$(grep -m1 'cat > "\$_css_launcher" <<' "$INSTALL_SH" \
    | sed -E "s/.*<<-?[[:space:]]*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?.*/\1/")
_launcher=$(awk -v delim="$_launcher_delim" \
    '/cat > "\$_css_launcher"/{found=1} found{print} $0 == delim{found=0}' "$INSTALL_SH")
assert_contains \
    "launcher template: extraction found the heredoc content" \
    "$_launcher" "#!/usr/bin/env bash"
assert_no_wildcard_bind \
    "launcher template: the generated launcher binds no wildcard host" \
    "$_launcher"

echo ""
# Anchored on content, not a line count: both installers outgrew their tail windows.
echo "=== install.sh end-of-install block ==="

_end=$(awk '/In interactive terminals/{found=1} found{print}' "$INSTALL_SH")
# "read" alone also matches "readable" and "_can_read_tty", so pin the full prompt.
assert_contains \
    "install.sh: interactive block prompts user (read)" \
    "$_end" "read -r _reply"
assert_no_wildcard_bind \
    "install.sh: end-of-install commands bind no wildcard host" \
    "$_end"

echo ""
echo "=== install.ps1 end-of-install block ==="

_ps1_end=$(awk '/In interactive terminals/{found=1} found{print}' "$INSTALL_PS1")
assert_contains \
    "install.ps1: interactive block prompts user (Read-Host)" \
    "$_ps1_end" "Read-Host"
assert_no_wildcard_bind \
    "install.ps1: end-of-install commands bind no wildcard host" \
    "$_ps1_end"

echo ""
echo "=== studio/setup.sh launch hint ==="

_setup_tail=$(awk '/"launch"/{found=1} found{print}' "$SETUP_SH")
# Canary: an empty window would let the negative assertion below pass vacuously.
assert_contains \
    "studio/setup.sh: extraction found the launch hint" \
    "$_setup_tail" "unsloth studio -p 8888"
assert_no_wildcard_bind \
    "studio/setup.sh: the launch hint binds no wildcard host" \
    "$_setup_tail"

echo ""
echo "=== README.md launch commands ==="

_readme_facts=$(python3 "$PROBE" readme "$README" "$STUDIO_CLI")

# The probe crashing is a different failure from the README being wrong, and it
# should not arrive as a bare `set -e` abort halfway down the output.
assert_contains \
    "README: the structural probe reported its facts" \
    "$_readme_facts" "commands"

# Three canaries before the two real assertions. Each one is a way the parse
# could come back empty or wrong, and an empty parse is exactly how a negative
# assertion passes while guarding nothing.
assert_eq \
    "README: every fenced code block is terminated" \
    "balanced" "$(probe_fact fences)"
assert_ge \
    "README: fenced code blocks were parsed" \
    "$(probe_fact blocks)" 1
assert_ge \
    "README: the README shows at least two studio launch commands" \
    "$(probe_fact commands)" 2

# The property. No heading is consulted anywhere in either of these: the first
# command in document order is the primary one, wherever it has been moved to.
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
