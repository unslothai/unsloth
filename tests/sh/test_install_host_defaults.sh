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
# Three things this file used to do, and no longer does. The first two went red
# on an edit that was correct; the third went GREEN on one that was not:
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
#   * It sliced the three installer files on hand-written markers, two of which
#     were COMMENT PROSE ("In interactive terminals, ..."). A comment has no
#     reason to stay stable, so rewording one turned this guard red for reasons
#     that had nothing to do with host defaults, and three of the four windows
#     ran to end of file, so each negative assertion covered a superset of the
#     region its label named. Worse, the launcher window could go VACUOUS rather
#     than loud: see the reproduction in the install.sh launcher section below.
#     Every installer window is now derived from the file's own structure -- a
#     heredoc delimiter, a top-level `if`/`fi`, a PowerShell brace block -- and
#     every one of them asserts that it closed on its own closing token instead
#     of running off the end of the file.
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
assert_detects "trailing flags after the bind"      'unsloth studio -H 0.0.0.0 -p 8888'            detected
assert_detects "shell-prompt prefix"                '$ unsloth studio -H 0.0.0.0'                  detected
assert_detects "inline env assignment prefix"       "PW='x' unsloth studio -H 0.0.0.0"             detected
# The exact shape #9654 found getting past a line-start-anchored detector. Kept
# verbatim rather than paraphrased, because it is the one that was actually let through.
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

# ── the windows ──────────────────────────────────────────────────────────────
# Every installer window below is cut from the file's own STRUCTURE, never from a
# comment and never from a sentence someone might reword. Anchoring on a comment
# is the worst case of all, because a comment exists to be rewritten.

# The heredoc that redirects into the path held by $2, from the line that opens
# it through its own terminator. `want=delim` returns that terminator instead of
# the body. Both come from the SAME matched line in the SAME pass, which is the
# whole point: a window opened by one rule and closed by another can disagree,
# and when it does the disagreement is silent.
#
# While looking for the opener it joins backslash continuations, so the redirect
# and its `<<` are matched as one shell LOGICAL line however they are wrapped.
# The joining stops the moment the heredoc opens, because the launcher body has
# continuations of its own and they are content, not syntax to be folded away.
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

# A top-level shell branch: the `if` line matching $2, through the `fi` in column
# zero that closes it. Nested branches inside it are indented, so the first
# column-zero `fi` is the right one. Callers assert the window ended on that `fi`,
# which is what turns a close that never matched into a report instead of a
# silent run to end of file.
_shell_if_block() {
    awk -v pat="$2" '
    !found && $0 ~ pat { found = 1; print; next }
    found { print; if ($0 == "fi") exit }
    ' "$1"
}

# A PowerShell block: the line matching $2, through the line where its braces
# balance again. `[{]` rather than `{` so the brace is a literal and not the
# start of an ERE interval.
_ps_brace_block() {
    awk -v pat="$2" '
    !found && $0 ~ pat { found = 1 }
    found { print; depth += gsub(/[{]/, "&") - gsub(/[}]/, "&"); if (depth <= 0) exit }
    ' "$1"
}

# The last line of a window, trimmed. A window that ran off the end of the file
# does not end on its own closing token; that is the difference between a window
# that closed on purpose and one that was never closed at all.
_window_close() { printf '%s\n' "$1" | tail -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

echo ""
echo "=== install.sh launcher template ==="

# The heredoc that generates ~/.local/share/unsloth/launch-studio.sh.
#
# This window is the one shape in this file that could go vacuous rather than
# loud, so it is worth stating exactly. It used to find its start with an awk
# pattern and its terminator with a SEPARATE, stricter grep that also required
# the `<<` on that same line. Split the redirect across a line continuation --
#
#     cat > "$_css_launcher" \
#         << 'LAUNCHER_EOF'
#
# -- and the grep matched nothing, the derived delimiter was the empty string,
# and `$0 == delim` then matched the first BLANK LINE inside the launcher. The
# window collapsed from 357 lines to 6. The `#!/usr/bin/env bash` canary sits in
# those 6 and still passed, so `-H 0.0.0.0` added to the launcher's own `exec`
# line went through at 39 pass / 0 fail. One helper, one matched line, no second
# pattern to disagree with the first.
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

# The post-install prompt-or-print branch, anchored on the `if` that declares it.
# The anchor used to be the comment ABOVE that `if` ("In interactive terminals,
# ask the user before starting Unsloth ..."), and the window ran from there to
# end of file. `_SKIP_AUTOSTART` is a shell variable this branch reads: renaming
# it is a code change, and the prompt canary below reports it at once instead of
# leaving an empty window behind.
_end=$(_shell_if_block "$INSTALL_SH" '^if .*_SKIP_AUTOSTART.*; then$')
# "read" alone also matches "readable" and "_can_read_tty", so pin the full prompt.
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

# Same branch, same reasoning, PowerShell braces instead of `fi`. Also anchored
# on the comment above it until now, and also ran to end of file.
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

# Deliberately NOT windowed, and the label says so. The footer that prints the
# hint lives in a top-level `if [ "$_LLAMA_ONLY" = "1" ]; then` block, and that
# same header opens three different top-level blocks in setup.sh, so there is no
# structural boundary that picks out the footer -- only the ordinal "the third
# one", which is a hand-written marker wearing a different hat. The old anchor
# was the bare string `"launch"` and ran to end of file from there, so reading
# the whole file loses no coverage, gains the rest of it, and cannot be narrowed
# by an edit anywhere. Over-reporting here is cheap; the file's own wildcard
# opt-in is prose ("add -H 0.0.0.0 for LAN / cloud access") with no `studio` word
# in it, so it is not a launch command and does not trip this.
_setup_all=$(cat "$SETUP_SH")
# Canary: setup.sh still prints a launch hint at all. A file that stopped
# mentioning studio would otherwise satisfy the assertion below by saying nothing.
assert_contains \
    "studio/setup.sh: the footer still prints a launch hint" \
    "$_setup_all" "unsloth studio -p 8888"
assert_no_wildcard_bind \
    "studio/setup.sh: no launch command in setup.sh binds a wildcard host" \
    "$_setup_all"

echo ""
echo "=== the installers, whole file ==="

# The windows above name the region each guard is ABOUT, which is what makes a
# failure readable. These two name the file, which is what makes the family hard
# to defeat: a window can be narrowed by an edit and still look healthy, a whole
# file cannot, so a wildcard bind that lands outside every window is still
# reported, just with a less specific label. Both installers are clean today
# because their opt-in text is prose rather than a runnable command; if that ever
# stops being true, prefer rephrasing the prose over deleting these two.
assert_no_wildcard_bind \
    "install.sh: no launch command anywhere in the file binds a wildcard host" \
    "$(cat "$INSTALL_SH")"
assert_no_wildcard_bind \
    "install.ps1: no launch command anywhere in the file binds a wildcard host" \
    "$(cat "$INSTALL_PS1")"

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
