#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Reading a .ps1 as text, for the suites that assert on installer structure.
# Sourced, never executed: both runners glob test_*.sh, so this name sits
# outside the glob.
#
# The problem this exists for: our .ps1 files emit other scripts through
# here-strings. studio/setup.ps1 emits a probe whose body contains `exit 1`, and
# install.ps1 emits a launcher containing nine `function X {` and two
# `} finally {`. A grep over the raw file counts those as the installer's own
# code and answers the wrong question -- #10540 turned
# test_tauri_retry_failure_context.sh red that way without changing one line of
# installer control flow.
#
# ps1_code blanks here-string bodies, keeping one blank line per body line so
# `grep -n` line numbers and `sed` line addresses still refer to the real file.

# Blank the body of every here-string in $1. The opener is detected on the line
# with its quoted strings blanked first: both setup.ps1 and install.ps1 carry a
# credential-redaction `-replace ... , '$1<redacted>@'` whose line genuinely ends
# in `@'`, and a scanner that misses that swallows the ~780 lines up to the next
# terminator. The terminator must sit in column 0 -- that is PowerShell's rule,
# and install.ps1 has indented `"@echo off",` array entries that a lenient match
# would close on.
ps1_code() {
    awk '
    function blank_strings(line,   out, i, c, n, q) {
        out = ""; q = ""; n = length(line)
        for (i = 1; i <= n; i++) {
            c = substr(line, i, 1)
            if (q == "") {
                # An unquoted # starts a line comment; nothing after it is code.
                if (c == "#") { break }
                if (c == "\"" || c == "'"'"'") { q = c }
                out = out c
            } else {
                # ` escapes the next character inside a double-quoted string.
                if (q == "\"" && c == "`") { i++; out = out "  "; continue }
                if (c == q) { q = "" }
                else c = " "
                out = out c
            }
        }
        return out
    }
    {
        if (delim != "") {
            closed = (index($0, delim) == 1)
            print ""
            if (closed) delim = ""
            next
        }
        code = blank_strings($0)
        if (code ~ /@'"'"'[ \t]*$/)     { delim = "'"'"'@"; print; next }
        if (code ~ /@"[ \t]*$/) { delim = "\"@"; print; next }
        print
    }
    END {
        if (delim != "") {
            print "ps1_code: unterminated here-string opened in " FILENAME > "/dev/stderr"
            exit 1
        }
    }
    ' "$1"
}
