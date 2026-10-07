#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Sourced helper. Our .ps1 files emit other scripts via here-strings, so a raw grep counts
# their bodies as installer code. ps1_code blanks them, keeping line numbers intact.

# Openers are detected with quoted strings blanked (a `-replace ... '$1<redacted>@'` line ends
# in `@'`), and terminators must sit in column 0, as PowerShell requires.
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
