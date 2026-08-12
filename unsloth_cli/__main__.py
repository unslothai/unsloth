# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`python -m unsloth_cli` -- reach the CLI without the generated console script.

Windows materialises the `unsloth` entry point as a generated, unsigned
`unsloth.exe`. AppLocker, WDAC and Smart App Control deny that executable while
the signed interpreter beside it keeps running, so `unsloth ...` is unusable on
a locked-down machine even though the install is perfectly healthy (issue
https://github.com/unslothai/unsloth/issues/8490). This module is the supported
way in for those users, and for anyone who would rather not depend on a
generated launcher:

    python -X utf8 -m unsloth_cli studio -p 8888

Output is identical to the console script, which takes three things:

  * argv[0] is rewritten, so anything reading it sees the name the console
    script would have given it.
  * _prepare_entry_point() applies the rest of the console-script behaviour
    (UTF-8 streams, the `-np<N>` rewrite). The import-time gate in
    unsloth_cli/__init__ cannot do it here, because `-m` imports the package in
    order to locate this module: __init__ has already run, with argv[0] still
    "-m", before the first statement below executes.
  * prog_name is passed explicitly. click ignores argv[0] once it sees a
    __main__ with a __package__ (its _detect_program_name treats that as "python
    -m example") and would print `Usage: python -m unsloth_cli` in every usage
    and error message.
"""

import sys

# Before the import, so a direct `python path/to/unsloth_cli/__main__.py` run
# takes the console-script gate in __init__ rather than needing the call below.
sys.argv[0] = "unsloth"

import unsloth_cli  # noqa: E402

unsloth_cli._prepare_entry_point()
unsloth_cli.app(prog_name = "unsloth")
