# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys

# unsloth_cli reserves its console-script setup (stream encoding, the `-np<N>`
# rewrite, the Windows system-folder guard) for argv[0] == "unsloth", so that it
# never reaches a host application that merely imports it. Run directly, this
# file *is* that entry point, and the guard has to run before the command modules
# are imported: unsloth_cli.commands.studio resolves STUDIO_HOME at import time.
if __name__ == "__main__":
    sys.argv[0] = "unsloth"

from unsloth_cli import app

if __name__ == "__main__":
    app()
