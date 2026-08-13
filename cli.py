# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys

# unsloth_cli only runs its console-script setup (stream encoding, `-np<N>`,
# the Windows folder guard) for argv[0] == "unsloth", and it must precede the
# command imports: commands.studio resolves STUDIO_HOME at import time.
if __name__ == "__main__":
    sys.argv[0] = "unsloth"

from unsloth_cli import app

if __name__ == "__main__":
    app()
