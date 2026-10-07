# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth install-kernels`, reached before typer is imported: notebooks install unsloth with --no-deps."""

import sys
from importlib.util import find_spec
from pathlib import Path


def main(argv):
    spec = find_spec("studio")
    roots = list(spec.submodule_search_locations or []) if spec else []
    roots.append(str(Path(__file__).resolve().parent.parent / "studio"))
    backend = next(
        (
            Path(r) / "backend"
            for r in roots
            if (Path(r) / "backend" / "utils" / "kernel_install.py").is_file()
        ),
        None,
    )
    if backend is None:
        print(
            "Unsloth: studio/backend is missing from this install; cannot resolve kernel wheels.",
            file = sys.stderr,
        )
        return 1
    # The backend imports its siblings as top-level `utils.*`.
    sys.path.insert(0, str(backend))
    from utils.kernel_install import main as install_kernels_main

    return install_kernels_main(argv)
