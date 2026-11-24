# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth's transformers CVE back-ports for a process that parses a repo's files before, or
without, importing unsloth; stdlib-only, so a ``python -c`` child can load this file by path."""

import importlib.util
import os


def _load_import_fixes():
    """unsloth/import_fixes.py by path: importing unsloth would initialize the GPU stack."""
    spec = importlib.util.find_spec("unsloth")
    for root in list(spec.submodule_search_locations or []) if spec else []:
        path = os.path.join(root, "import_fixes.py")
        if os.path.isfile(path):
            module_spec = importlib.util.spec_from_file_location("_unsloth_import_fixes", path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module
    raise ImportError("unsloth/import_fixes.py was not found")


def install() -> None:
    """Call once the sidecar is on sys.path, before the first from_pretrained. Below their fix
    versions transformers runs a config.json's nested auto_map code (CVE-2026-5241, LightGlue) and
    opens a tokenizer_config.json's vocab_file path, both with trust_remote_code False. Idempotent,
    and a no-op on a fixed transformers; raises when the back-ports cannot be loaded, so the
    caller's parse fails instead of running unguarded."""
    fixes = _load_import_fixes()
    fixes.fix_transformers_untrusted_config_fields()
    fixes.fix_transformers_tokenizer_config_vocab_file_override()


def inline_install_source() -> str:
    """``_install_transformers_backports()`` as source, for a ``python -c`` child that cannot import
    backend modules."""
    return (
        "def _install_transformers_backports():\n"
        "    import importlib.util as _bp_util\n"
        f"    _bp_spec = _bp_util.spec_from_file_location('_studio_transformers_backports', {os.path.abspath(__file__)!r})\n"
        "    _bp_mod = _bp_util.module_from_spec(_bp_spec)\n"
        "    _bp_spec.loader.exec_module(_bp_mod)\n"
        "    _bp_mod.install()\n"
    )
