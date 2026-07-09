# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unit tests for the pure constant folder used by the sandbox classifier."""

import ast
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _build_const_prop_env, _const_fold


def _fold(expr: str, env = None):
    return _const_fold(ast.parse(expr, mode = "eval").body, env = env)


class TestConstFoldLiterals:
    def test_string_constant(self):
        assert _fold('"hello"') == "hello"

    def test_int_constant(self):
        assert _fold("42") == 42

    def test_bytes_constant(self):
        assert _fold('b"abc"') == b"abc"

    def test_none(self):
        assert _fold("None") is None


class TestConstFoldArithAndConcat:
    def test_str_concat(self):
        assert _fold('"os" + "." + "system"') == "os.system"

    def test_str_repeat(self):
        assert _fold('"ab" * 3') == "ababab"

    def test_bytes_concat(self):
        assert _fold('b"a" + b"b"') == b"ab"

    def test_int_add(self):
        assert _fold("2 + 2") == 4

    def test_percent_format(self):
        assert _fold('"os.%s" % "system"') == "os.system"

    def test_huge_repeat_refused(self):
        assert _fold('"x" * (10 ** 8)') is None

    def test_pow_refused(self):
        assert _fold("2 ** 4") is None


class TestConstFoldJoinFormatFstring:
    def test_sep_join(self):
        assert _fold('".".join(["os", "system"])') == "os.system"

    def test_str_format(self):
        assert _fold('"{}.{}".format("os", "system")') == "os.system"

    def test_fstring(self):
        assert _fold('f"{2 + 2}"') == "4"

    def test_fstring_all_const(self):
        assert _fold("f\"import {'os'}\"") == "import os"


class TestConstFoldEncodeDecodeBaseHex:
    def test_encode(self):
        assert _fold('"abc".encode("utf-8")') == b"abc"

    def test_decode(self):
        assert _fold('b"abc".decode()') == "abc"

    def test_b64decode(self):
        assert _fold('base64.b64decode("aW1wb3J0IG9z")') == b"import os"

    def test_urlsafe_b64decode(self):
        assert _fold('base64.urlsafe_b64decode("aW1wb3J0IG9z")') == b"import os"

    def test_bytes_fromhex(self):
        assert _fold('bytes.fromhex("696d706f7274")') == b"import"

    def test_binascii_unhexlify(self):
        assert _fold('binascii.unhexlify("6f73")') == b"os"

    def test_codecs_rot13(self):
        assert _fold('codecs.decode("vzcbeg bf", "rot_13")') == "import os"

    def test_codecs_hex(self):
        assert _fold('codecs.decode("6f73", "hex")') == b"os"


class TestConstFoldCharOrdSliceReverse:
    def test_chr_concat(self):
        assert _fold("chr(50) + chr(43) + chr(50)") == "2+2"

    def test_ord(self):
        assert _fold('ord("A")') == 65

    def test_reverse_slice(self):
        assert _fold('"tidbe"[::-1]') == "ebdit"

    def test_slice(self):
        assert _fold('"abcdef"[1:3]') == "bc"


class TestConstFoldContainers:
    def test_list(self):
        assert _fold("[1, 2, 3]") == [1, 2, 3]

    def test_str_join_of_folded_chr(self):
        assert _fold('"".join([chr(111), chr(115)])') == "os"


class TestConstFoldUnknown:
    def test_bare_name_unknown(self):
        assert _fold("x") is None

    def test_call_unknown(self):
        assert _fold("requests.get(url)") is None

    def test_pickle_never_folds(self):
        assert _fold("pickle.loads(b'x')") is None

    def test_getattr_never_folds(self):
        assert _fold('getattr(os, "system")') is None


class TestConstPropEnv:
    def test_single_assignment_folds(self):
        tree = ast.parse('p = "2 + 2"\nx = p')
        env = _build_const_prop_env(tree)
        assert "p" in env
        assert _const_fold(ast.parse("p", mode = "eval").body, env = env) == "2 + 2"

    def test_reassigned_name_excluded(self):
        tree = ast.parse('p = "safe"\np = "os.system"')
        env = _build_const_prop_env(tree)
        assert "p" not in env

    def test_loop_target_excluded(self):
        tree = ast.parse("for p in range(3):\n    pass")
        env = _build_const_prop_env(tree)
        assert "p" not in env

    def test_concat_prop(self):
        tree = ast.parse('p = "os.system(\'rm -rf /\')"\ny = "import os; " + p')
        env = _build_const_prop_env(tree)
        folded = _const_fold(ast.parse('"import os; " + p', mode = "eval").body, env = env)
        assert folded == "import os; os.system('rm -rf /')"
