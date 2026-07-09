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

    def test_small_bytes_count_folds(self):
        from core.inference.tools import _FOLD_MAXLEN

        assert _fold("bytes(10)") == b"\x00" * 10
        assert _fold("bytes(b'abc')") == b"abc"
        assert _fold(f"bytes({_FOLD_MAXLEN})") == b"\x00" * _FOLD_MAXLEN

    def test_huge_bytes_count_refused(self):
        # bytes(N) / bytearray(N) allocate N zero bytes; an oversized count is a
        # memory-DoS during folding, so it must refuse rather than materialize it.
        from core.inference.tools import _FOLD_MAXLEN

        assert _fold(f"bytes({_FOLD_MAXLEN + 1})") is None
        assert _fold(f"bytearray({_FOLD_MAXLEN + 1})") is None
        assert _fold("bytes(10 ** 9)") is None


class TestConstFoldAllocationDoS:
    """Oversized format widths / sequence repetitions must refuse BEFORE the folder
    allocates the result (folding runs in the Studio process, ahead of subprocess
    rlimits)."""

    def test_fstring_width_refused(self):
        assert _fold("f'{1:1000000000}'") is None

    def test_str_format_width_refused(self):
        assert _fold("'{:1000000000}'.format(1)") is None

    def test_percent_format_width_refused(self):
        assert _fold("'%1000000000d' % 1") is None

    def test_pad_method_width_refused(self):
        assert _fold("'x'.ljust(1000000000)") is None
        assert _fold("'x'.rjust(10 ** 9)") is None
        assert _fold("'x'.center(2000000000)") is None
        assert _fold("'x'.zfill(10 ** 9)") is None

    def test_list_tuple_repeat_refused(self):
        assert _fold("[0] * 1000000000") is None
        assert _fold("(1,) * 10 ** 9") is None

    def test_benign_format_and_repeat_still_fold(self):
        assert _fold("f'{2 + 2}'") == "4"
        assert _fold("'{:>8}'.format('hi')") == "      hi"
        assert _fold("'%05d' % 7") == "00007"
        assert _fold("'x'.ljust(10)") == "x         "
        assert _fold("[0] * 8") == [0] * 8


class TestConstFoldPathJoin:
    """os.path.join / posixpath.join of string literals fold so the sensitive-read
    scanner sees the concrete path (628)."""

    def test_os_path_join_literal(self):
        assert _fold("os.path.join('/etc', 'passwd')") == "/etc/passwd"

    def test_posixpath_join_literal(self):
        assert _fold("posixpath.join('/etc', 'shadow')") == "/etc/shadow"

    def test_relative_join_literal(self):
        assert _fold("os.path.join('sub', 'a.txt')") == "sub/a.txt"

    def test_join_nonliteral_unknown(self):
        assert _fold("os.path.join('/etc', x)") is None


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
