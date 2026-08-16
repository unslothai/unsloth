# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A split that truncates its own rows may say so instead of being scanned.

`pretokenized_within_cap` decides whether `max_length` is already enforced by
reading EVERY row of the split. For a lazily-tokenizing split -- a
`datasets.with_transform` view, which is what Studio's online tokenization
produces -- reading a row IS tokenizing it, so that scan runs the whole eager
tokenize pass the view exists to avoid, inside `__init__` where nothing overlaps
it.

The attestation is the escape: `_unsloth_truncated_to = N` on the split means
"every row I yield is already cut at N". The two copies of the scan (this
module's, and the one `rl.py` inlines into every generated trainer) must agree
on it, so both are exercised here -- the inlined one by extracting it from the
codegen string and executing it, which is the only way to test source that only
exists as a string.
"""

from __future__ import annotations

import re
import sys
import textwrap
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RL_PATH = REPO_ROOT / "unsloth" / "models" / "rl.py"


def _load_rl_module():
    """`unsloth.models.rl` without importing unsloth (torch, CUDA, the lot).

    Only the pure helpers are needed, and they are stdlib-only.
    """
    import ast

    source = RL_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    wanted = {
        "_attested_within_cap",
        "pretokenized_within_cap",
        "splits_within_cap",
        "_SCAN_ROWS",
        "_TRUNCATION_ATTESTATION_ATTR",
    }
    kept = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        )
    ]
    module = types.ModuleType("rl_helpers_under_test")
    exec(compile(ast.Module(body = kept, type_ignores = []), str(RL_PATH), "exec"), module.__dict__)
    return module


rl = _load_rl_module()


class _Lazy:
    """A split that rebuilds its rows on read, and counts the reads.

    Stands in for `with_transform`: same shape, no `datasets` dependency, and it
    can prove the scan did not happen.
    """

    def __init__(
        self,
        n,
        width,
        attest = None,
    ):
        self.n = n
        self.width = width
        self.reads = 0
        if attest is not None:
            self._unsloth_truncated_to = attest

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.reads += 1
        return {"input_ids": [1] * self.width}

    def __iter__(self):
        for i in range(self.n):
            yield self[i]


def test_an_attesting_split_is_believed_without_a_single_read():
    split = _Lazy(100_000, width = 2048, attest = 2048)
    assert rl.pretokenized_within_cap(split, 2048) is True
    assert split.reads == 0, "the attestation was ignored and the split was scanned"


def test_a_split_attesting_a_wider_cap_is_refused():
    """Truncated to 4096 proves nothing about a 2048 cap, and the refusal must
    not silently fall through to a scan that would find the same answer."""
    split = _Lazy(8, width = 4096, attest = 4096)
    assert rl.pretokenized_within_cap(split, 2048) is False
    assert split.reads == 0


def test_a_split_attesting_a_narrower_cap_is_accepted():
    split = _Lazy(8, width = 512, attest = 512)
    assert rl.pretokenized_within_cap(split, 2048) is True


def test_a_split_with_no_attestation_is_still_scanned():
    split = _Lazy(4, width = 8)
    assert rl.pretokenized_within_cap(split, 2048) is True
    assert split.reads == 4


def test_an_overlength_unattested_split_is_still_caught():
    split = _Lazy(4, width = 9000)
    assert rl.pretokenized_within_cap(split, 2048) is False


@pytest.mark.parametrize("claim", [True, False, "2048", 2048.0, None, object()])
def test_a_non_int_claim_is_not_an_attestation(claim):
    """`True` is an `int` in Python and would read as a cap of 1. Anything that
    is not a plain integer falls through to the scan."""
    split = _Lazy(4, width = 9000)
    if claim is not None:
        split._unsloth_truncated_to = claim
    assert rl.pretokenized_within_cap(split, 2048) is False


def test_the_claim_is_read_from_the_split_itself_not_through_a_wrapper():
    """`_CappedBase.__getattr__` forwards unknown names to the split inside, so
    a plain `getattr` would let an inner split's guarantee answer for a wrapper
    that carries none."""

    class _Forwarding:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __len__(self):
            return len(self._inner)

        def __iter__(self):
            for row in self._inner:
                yield {"input_ids": row["input_ids"] * 8}

    wrapper = _Forwarding(_Lazy(4, width = 4096, attest = 4096))
    assert rl._attested_within_cap(wrapper, 40960) is None


def test_splits_within_cap_honours_the_attestation_per_split():
    good = _Lazy(4, width = 2048, attest = 2048)
    bad = _Lazy(4, width = 4096, attest = 4096)
    assert rl.splits_within_cap({"a": good}, 2048) is True
    assert rl.splits_within_cap({"a": good, "b": bad}, 2048) is False


# ------------------------------------------------- the copy rl.py inlines


def _inlined_within_cap(cap):
    """Build `_unsloth_within_cap` out of the codegen string and return it.

    The generated trainer is a standalone module and cannot import from
    `rl.py`, so the scan exists twice. This extracts the string literals that
    make up the second copy and executes them, which is the only way to hold
    the two to the same verdict.
    """
    source = RL_PATH.read_text(encoding = "utf-8")
    start = source.index('"    def _unsloth_within_cap(_ds):\\n"')
    end = source.index('"    def _unsloth_splits_within_cap(_ev):\\n"')
    body = "".join(re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*$', source[start:end], re.MULTILINE))
    # The literals carry the indentation they will have inside the generated
    # `__init__`; strip it so the block can stand on its own here.
    body = textwrap.dedent(body.encode().decode("unicode_escape"))
    namespace = {"_unsloth_cap": cap}
    exec(body, namespace)
    return namespace["_unsloth_within_cap"]


@pytest.mark.parametrize(
    "attest, width, cap, expected, expected_reads",
    [
        (2048, 2048, 2048, True, 0),
        (512, 512, 2048, True, 0),
        (4096, 4096, 2048, False, 0),
        (None, 8, 2048, True, 4),
        (None, 9000, 2048, False, 1),
    ],
)
def test_the_inlined_copy_gives_the_same_verdict(attest, width, cap, expected, expected_reads):
    inlined = _inlined_within_cap(cap)
    split = _Lazy(4, width = width, attest = attest)
    assert inlined(split) is expected
    assert split.reads == expected_reads

    module_level = _Lazy(4, width = width, attest = attest)
    assert rl.pretokenized_within_cap(module_level, cap) is expected


def test_the_codegen_and_the_module_agree_on_the_attribute_name():
    """A rename on one side and not the other is a silent no-op, not a failure:
    the scan would simply never see an attestation again."""
    source = RL_PATH.read_text(encoding = "utf-8")
    assert rl._TRUNCATION_ATTESTATION_ATTR == "_unsloth_truncated_to"
    assert (
        source.count("'_unsloth_truncated_to'") >= 2
    ), "the codegen no longer reads the attribute the module writes"


def test_studio_stamps_the_attribute_this_scan_reads():
    """The producer and the consumer live in different packages; nothing but a
    test ties the string they share together."""
    studio = REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "online_tokenization.py"
    if not studio.exists():
        pytest.skip("studio backend not present in this checkout")
    text = studio.read_text(encoding = "utf-8")
    assert f'TRUNCATION_ATTESTATION_ATTR = "{rl._TRUNCATION_ATTESTATION_ATTR}"' in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------- the generated block must still parse


def test_the_generated_max_length_block_is_valid_python():
    """The block only exists as string literals, so a stray indent or an unclosed
    bracket is invisible until a user on a TRL that ships the `max_length` guard
    gets a `SyntaxError` out of a generated trainer. Assemble it and parse it.

    The literals are read off the source rather than by running the generator,
    which needs a TRL carrying the guard string this installation may not have.
    """
    import ast

    source = RL_PATH.read_text(encoding = "utf-8")
    start = source.index("            max_length_check = (")
    end = source.index("            extra_args += max_length_check")
    literals = re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*$', source[start:end], re.MULTILINE)
    block = "".join(literals).encode().decode("unicode_escape")
    # The generator emits this at one indent level inside the trainer's __init__.
    ast.parse(textwrap.dedent(block))


def test_the_attestation_branch_is_present_in_the_generated_block():
    """Pinned by name: without it a `with_transform` split loses padding-free and
    is scanned row by row, which is the eager tokenize pass it exists to avoid."""
    source = RL_PATH.read_text(encoding = "utf-8")
    assert "_unsloth_attests" in source
    assert "not _unsloth_prep_truncates and not _unsloth_eval_packing" in source
