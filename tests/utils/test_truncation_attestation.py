# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A split that truncates its own rows may say so instead of being scanned.

`pretokenized_within_cap` checks `max_length` by reading every row. On a
lazily-tokenizing `with_transform` view -- what Unsloth's online tokenization
produces -- reading a row is tokenizing it, so the scan runs the whole eager pass
the view exists to avoid, inside `__init__` where nothing overlaps it.

`_unsloth_truncated_to = N` is the escape: every row is already cut at N. Both
copies of the scan (this module's and the one `rl.py` inlines into every
generated trainer) must agree, so both run here; the inlined one is extracted
from the codegen string and executed, the only way to test source that only
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
    """`unsloth.models.rl` without importing unsloth: only the pure, stdlib-only
    helpers are needed."""
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
    """A split that rebuilds its rows on read and counts the reads: stands in for
    `with_transform` without the `datasets` dependency, and can prove the scan
    did not happen."""

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


def _inlined_within_cap(cap):
    """Build `_unsloth_within_cap` out of the codegen string and return it.

    The generated trainer cannot import from `rl.py`, so the scan exists twice;
    extracting and executing the literals is the only way to hold both copies to
    the same verdict.
    """
    source = RL_PATH.read_text(encoding = "utf-8")
    start = source.index('"    def _unsloth_within_cap(_ds):\\n"')
    end = source.index('"    def _unsloth_splits_within_cap(_ev):\\n"')
    body = "".join(re.findall(r'^\s*"((?:[^"\\]|\\.)*)"\s*$', source[start:end], re.MULTILINE))
    # Strip the indentation the literals carry for the generated `__init__`.
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
# ------------------------------------------------- the copy rl.py inlines
def test_the_generated_max_length_block_is_valid_python():
    """The block only exists as string literals, so a stray indent or unclosed
    bracket stays invisible until a user gets a `SyntaxError` from a generated
    trainer. Assemble and parse it, reading the literals off the source rather
    than running the generator, which needs a TRL this install may not have.
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
