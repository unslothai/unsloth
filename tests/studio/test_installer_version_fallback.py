# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Order duplicate distributions correctly when packaging is unavailable.

The post-update verifier picks among leftover .dist-info directories with max(_vkey).
Without packaging it falls back to parsing the version itself, so that fallback has to
order suffixes fully: ranking .post1 and .post2 both as "post" tied their keys and max()
then kept whichever enumerated first, letting the verifier confirm a stale install.
"""

import re
import sys
from pathlib import Path

import pytest

STUDIO = Path(__file__).resolve().parents[2] / "studio"
BLOCK = re.compile(r"^def _vkey\(c\):\n(?:[ \t].*\n|\n)*", re.MULTILINE)


def _extract(path):
    m = BLOCK.search(path.read_text(encoding = "utf-8"))
    assert m, f"_vkey is gone from {path.name}"
    return m.group(0)


def _load():
    ns = {"re": re}
    exec(_extract(STUDIO / "setup.sh"), ns)
    vkey = ns["_vkey"]

    class Dist:
        def __init__(self, v):
            self.version = v

    def key(v):
        # A None entry makes the import inside _vkey raise, forcing the fallback path.
        saved = sys.modules.get("packaging.version", ...)
        sys.modules["packaging.version"] = None
        try:
            return vkey((Dist(v),))
        finally:
            if saved is ...:
                del sys.modules["packaging.version"]
            else:
                sys.modules["packaging.version"] = saved

    return key


ASCENDING = [
    "1.0.dev1",
    "1.0.dev9",
    "1.0a1",
    "1.0a2",
    "1.0b1",
    "1.0rc1",
    "1.0",
    "1.0.post1",
    "1.0.post2",
    "1.1",
]


def test_suffix_number_breaks_the_tie_between_duplicate_metadata():
    key = _load()
    assert key("1.0.post1") != key("1.0.post2")
    # max() keeps the first maximal element, so a tie let enumeration order decide.
    for order in (["1.0.post1", "1.0.post2"], ["1.0.post2", "1.0.post1"]):
        assert max(order, key = key) == "1.0.post2"


def test_fallback_orders_every_pep440_stage():
    key = _load()
    for lo, hi in zip(ASCENDING, ASCENDING[1:]):
        assert key(lo) < key(hi), f"{lo} must sort below {hi}"


def test_every_pep440_separator_reaches_the_suffix_number():
    key = _load()
    # PEP 440 allows . - _ between a stage and its number, all the same release. Reading
    # only a dot dropped the number, tying post-1 and post-2 back to post0.
    for sep in (".", "-", "_", ""):
        one, two = f"1.0.post{sep}1", f"1.0.post{sep}2"
        assert key(one) != key(two), f"post{sep}1 and post{sep}2 must not tie"
        assert max([one, two], key = key) == two
        assert key(one) == key("1.0.post1")
        assert key(f"1.0a{sep}2") == key("1.0a2")
        assert key(f"1.0.dev{sep}2") == key("1.0.dev2")


def test_epoch_outranks_a_higher_release():
    key = _load()
    # PEP 440: an epoch beats the release segment outright, so 1!0.1 is newer than 2.0.
    assert key("1!0.1") > key("2.0")
    assert key("2!0.1") > key("1!9.9")
    assert key("1!1.0") == key("1!1.0.0")
    assert max(["2.0", "1!0.1"], key = key) == "1!0.1"


def test_equivalent_stage_spellings_agree():
    key = _load()
    assert key("1.0alpha1") == key("1.0a1")
    assert key("1.0beta1") == key("1.0b1")
    assert key("1.0rc1") == key("1.0c1") == key("1.0pre1")
    # PEP 440: trailing zero segments do not outrank a suffix.
    assert key("1.0.0") == key("1.0")


def test_unparsable_versions_stay_comparable():
    key = _load()
    # "0" strips to an empty release, so it meets a non-numeric version on equal terms;
    # an int rank against a tuple rank would raise instead of sorting.
    assert key("0") < key("junk") or key("junk") < key("0")
    assert key("junk") < key("1.0")


@pytest.mark.parametrize("script", ["setup.sh", "setup.ps1"])
def test_both_installers_carry_the_same_fallback(script):
    assert _extract(STUDIO / script) == _extract(STUDIO / "setup.sh")
