# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio_cpp_spec_packages against real v0.9.5 model_specs kept in tests/fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from core.inference import audio_cpp_spec_packages as sp

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "audio_cpp_package_specs.json"
SPECS = json.loads(FIXTURE.read_text(encoding = "utf-8"))
ZONOS_PTQ = ["gguf/backbone-ptq.gguf", "gguf/zac-keep2.gguf", "gguf/spk-encoder.gguf"]


def _pkgs(family):
    return sp.spec_packages(SPECS[family])


def _parse(*packages, **extra):
    return sp.spec_packages({"family": "test", "packages": list(packages), **extra})


def _pkg(package_id, files, **extra):
    return {"id": package_id, "format": "gguf", "files": files, **extra}


def test_zonos3_packages_strip_prefix_and_primary():
    main, raw = _pkgs("zonos3")
    assert (main.id, main.repo, main.strip_prefix) == (
        "zonos3_mini_ptq",
        "Zyphra/ZONOS3-Mini",
        "gguf",
    )
    assert list(main.files) == ZONOS_PTQ
    # The raw reference names two more GGUFs; the backbone is still the primary.
    assert main.primary == raw.primary == "gguf/backbone-ptq.gguf"


def test_packages_are_scoped_to_their_repo_and_folder():
    lfm = _pkgs("lfm2_audio")
    base = sp.packages_for_repo(lfm, "liquidai/lfm2.5-audio-1.5b-gguf", "")
    assert set(sp.variant_keys(base).values()) == {"Q8_0", "F16", "Q4_0"}
    # audiocpp_server refuses a directory of several GGUFs, so the first one is handed over.
    assert base[0].primary == "LFM2.5-Audio-1.5B-Q8_0.gguf"
    assert sp.packages_for_repo(lfm, "Nobody/Nothing", "") == ()
    zonos = _pkgs("zonos3")
    assert len(sp.packages_for_repo(zonos, "zyphra/zonos3-mini", "", folder = "gguf")) == 2
    assert sp.packages_for_repo(zonos, "zyphra/zonos3-mini", "", folder = "other") == ()


def test_packages_without_repo_live_in_umbrella():
    packages = _parse(
        _pkg("a", ["a.gguf"]),
        _pkg("b", ["b.gguf"], download = {"repo": "Own/Repo"}),
        package_defaults = {"download": {"repo": "Umbrella/Repo"}},
    )
    assert {p.id: p.repo for p in packages} == {"a": "Umbrella/Repo", "b": "Own/Repo"}
    assert [p.id for p in sp.packages_for_repo(packages, "Umbrella/Repo", "Umbrella/Repo")] == ["a"]


def test_malformed_input_is_dropped_and_never_raises():
    for spec in (None, {}, [], "spec", 3, {"packages": "x"}, {"packages": [None, 7, "a"]}):
        assert sp.spec_packages(spec) == ()
    assert sp.packages_for_repo(None, "a/b", "a/b") == ()
    assert sp.available(None, ["x"]) == ()
    assert sp.available(_pkgs("zonos3"), None) == ()
    assert sp.variant_keys(None) == {}
    packages = _parse(
        {"format": "gguf", "files": ["a.gguf"]},
        _pkg("not_list", "a.gguf"),
        _pkg("non_str", [1]),
        _pkg("empty", []),
        {"id": "other_format", "format": "safetensors", "files": ["e.bin"]},
        _pkg("ok", ["ok.gguf"]),
    )
    assert [p.id for p in packages] == ["ok"]


def test_files_outside_strip_prefix_or_escaping_drop_package():
    packages = _parse(
        _pkg("outside", ["gguf/a.gguf", "other/b.gguf"], strip_prefix = "gguf"),
        _pkg("prefix_sibling", ["gguf2/x.gguf"], strip_prefix = "gguf"),
        _pkg("escape", ["gguf/../x.gguf"], strip_prefix = "gguf"),
        _pkg("backslash_escape", ["gguf\\..\\x.gguf"], strip_prefix = "gguf"),
        _pkg("dotdot_first", ["../x.gguf"]),
        _pkg("absolute", ["/etc/x.gguf"]),
        _pkg("inside", ["gguf/a.gguf", "gguf/b.json"], strip_prefix = "gguf"),
    )
    assert [p.id for p in packages] == ["inside"]


def test_licenses_and_checksums_are_not_package_files():
    # Kitten TTS 2 lists LICENSE, NOTICE and SHA256SUMS beside its one GGUF: still a one-file package.
    packages = _parse(
        _pkg("a", ["a.gguf", "LICENSE", "NOTICE", "native/LICENSE", "SHA256SUMS", "README.md"]),
        _pkg("only_docs", ["LICENSE", "README.md"]),
    )
    assert [(p.id, p.files) for p in packages] == [("a", ("a.gguf",))]


def test_variant_key_is_upper_precision_when_unique_else_id():
    packages = _parse(
        _pkg("a", ["a.gguf"], precision = "Q8_0"),
        _pkg("b", ["b.gguf"], precision = "q8_0"),
        _pkg("c", ["c.gguf"], precision = "f16"),
        _pkg("d", ["d.gguf"]),
    )
    assert sp.variant_keys(packages) == {"a": "a", "b": "b", "c": "F16", "d": "d"}


def test_available_needs_every_file_and_orders_recommended_default_first():
    zonos = _pkgs("zonos3")
    assert [p.id for p in sp.available(zonos, ZONOS_PTQ)] == ["zonos3_mini_ptq"]
    names = [f.removeprefix("gguf/") for f in ZONOS_PTQ]
    assert [p.id for p in sp.available(zonos, names, folder = "gguf")] == ["zonos3_mini_ptq"]
    assert sp.available(zonos, names, folder = "other") == ()
    packages = _parse(
        _pkg("a", ["a.gguf"]),
        _pkg("b", ["b.gguf"], default = True),
        _pkg("c", ["c.gguf"]),
        ui = {"recommended_package": "c"},
    )
    assert [p.id for p in sp.available(packages, ["a.gguf", "b.gguf", "c.gguf"])] == ["c", "b", "a"]
