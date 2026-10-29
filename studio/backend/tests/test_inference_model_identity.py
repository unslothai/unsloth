# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether two model names mean the same model, and plan_load agreeing with load_model on it."""

import os

import pytest

from core.inference import model_identity as mi
from studio_mcp import loading, placement
from studio_mcp.outputs import SlotModel

AUDIO = "audio-cpp/audio.cpp-gguf/Qwen3-TTS"


@pytest.mark.parametrize(
    "name, local",
    [
        ("unsloth/Qwen3-0.6B-GGUF", False),
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", False),
        (AUDIO, False),
        ("/models/Foo.gguf", True),
        ("~/models/Foo", True),
        ("./Foo.gguf", True),
        ("../Foo", True),
        ("C:\\models\\a.gguf", True),
        ("C:/models/a.gguf", True),
        ("models\\Foo.gguf", True),
        ("\\\\host\\share\\a.gguf", True),
        ("ref:0123abcd", True),
        ("models/Foo.gguf", False),
        ("", False),
        (None, False),
    ],
)
def test_is_local(name, local):
    assert mi.is_local(name) is local


@pytest.mark.parametrize(
    "name, split",
    [
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", ("unsloth/Qwen3-0.6B-GGUF", "Q4_K_M")),
        ("unsloth/Qwen3-0.6B-GGUF", ("unsloth/Qwen3-0.6B-GGUF", None)),
        ("unsloth/Qwen3-0.6B-GGUF:", ("unsloth/Qwen3-0.6B-GGUF", None)),
        (f"{AUDIO}:Q8_0", (AUDIO, "Q8_0")),
        ("org/repo:Q4_K_M-3.97bpw", ("org/repo", "Q4_K_M-3.97bpw")),
        ("C:\\models\\a.gguf", ("C:\\models\\a.gguf", None)),
        ("C:/models/a.gguf", ("C:/models/a.gguf", None)),
        ("/models/a:b.gguf", ("/models/a:b.gguf", None)),
        ("ref:0123abcd", ("ref:0123abcd", None)),
        ("org/repo:sub/dir", ("org/repo:sub/dir", None)),
    ],
)
def test_split_variant(name, split):
    assert mi.split_variant(name) == split


@pytest.mark.parametrize(
    "left, right, model, repo",
    [
        # Hub ids fold case.
        ("unsloth/Qwen3-0.6B-GGUF", "UNSLOTH/qwen3-0.6b-gguf", True, True),
        (" unsloth/Qwen3 ", "unsloth/qwen3", True, True),
        # repo:variant is a variant of the bare repo, not the same model.
        ("unsloth/Qwen3-GGUF:Q4_K_M", "unsloth/qwen3-gguf", False, True),
        ("unsloth/Qwen3-GGUF:Q4_K_M", "unsloth/Qwen3-GGUF:Q8_0", False, True),
        ("unsloth/Qwen3-GGUF:q4_k_m", "unsloth/Qwen3-GGUF:Q4_K_M", True, True),
        ("org/repo:Q4_K_M-3.97bpw", "org/repo:Q4_K_M-4.19bpw", False, True),
        ("unsloth/Qwen3-GGUF", "unsloth/Gemma-GGUF", False, False),
        # audio.cpp ids are Hub ids, three segments and all.
        (AUDIO, AUDIO.lower(), True, True),
        (f"{AUDIO}:Q8_0", AUDIO, False, True),
        # Local paths keep their case; either side being a path makes it a path compare.
        ("/models/Foo.gguf", "/models/Foo.gguf", True, True),
        ("/models/Foo.gguf", "/models/foo.gguf", os.name == "nt", os.name == "nt"),
        ("/models/Foo/", "/models/Foo", True, True),
        ("~/models/Foo", "~/models/foo", os.name == "nt", os.name == "nt"),
        ("C:\\models\\a.gguf", "C:/models/a.gguf", True, True),
        # A drive letter is not a variant.
        ("C:\\models\\a.gguf", "C:\\models\\b.gguf", False, False),
        ("ref:0123abcd", "ref:0123abcd", True, True),
        ("ref:0123abcd", "ref:99", False, False),
        ("", "", False, False),
        (None, "unsloth/Qwen3", False, False),
    ],
)
def test_same_model_and_repo(left, right, model, repo):
    assert mi.same_model(left, right) is model
    assert mi.same_model(right, left) is model
    assert mi.same_repo(left, right) is repo
    assert mi.same_repo(right, left) is repo


def test_matches_any_skips_empty_candidates():
    assert mi.matches_any("Org/Repo", (None, "", "org/repo"))
    assert not mi.matches_any("Org/Repo", (None, ""))


@pytest.mark.parametrize(
    "variant, filename, settled, result",
    [
        ("Q4_K_M", "model-Q4_K_M.gguf", True, True),
        ("q4_k_m", "model-Q4_K_M.gguf", False, True),
        ("Q8_0", "model-Q4_K_M.gguf", True, False),
        (None, "model-Q4_K_M.gguf", True, False),
        # Status drops the bpw suffix: settled it is this build, mid-load a sibling cannot be told apart.
        ("IQ4_XS", "model-IQ4_XS-3.97bpw.gguf", True, True),
        ("IQ4_XS", "model-IQ4_XS-3.97bpw.gguf", False, False),
        ("Q4_K_M", "model.safetensors", True, None),
        ("Q4_K_M", "model.gguf", True, None),
    ],
)
def test_resident_quant_matches(variant, filename, settled, result):
    assert mi.resident_quant_matches(variant, filename, settled = settled) is result


def test_an_existing_relative_path_keeps_its_case_when_probed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Foo.gguf").write_bytes(b"")
    assert mi.is_local("models/Foo.gguf", probe = True)
    assert not mi.is_local("models/Foo.gguf")
    assert not mi.is_local("unsloth/Qwen3", probe = True)
    # Either side existing makes it a path compare, whichever order they come in.
    assert mi.same_model("models/Foo.gguf", "models/foo.gguf", probe = True) is (os.name == "nt")
    assert mi.same_model("models/foo.gguf", "models/Foo.gguf", probe = True) is (os.name == "nt")
    assert mi.same_model("models/Foo.gguf", "models/foo.gguf")


def test_plan_load_keeps_the_case_of_an_existing_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Foo.gguf").write_bytes(b"")
    (tmp_path / "models" / "foo.gguf").write_bytes(b"")
    if os.path.samefile(tmp_path / "models" / "Foo.gguf", tmp_path / "models" / "foo.gguf"):
        pytest.skip("case-insensitive filesystem")
    # load_model reads them as two files, so it would not replace one with the other in place.
    assert not loading._same_id("models/Foo.gguf", "models/foo.gguf")
    _, _, refuse = placement.predict(
        kind = "llm",
        model = "models/Foo.gguf",
        device = None,
        alongside = None,
        mode = "always",
        status = {},
        slots = [SlotModel(kind = "llm", model = "models/foo.gguf")],
    )
    assert refuse
    others, exact, _ = placement.predict(
        kind = "stt",
        model = "models/Foo.gguf",
        device = None,
        alongside = None,
        mode = None,
        status = {},
        slots = [SlotModel(kind = "stt", model = "models/foo.gguf")],
    )
    assert [s.model for s in others] == ["models/foo.gguf"] and not exact


def test_load_and_plan_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Foo.gguf").write_bytes(b"")
    pairs = [
        ("models/Foo.gguf", "models/foo.gguf"),
        ("unsloth/Qwen3-GGUF:Q4_K_M", "UNSLOTH/qwen3-gguf"),
        ("/m/Foo", "/m/foo"),
        ("C:\\m\\a.gguf", "C:\\m\\b.gguf"),
        (AUDIO, AUDIO.upper()),
    ]
    for left, right in pairs:
        _, _, refuse = placement.predict(
            kind = "llm",
            model = left,
            device = None,
            alongside = None,
            mode = "always",
            status = {},
            slots = [SlotModel(kind = "llm", model = right)],
        )
        assert refuse is not loading._same_id(left, right), (left, right)


POSIX = os.name != "nt"


def test_route_local_path_test_keeps_its_old_spelling_rule():
    import routes.inference as route

    # Kept as it was: a dot-relative or forward-slash drive name is not vouched for as a path here.
    for name, path in (
        ("/m/Foo", True),
        ("~/m", True),
        ("C:\\m\\a.gguf", True),
        ("m\\a.gguf", True),
        ("./m/Foo", False),
        ("C:/m/a.gguf", False),
        ("org/repo", False),
        (None, False),
    ):
        assert route._looks_like_local_path(name) is path, name


def test_route_name_match_keeps_a_local_paths_case():
    import routes.inference as route

    assert route._matches_any("Org/Repo", (None, "org/repo"))
    assert route._matches_any("/m/Foo/", ("/m/Foo",))
    assert route._matches_any("C:\\m\\a.gguf", ("C:/m/a.gguf",))
    assert route._matches_any("./m/Foo", ("./m/foo",)) is not POSIX
    assert route._matches_any("C:/m/Foo", ("C:/m/foo",)) is not POSIX
    assert route._matches_any("ref:ABC", ("ref:abc",)) is not POSIX
    assert route._norm_path(" /m/Foo/ ") == os.path.normcase("/m/Foo").replace("\\", "/")
    assert route._norm_path(None) == ""


def test_loaded_identity_satisfies_keeps_a_relative_paths_case(monkeypatch):
    import routes.inference as route
    from types import SimpleNamespace

    backend = SimpleNamespace(active_model_name = "./m/Foo", _openai_advertised_id = "./m/Foo")
    monkeypatch.setattr(route, "_resident_npu_model", lambda: None)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False))
    monkeypatch.setattr(route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(route, "_loaded_satisfies", lambda requested: True)
    assert route._loaded_identity_satisfies("./m/Foo")
    assert route._loaded_identity_satisfies("./m/foo") is not POSIX
    backend.active_model_name = backend._openai_advertised_id = "Org/Repo"
    assert route._loaded_identity_satisfies("org/repo")


def test_loading_matches_drive_relative_and_ref_names_as_paths():
    assert loading._same("/m/Foo/", "/m/Foo")
    assert loading._same("C:x", "C:x")
    assert loading._same_id("C:x", "C:X") is not POSIX
    assert loading._same_id("ref:ABC", "ref:abc") is not POSIX
    # A colon in a file name is not a variant.
    assert not loading._same_id("/models/a:b.gguf", "/models/a:c.gguf")
    assert loading._same_id("Org/Repo:Q4_K_M", "org/repo")


def test_resident_media_check_still_folds_case():
    # Reverted to the old rule: status may spell the resident path differently from the request.
    status = {"loaded": True, "repo_id": "/m/foo", "display_repo_id": None}
    assert loading._resident_matches(status, "/m/Foo")
    assert loading._resident_matches({"loaded": True, "repo_id": "Org/Repo"}, "org/repo")
    assert not loading._resident_matches({"loaded": True, "repo_id": "/m/bar"}, "/m/Foo")


@pytest.mark.parametrize(
    "model, polls",
    [
        ("unsloth/Qwen3-0.6B-GGUF", True),
        ("unsloth/Qwen3-0.6B-GGUF:Q4_K_M", True),
        (AUDIO, True),
        ("ab:x", True),
        ("/m/Foo.gguf", False),
        ("./m/Foo.gguf", False),
        ("~/m/Foo.gguf", False),
        ("C:\\m\\a.gguf", False),
        ("ref:0123abcd", False),
    ],
)
def test_llm_poll_reads_hub_download_progress_only_for_hub_names(monkeypatch, model, polls):
    import asyncio

    asked = []

    async def route_json(method, path, **kwargs):
        asked.append(path)
        return {}

    monkeypatch.setattr(loading, "route_json", route_json)
    asyncio.run(loading._llm_poll(None, model)())
    assert ("/api/models/download-progress" in asked) is polls


@pytest.mark.parametrize(
    "name, local, split",
    [
        ("ab:x", False, ("ab", "x")),
        ("1:x", False, ("1", "x")),
        ("C:x", True, ("C:x", None)),
        ("C:/m/a.gguf", True, ("C:/m/a.gguf", None)),
        ("m\\a.gguf", True, ("m\\a.gguf", None)),
        ("~/m/a.gguf", True, ("~/m/a.gguf", None)),
        ("ref:0123abcd", True, ("ref:0123abcd", None)),
        ("unsloth/Qwen3-GGUF:Q4_K_M", False, ("unsloth/Qwen3-GGUF", "Q4_K_M")),
    ],
)
def test_placement_and_downloads_read_local_names_alike(name, local, split):
    from studio_mcp import model_downloads

    assert placement.is_local(name) is local
    assert model_downloads._looks_local(name) is local
    assert placement._split(name, None) == split
    assert model_downloads._split(name, None) == split


def test_downloads_skip_manager_jobs_for_a_home_path():
    from studio_mcp import model_downloads
    assert model_downloads.keys_for(None, "llm", "~/m/a.gguf", None) == []


def test_predict_keeps_a_slashed_suffix_as_part_of_the_repo():
    _, _, refuse = placement.predict(
        kind = "llm",
        model = "org/repo:sub/dir",
        device = None,
        alongside = None,
        mode = "always",
        status = {},
        slots = [SlotModel(kind = "llm", model = "org/repo")],
    )
    assert refuse
