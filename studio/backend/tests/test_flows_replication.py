# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exports record what made them (Studio version, model revisions and GGUF files from the local cache), and an
import says where this Studio differs. A fake Hugging Face cache stands in for the real one."""

import json

from core.flows import package, replication

from ._flows_helpers import doc, node, run_async

QWEN = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
FLUX = "black-forest-labs/FLUX.2-klein-4B"
SHA_A = "a" * 40
SHA_B = "b" * 40


def _cache(
    root,
    repo,
    sha,
    files = (),
):
    d = root / ("models--" + repo.replace("/", "--"))
    (d / "refs").mkdir(parents = True)
    (d / "refs" / "main").write_text(sha)
    snap = d / "snapshots" / sha
    snap.mkdir(parents = True)
    for name in files:
        (snap / name).parent.mkdir(parents = True, exist_ok = True)
        (snap / name).write_bytes(b"")
    return root


def _flow():
    return doc(
        [
            node("chat", "llm.generate", model = {"repo": QWEN, "variant": "Q4_K_M"}, prompt = "hi"),
            node("img", "image.generate", model = {"repo": FLUX, "variant": None}, prompt = "a fox"),
            node(
                "local", "llm.generate", model = {"repo": "/home/me/models/secret", "variant": None}
            ),
        ],
        requirements = {
            "models": [{"ref": "chat", "kind": "chat", "repo": QWEN, "variant": "Q4_K_M"}]
        },
    )


def test_pins_carry_the_cached_revision_and_gguf_file_but_never_a_local_path(tmp_path):
    cache = _cache(
        tmp_path,
        QWEN,
        SHA_A,
        ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "Qwen3-4B-Instruct-2507-Q8_0.gguf"],
    )
    pins = replication.pins(_flow(), cache, "2026.10.1")
    assert pins["studio_version"] == "2026.10.1"
    by_repo = {m["repo"]: m for m in pins["models"]}
    assert by_repo[QWEN] == {
        "repo": QWEN,
        "variant": "Q4_K_M",
        "revision": SHA_A,
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    }
    # Not downloaded here: named, with no revision to claim.
    assert by_repo[FLUX] == {"repo": FLUX}
    assert not any("secret" in json.dumps(m) for m in pins["models"])


def test_import_notices_name_a_different_revision_and_studio_version(tmp_path):
    made = _cache(tmp_path / "a", QWEN, SHA_A, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
    d = _flow()
    replication.stamp(d, made, "2026.10.1")
    # Same cache, same version: nothing to say.
    assert replication.notices(d, made, "2026.10.1") == []
    other = _cache(tmp_path / "b", QWEN, SHA_B, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
    notes = replication.notices(d, other, "2026.11.0")
    assert notes[0] == "Made with Unsloth Studio 2026.10.1; you have 2026.11.0."
    assert notes[1].startswith(f"Made with {QWEN} revision {SHA_A[:12]}; you have {SHA_B[:12]}")
    # A model the importer has not downloaded is the plan's job (missing models), not a notice.
    assert replication.notices(d, tmp_path / "empty", "2026.10.1") == []


def test_files_without_a_stamp_import_as_before(tmp_path):
    assert replication.notices(_flow(), tmp_path, "x") == []
    assert replication.notices({"meta": {"replication": "junk"}}, tmp_path, "x") == []


def test_malformed_replication_metadata_imports_without_notices(tmp_path):
    cache = _cache(tmp_path, QWEN, SHA_B, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
    for models in ({"a": 1}, 7, "junk", [None, 3]):
        d = {"meta": {"replication": {"models": models}}}
        assert replication.notices(d, cache, "x") == []
    # Same revision, so the file is compared: a variant that is not text names no file.
    d = {
        "meta": {
            "replication": {
                "models": [{"repo": QWEN, "revision": SHA_B, "file": "x.gguf", "variant": 5}]
            }
        }
    }
    assert replication.notices(d, cache, "x") == []


@run_async
async def test_every_export_mode_is_stamped_and_still_imports(tmp_path, monkeypatch):
    cache = _cache(tmp_path, QWEN, SHA_A, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
    monkeypatch.setattr(replication, "_hub_cache", lambda: cache)
    monkeypatch.setattr(replication, "_studio_version", lambda: "2026.10.1")

    async def read_media(media):
        raise AssertionError("no media in this flow")

    for mode in ("json", "package", "baked"):
        data, _, _ = await package.export_flow(_flow(), mode, read_media)
        parsed, _, _ = package.read_package(data)
        stamp = parsed["meta"]["replication"]
        assert stamp["studio_version"] == "2026.10.1"
        assert {
            "repo": QWEN,
            "variant": "Q4_K_M",
            "revision": SHA_A,
            "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        } in stamp["models"]


def test_live_conversations_flags_worlds_whose_players_call_a_model():
    assert replication.live_conversations(doc([node("r", "world.npc_reply")]))
    assert not replication.live_conversations(
        doc([{**node("r", "world.npc_reply"), "disabled": True}])
    )
    assert not replication.live_conversations(doc([node("t", "input.text", value = "x")]))


def test_the_recorded_gguf_is_the_exact_variant_not_a_substring(tmp_path):
    cache = _cache(tmp_path, QWEN, SHA_A, ["m-BF16.gguf", "m-F16.gguf", "mmproj-F16.gguf"])
    assert replication.cached_gguf(cache, QWEN, SHA_A, "F16") == "m-F16.gguf"
    assert replication.cached_gguf(cache, QWEN, SHA_A, "BF16") == "m-BF16.gguf"


def test_a_path_qualified_gguf_variant_is_pinned_but_a_local_path_is_not(tmp_path):
    cache = _cache(tmp_path, QWEN, SHA_A, ["weights/model-Q6_K.gguf", "other/model-Q6_K.gguf"])
    d = doc(
        [
            node("a", "llm.generate", model = {"repo": QWEN, "variant": "weights/model-Q6_K"}),
            node("b", "llm.generate", model = {"repo": QWEN, "variant": "/home/me/model-Q6_K"}),
        ]
    )
    pins = replication.pins(d, cache, "x")["models"]
    assert {
        "repo": QWEN,
        "variant": "weights/model-Q6_K",
        "revision": SHA_A,
        "file": "weights/model-Q6_K.gguf",
    } in pins
    assert not any("/home/me" in json.dumps(m) for m in pins)
