# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spec packages resolve to variants: a repo its runtime spec names is vouched for whatever
architecture its headers say, and a repo no spec names is not."""

import json
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401

from core.inference import audio_cpp_models as acm  # noqa: E402
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, RepoFile  # noqa: E402

ZONOS_REPO = "Zyphra/ZONOS3-Mini"
LFM_REPO = "LiquidAI/LFM2.5-Audio-1.5B-GGUF"
XTTS_REPO = "onicubag/xTTSv2-GGUF"
SPECS = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "audio_cpp_package_specs.json").read_text(
        encoding = "utf-8"
    )
)
ZONOS_PTQ = ("gguf/backbone-ptq.gguf", "gguf/zac-keep2.gguf", "gguf/spk-encoder.gguf")
LFM_PACKAGES = ("lfm2_audio_1_5b_q8_0", "lfm2_audio_1_5b_f16", "lfm2_audio_1_5b_q4_0")
LFM_SIZES = {
    name: 1000 + i
    for i, name in enumerate(
        f for pk in SPECS["lfm2_audio"]["packages"] if pk["id"] in LFM_PACKAGES for f in pk["files"]
    )
}
XTTS_Q8 = ("q8_0/xtts-v2-gpt-q8_0.gguf", "q8_0/xtts-v2-vocoder-f16.gguf", "q8_0/mel_norms.bin")


def _use_specs(monkeypatch, specs = SPECS):
    """Runtime specs and release come from the fixture, so no installed runtime is read."""
    monkeypatch.setattr(acm, "runtime_specs", lambda: specs)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: specs.get(family) if family else None)
    monkeypatch.setattr(
        "core.inference.audio_cpp_server.runtime_release_key", lambda binary = None: (0, 9, 2)
    )
    acm.forget()


def _listing(monkeypatch, repo, sizes):
    """The Hub listing of repo as {repo path: size}; any other repo lists nothing."""
    files = [RepoFile(path, size) for path, size in sizes.items()]

    def fake(
        repo_id,
        folder = "",
        hf_token = None,
        *,
        network = True,
    ):
        if repo_id.lower() != repo.lower():
            return [], False
        return [f for f in files if not folder or f.path.startswith(folder + "/")], False

    monkeypatch.setattr(acm, "list_files", fake)


def _pkg_spec(repo):
    package = {"id": "pkg", "format": "gguf", "files": ["model.gguf"], "download": {"repo": repo}}
    return {"family": "x", "tasks": ["tts"], "packages": [package]}


def test_zonos3_package_resolves_from_its_spec_alone(hub, monkeypatch):
    sizes = {
        "gguf/backbone-ptq.gguf": 300,
        "gguf/zac-keep2.gguf": 200,
        "gguf/spk-encoder.gguf": 100,
        "gguf/zac-encoder.gguf": 50,
        "gguf/ssl-teacher.gguf": 40,
    }
    _use_specs(monkeypatch)
    _listing(monkeypatch, ZONOS_REPO, sizes)
    _put(_snapshot(hub, ZONOS_REPO), "gguf/backbone-ptq.gguf", _gguf_bytes(arch = "zonos3"))

    model = acm.resolve(ZONOS_REPO, network = False)

    assert model is not None and not model.unsupported
    assert model.family == "zonos3"
    # Clone only: without a reference voice the runtime often decodes silence.
    assert list(model.workflows) == ["clone"] and model.speaks is False
    assert sorted(v.key for v in model.variants) == [
        "zonos3_mini_ptq",
        "zonos3_mini_ptq_raw_reference",
    ]
    assert model.variant.key == "zonos3_mini_ptq"
    assert tuple(f.path for f in model.variant.files) == ZONOS_PTQ
    assert model.variant.primary == "gguf/backbone-ptq.gguf"
    assert model.variant.root == "gguf"
    assert model.is_package is False


def test_lfm2_audio_lists_three_precisions_recommended_first(hub, monkeypatch):
    _use_specs(monkeypatch)
    _listing(monkeypatch, LFM_REPO, LFM_SIZES)
    _put(_snapshot(hub, LFM_REPO), "LFM2.5-Audio-1.5B-Q8_0.gguf", _gguf_bytes(arch = "lfm2"))

    model = acm.resolve(LFM_REPO, network = False)

    assert model is not None and not model.unsupported
    assert [v.key for v in model.variants] == ["Q8_0", "F16", "Q4_0"]
    assert all(len(v.files) == 4 for v in model.variants)
    assert all("JP" not in f.path for v in model.variants for f in v.files)
    assert model.fills_option_defaults is False


def test_repo_no_spec_names_with_foreign_architecture_does_not_resolve(hub, monkeypatch):
    repo = "someone/plain-llama-gguf"
    _use_specs(monkeypatch)
    _listing(monkeypatch, repo, {"model.gguf": 100})
    _put(_snapshot(hub, repo), "model.gguf", _gguf_bytes(arch = "llama"))

    assert acm.spec_vouched_family(repo) is None
    assert acm.spec_vouched_family(AUDIO_CPP_REPO) is None
    assert acm.spec_vouched_family("zyphra/zonos3-mini") == "zonos3"
    assert acm.resolve(repo, network = False) is None
    # A repo two families download from is vouched for by neither.
    _use_specs(monkeypatch, {"fam_a": _pkg_spec("shared/repo"), "fam_b": _pkg_spec("shared/repo")})
    assert acm.spec_vouched_family("shared/repo") is None


def test_a_vouched_repo_studio_cannot_run_stays_llama_cpps(hub, monkeypatch):
    repo = "LiquidAI/d1-Omni-2B-GGUF"
    spec = {**_pkg_spec(repo), "family": "lfm_d1_omni", "tasks": ["classify"]}
    _use_specs(monkeypatch, {**SPECS, "lfm_d1_omni": spec})
    _listing(monkeypatch, repo, {"model.gguf": 100})
    _put(_snapshot(hub, repo), "model.gguf", _gguf_bytes(arch = "lfm2"))

    assert acm.spec_vouched_family(repo) == "lfm_d1_omni"
    assert acm.resolve(repo, network = False) is None


def test_single_file_packages_are_no_package_variant_set(monkeypatch):
    _use_specs(monkeypatch)
    files = [
        RepoFile("Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf", 10),
        RepoFile("Kokoro-82M-GGUF/kokoro-82m-bf16.gguf", 20),
    ]

    assert (
        acm.spec_package_variants(SPECS["kokoro_tts"], AUDIO_CPP_REPO, "Kokoro-82M-GGUF", files)
        is None
    )


def test_xtts_q8_package_is_clone_only_at_its_q8_root(hub, monkeypatch):
    _use_specs(monkeypatch)
    _listing(monkeypatch, XTTS_REPO, {path: 10 for path in XTTS_Q8})
    _put(_snapshot(hub, XTTS_REPO), XTTS_Q8[0], _gguf_bytes(arch = "xtts"))

    model = acm.resolve(XTTS_REPO, network = False)

    assert model is not None
    assert model.speaks is False
    assert list(model.workflows) == ["clone"]
    assert model.variant.root == "q8_0"


def test_a_partly_downloaded_package_is_not_a_downloaded_model(hub, monkeypatch):
    # One GGUF of the three (a lone file fetched from the Model Hub page) is no model yet.
    _use_specs(monkeypatch)
    snap = _snapshot(hub, ZONOS_REPO)
    _put(snap, ZONOS_PTQ[0], _gguf_bytes(arch = "zonos3"))
    assert acm.downloaded_models() == []
    for path in ZONOS_PTQ[1:]:
        _put(snap, path, _gguf_bytes(arch = "zonos3"))
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    [model] = acm.downloaded_models()
    assert (model.id, model.unsupported) == (ZONOS_REPO, None)
    assert [f.path for f in model.variant.files] == list(ZONOS_PTQ)


def test_a_local_gguf_of_a_split_package_needs_its_other_files_beside_it(
    hub, monkeypatch, tmp_path
):
    package = {"id": "pkg", "format": "gguf", "files": ["main.gguf", "codec.gguf"]}
    spec = {"family": "split_tts", "tasks": ["tts"], "packages": [package]}
    _use_specs(monkeypatch, {"split_tts": spec})
    main = _put(tmp_path / "local", "main.gguf", _gguf_bytes(family = "split_tts", spec = spec))

    alone = acm.resolve(str(main), network = False)
    assert alone is not None and alone.unsupported == acm.LOCAL_PACKAGE_INCOMPLETE

    codec = _put(tmp_path / "local", "codec.gguf", _gguf_bytes(family = "split_tts", spec = spec))
    acm.forget()
    assert acm.resolve(str(main), network = False).unsupported is None
    # Another member of the package is no model on its own: the runtime is handed the primary.
    assert acm.resolve(str(codec), network = False).unsupported == acm.LOCAL_PACKAGE_INCOMPLETE
