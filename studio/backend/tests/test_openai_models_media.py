# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402
from auth.authentication import get_current_subject  # noqa: E402
from core.inference import local_model_resolver as resolver  # noqa: E402
from core.inference import media_model_index as mmi  # noqa: E402
from core.inference.media_model_index import MediaModelPick  # noqa: E402
from utils.api_errors import install_api_error_handlers  # noqa: E402


class _Info:
    def __init__(
        self,
        id,
        display_name,
        model_id = None,
        path = None,
        task = None,
        is_gguf = True,
    ):
        self.id = id
        self.display_name = display_name
        self.model_id = model_id
        self.path = path or id
        self.task = task
        self.is_gguf = is_gguf


class _FakeLlama:
    is_loaded = False
    model_identifier = None
    context_length = None
    max_context_length = None
    native_context_length = None
    _is_audio = False
    _audio_type = None


class _FakeUnsloth:
    active_model_name = None
    models: dict = {}
    context_length = None
    max_seq_length = None


def _media_index(monkeypatch, picks_by_task):
    """Stand in for the media index the generation routes resolve against."""
    monkeypatch.setattr(
        mmi,
        "available_media_model_ids",
        lambda task: sorted(p.model_id for p in picks_by_task.get(task, [])),
    )
    monkeypatch.setattr(
        mmi,
        "resolve_local_media_model",
        lambda name, *, task: next(
            (p for p in picks_by_task.get(task, []) if p.model_id == name), None
        ),
    )


def _catalog(
    monkeypatch,
    infos,
    resident = None,
    picks = None,
):
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    async def _fake_catalog():
        return infos

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(
        resolver, "local_gguf_quants", lambda info: ("Q8_0",) if info.is_gguf else None
    )
    monkeypatch.setattr(inf, "_resident_media_status", lambda task: (resident or {}).get(task))
    monkeypatch.setattr(inf, "_stt_model_objects", lambda created: [])
    _media_index(monkeypatch, picks if picks is not None else _PICKS)


_INFOS = [
    _Info("/data/models/Qwen3-Q4.gguf", "Qwen3-Q4", task = "text-generation"),
    _Info(
        "models--unsloth--Z-Image-Turbo-GGUF",
        "Z-Image-Turbo",
        model_id = "unsloth/Z-Image-Turbo-GGUF",
        path = "/hf/models--unsloth--Z-Image-Turbo-GGUF/snapshots/abc",
        task = "text-to-image",
    ),
    _Info(
        "models--Lightricks--LTX-2",
        "LTX-2",
        model_id = "Lightricks/LTX-2",
        path = "/hf/models--Lightricks--LTX-2/snapshots/def",
        task = "text-to-video",
        is_gguf = False,
    ),
    _Info("/data/models/Unsupported.gguf", "Unsupported", task = "image-diffusion-unsupported"),
]


_PICKS = {
    "text-to-image": [
        MediaModelPick(
            "unsloth/Z-Image-Turbo-GGUF",
            "/hf/models--unsloth--Z-Image-Turbo-GGUF/snapshots/abc",
            "z-image-turbo-Q8_0.gguf",
            "gguf",
        )
    ],
    "text-to-video": [
        MediaModelPick("Lightricks/LTX-2", "/hf/models--Lightricks--LTX-2/snapshots/def")
    ],
}


def test_media_models_list_with_task_and_residency(monkeypatch):
    resident = {
        "text-to-image": {
            "loaded": True,
            "repo_id": "unsloth/z-image-turbo-gguf",
            "gguf_variant": "Q8_0",
            "model_kind": "gguf",
        }
    }
    _catalog(monkeypatch, _INFOS, resident)

    data = asyncio.run(inf._openai_catalog_objects())
    ids = {m["id"]: m for m in data}

    image = ids["unsloth/Z-Image-Turbo-GGUF"]
    assert image["task"] == "text-to-image" and image["loaded"] is True
    assert image["quant"] == "Q8_0" and image["display_name"] == "Z-Image-Turbo"
    assert image["object"] == "model" and image["owned_by"] == inf._OWNED_BY
    video = ids["Lightricks/LTX-2"]
    assert video["task"] == "text-to-video" and video["loaded"] is False
    assert "quant" not in video
    assert "Unsupported" not in ids
    assert "task" not in ids["Qwen3-Q4"] and ids["Qwen3-Q4"]["quant"] == "Q8_0"
    assert [m["id"] for m in data].count("unsloth/Z-Image-Turbo-GGUF") == 1
    blob = json.dumps(data)
    assert "/hf/" not in blob and "/data/" not in blob


def test_media_models_report_the_on_disk_quant_when_not_loaded(monkeypatch):
    _catalog(monkeypatch, _INFOS)
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["unsloth/Z-Image-Turbo-GGUF"]["loaded"] is False
    assert ids["unsloth/Z-Image-Turbo-GGUF"]["quant"] == "Q8_0"


def test_resident_media_model_matches_by_path(monkeypatch):
    resident = {
        "text-to-video": {"loaded": True, "repo_id": "/hf/models--Lightricks--LTX-2/snapshots/def"}
    }
    _catalog(monkeypatch, _INFOS, resident)
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["Lightricks/LTX-2"]["loaded"] is True
    assert sum(1 for m in ids.values() if m.get("task") == "text-to-video") == 1


def test_a_resident_sibling_quant_is_not_reported_as_the_indexed_build(monkeypatch):
    """Same repo, different weights: the indexed build is Q8_0 and Q4_K_M is resident."""
    resident = {
        "text-to-image": {
            "loaded": True,
            "repo_id": "unsloth/z-image-turbo-gguf",
            "gguf_variant": "Q4_K_M",
            "model_kind": "gguf",
        }
    }
    _catalog(monkeypatch, _INFOS, resident)
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["unsloth/Z-Image-Turbo-GGUF"]["loaded"] is False
    assert ids["unsloth/Z-Image-Turbo-GGUF"]["quant"] == "Q8_0"


def test_a_standalone_gguf_resident_is_matched_by_its_load_directory(monkeypatch):
    """A standalone GGUF loads with its PARENT directory as the model path.

    Comparing the public id or the catalog's own file path reports the resident model as
    unloaded and then adds a second entry named after the directory.
    """
    pick = MediaModelPick("z-image", "/srv/models", "z-image-Q4_K_M.gguf", "gguf")
    resident = {
        "text-to-image": {
            "loaded": True,
            "repo_id": "/srv/models",
            "gguf_variant": "Q4_K_M",
            "dtype": "gguf",
            "model_kind": "gguf",
        }
    }
    _catalog(monkeypatch, [], resident, picks = {"text-to-image": [pick]})
    data = asyncio.run(inf._openai_catalog_objects())
    assert [m["id"] for m in data] == ["z-image"]
    assert data[0]["loaded"] is True and data[0]["quant"] == "Q4_K_M"
    assert "/srv/" not in json.dumps(data)


def test_only_ids_the_media_resolver_accepts_are_listed(monkeypatch):
    """The index already drops partial pulls, unopenable paths and ambiguous builds.

    Listing anything it rejects advertises an id the generation route answers with
    model_not_found, so an empty index must advertise nothing -- even while a model the
    index does not know is resident.
    """
    resident = {
        "text-to-image": {
            "loaded": True,
            "repo_id": "/srv/models/half-pulled",
            "gguf_variant": "Q4_K_M",
        }
    }
    _catalog(monkeypatch, _INFOS, resident, picks = {})
    data = asyncio.run(inf._openai_catalog_objects())
    assert [m for m in data if m.get("task") in ("text-to-image", "text-to-video")] == []
    assert "half-pulled" not in json.dumps(data)


def test_stt_models_list_downloaded_and_loaded(monkeypatch):
    from core.inference import stt_ggml_sidecar, stt_mtmd_sidecar, stt_sidecar

    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda m: m == "small")
    monkeypatch.setattr(
        stt_ggml_sidecar, "_cached_model_path", lambda m: "/x" if m == "large-v3-turbo" else None
    )
    monkeypatch.setattr(stt_mtmd_sidecar, "is_model_downloaded", lambda m: m == "qwen3-asr-0.6b")
    monkeypatch.setattr(stt_mtmd_sidecar, "is_available", lambda: True)
    monkeypatch.setattr(
        stt_sidecar, "get_stt_sidecar", lambda: SimpleNamespace(loaded_model = "org/whisper-custom")
    )
    monkeypatch.setattr(
        stt_ggml_sidecar,
        "get_ggml_stt_sidecar",
        lambda: SimpleNamespace(loaded_model = "large-v3-turbo"),
    )
    monkeypatch.setattr(
        stt_mtmd_sidecar, "get_mtmd_stt_sidecar", lambda: SimpleNamespace(loaded_model = None)
    )

    objects = inf._stt_model_objects(7)
    assert [(o["id"], o["loaded"]) for o in objects] == [
        ("small", False),
        ("large-v3-turbo", True),
        ("qwen3-asr-0.6b", False),
        ("org/whisper-custom", True),
    ]
    assert all(o["task"] == "automatic-speech-recognition" and o["created"] == 7 for o in objects)


def test_mtmd_models_are_hidden_when_their_runtime_is_missing(monkeypatch):
    """Qwen3-ASR runs on no other engine, so without llama-server every
    /v1/audio/transcriptions call for one returns 501. Curated Whisper ids, which do have
    a fallback engine, must still be listed."""
    from core.inference import stt_ggml_sidecar, stt_mtmd_sidecar, stt_sidecar

    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", lambda m: m == "small")
    monkeypatch.setattr(stt_ggml_sidecar, "_cached_model_path", lambda m: None)
    monkeypatch.setattr(stt_mtmd_sidecar, "is_model_downloaded", lambda m: True)
    monkeypatch.setattr(stt_mtmd_sidecar, "is_available", lambda: False)
    for module, getter in (
        (stt_sidecar, "get_stt_sidecar"),
        (stt_ggml_sidecar, "get_ggml_stt_sidecar"),
        (stt_mtmd_sidecar, "get_mtmd_stt_sidecar"),
    ):
        monkeypatch.setattr(module, getter, lambda: SimpleNamespace(loaded_model = None))

    ids = [o["id"] for o in inf._stt_model_objects(3)]
    assert ids == ["small"]

    monkeypatch.setattr(stt_mtmd_sidecar, "is_available", lambda: True)
    assert [o["id"] for o in inf._stt_model_objects(3)] == [
        "small",
        "qwen3-asr-0.6b",
        "qwen3-asr-1.7b",
    ]


def test_stt_probe_failure_hides_nothing_else(monkeypatch):
    from core.inference import stt_sidecar

    def _boom():
        raise RuntimeError("no sidecar")

    monkeypatch.setattr(stt_sidecar, "get_stt_sidecar", _boom)
    assert inf._stt_model_objects(1) == []


def test_stt_models_join_the_catalog(monkeypatch):
    _catalog(monkeypatch, _INFOS[:1])
    monkeypatch.setattr(
        inf,
        "_stt_model_objects",
        lambda created: [
            {
                "id": "small",
                "object": "model",
                "created": created,
                "owned_by": inf._OWNED_BY,
                "task": "automatic-speech-recognition",
                "loaded": False,
            }
        ],
    )
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["small"]["task"] == "automatic-speech-recognition"
    assert ids["Qwen3-Q4"]["loaded"] is False


def test_loaded_tts_model_is_tagged(monkeypatch):
    class _Tts(_FakeLlama):
        is_loaded = True
        model_identifier = "/srv/models/orpheus-3b-Q4.gguf"
        _is_audio = True
        _audio_type = "snac"

    class _Chat(_FakeLlama):
        is_loaded = True
        model_identifier = "/srv/models/qwen3-Q4.gguf"

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _Tts())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())
    (entry,) = inf._openai_model_objects()
    assert entry["id"] == "orpheus-3b-Q4" and entry["task"] == "text-to-speech"

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _Chat())
    (entry,) = inf._openai_model_objects()
    assert entry["id"] == "qwen3-Q4" and "task" not in entry


def test_audio_input_models_are_not_tagged_text_to_speech(monkeypatch):
    """Only what /v1/audio/speech can actually serve is tagged text-to-speech.

    whisper (ASR) and audio_vlm (Gemma 3n chat) carry an _audio_type but that route
    400s on both, and csm is transformers-only, so none may advertise the task."""
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())
    for audio_type in ("whisper", "audio_vlm", "csm"):
        gguf = type(
            "_Gguf",
            (_FakeLlama,),
            {
                "is_loaded": True,
                "model_identifier": f"/srv/models/{audio_type}-Q4.gguf",
                "_is_audio": False,
                "_audio_type": audio_type,
            },
        )
        monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda gguf = gguf: gguf())
        (entry,) = inf._openai_model_objects()
        assert "task" not in entry, f"{audio_type} advertised as {entry.get('task')}"

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    for audio_type, is_audio in (("audio_vlm", False), ("whisper", True)):
        unsloth = type(
            "_Unsloth",
            (_FakeUnsloth,),
            {
                "active_model_name": f"org/{audio_type}-model",
                "models": {
                    f"org/{audio_type}-model": {"is_audio": is_audio, "audio_type": audio_type}
                },
            },
        )
        monkeypatch.setattr(inf, "get_inference_backend", lambda unsloth = unsloth: unsloth())
        (entry,) = inf._openai_model_objects()
        assert "task" not in entry, f"{audio_type} advertised as {entry.get('task')}"

    # A transformers TTS codec still is tagged.
    unsloth = type(
        "_Tts",
        (_FakeUnsloth,),
        {
            "active_model_name": "unsloth/csm-1b",
            "models": {"unsloth/csm-1b": {"is_audio": True, "audio_type": "csm"}},
        },
    )
    monkeypatch.setattr(inf, "get_inference_backend", lambda: unsloth())
    (entry,) = inf._openai_model_objects()
    assert entry["task"] == "text-to-speech"


def test_resident_media_status(monkeypatch):
    import core.inference.media_keepwarm as mk

    monkeypatch.setattr(mk, "engine_if_imported", lambda owner: None)
    assert inf._resident_media_status("text-to-image") is None
    monkeypatch.setattr(
        mk, "engine_if_imported", lambda owner: SimpleNamespace(status = lambda: {"loaded": False})
    )
    assert inf._resident_media_status("text-to-video") is None
    loaded = {"loaded": True, "repo_id": "unsloth/LTX-2.3-GGUF"}
    monkeypatch.setattr(
        mk, "engine_if_imported", lambda owner: SimpleNamespace(status = lambda: loaded)
    )
    assert inf._resident_media_status("text-to-video") == loaded

    def _boom():
        raise RuntimeError("cuda")

    monkeypatch.setattr(mk, "engine_if_imported", lambda owner: SimpleNamespace(status = _boom))
    assert inf._resident_media_status("text-to-image") is None


def test_engine_if_imported_stays_out_of_torch(monkeypatch):
    import core.inference.media_keepwarm as mk
    from core.inference.gpu_arbiter import DIFFUSION, VIDEO

    for name in (
        "core.inference.diffusion",
        "core.inference.sd_cpp_backend",
        "core.inference.video",
    ):
        monkeypatch.delitem(sys.modules, name, raising = False)
    assert mk.engine_if_imported(DIFFUSION) is None
    assert mk.engine_if_imported(VIDEO) is None


def test_classified_catalog_tags_task(monkeypatch):
    import routes.models as models_mod

    class _Model(BaseModel):
        id: str
        task: Optional[str] = None

    monkeypatch.setattr(models_mod, "_local_model_task", lambda m: "text-to-image")
    plain = object()
    tagged, kept, passthrough = inf._classified_catalog(
        [_Model(id = "a"), _Model(id = "b", task = "text-generation"), plain]
    )
    assert tagged.task == "text-to-image"
    assert kept.task == "text-generation"
    assert passthrough is plain


def test_retrieve_and_list_media_models_over_http(monkeypatch):
    _catalog(monkeypatch, _INFOS)
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(inf.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    listing = client.get("/v1/models").json()
    assert listing["object"] == "list"
    assert {m["id"]: m.get("task") for m in listing["data"]} == {
        "Qwen3-Q4": None,
        "unsloth/Z-Image-Turbo-GGUF": "text-to-image",
        "Lightricks/LTX-2": "text-to-video",
    }
    model = client.get("/v1/models/unsloth/Z-Image-Turbo-GGUF").json()
    assert model["task"] == "text-to-image" and model["loaded"] is False
    assert client.get("/v1/models/nope/missing").status_code == 404
