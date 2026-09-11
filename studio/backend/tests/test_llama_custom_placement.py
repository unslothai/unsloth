# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from fastapi import HTTPException
from routes import inference as r
from models.inference import LoadRequest
from core.inference import llama_cpp as lc
from core.inference.llama_custom_config import compile_custom_config, parse_option_catalog
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_placement_config_fixtures", Path(__file__).with_name("test_llama_custom_config.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
HELP = _fixtures.HELP


class Reached(HTTPException):
    def __init__(self):
        super().__init__(409, "Reached side-effect boundary")


def route_probe(
    mode,
    layers,
    ini_layers,
    *,
    attempts = False,
    raises = False,
    tensor = False,
    device = "none",
    vision = False,
    projector = None,
    inherited_config = False,
):
    events = []
    config = NS(
        identifier = "selected.gguf",
        display_name = "selected",
        is_gguf = True,
        is_vision = vision,
        gguf_file = "selected.gguf",
        gguf_mmproj_file = projector,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_variant = "Q4_K_M",
        gguf_hf_repo = None,
        is_local = True,
    )
    backend = NS(
        is_loaded = False,
        model_identifier = None,
        last_load_intent = None,
        requested_llama_cpp_config = None,
        prepare_custom_config = lambda intent, **k: compile_custom_config(
            intent.llama_cpp_config,
            parse_option_catalog(
                HELP + "\n--device DEVICES                       device selection\n"
            ),
        ),
        matches_load_source = lambda *a: False,
        adopt_load_intent_if_matched = lambda *a, **k: False,
        non_chat_gguf_refusal_for_intent = lambda *a: None,
        host_offload_warning_for_intent = lambda *a: None,
        load_cancelled = lambda: False,
    )

    def acquire(*a, **k):
        events.append("acquire_gpu")
        if not attempts:
            raise Reached()

    async def drained(**k):
        events.append("drain_without_acquire")
        if not attempts:
            raise Reached()

    async def failed_attempt(backend, intent, cancel):
        events.append(
            {
                "attempt": intent.llama_cpp_config,
                "tensor": intent.tensor_parallel,
                "extras": intent.extra_args,
            }
        )
        if raises:
            raise ValueError("Native load failed")
        return False

    async def no_audio(c, q, p):
        return p

    device_line = f"\ndevice={device}" if device is not None else ""
    source = {
        "version": 1,
        "mode": "custom",
        "ini": f"[*]\nnp=1\nngl={ini_layers}\nfit=off{device_line}",
        "section": None,
    }
    request = LoadRequest(
        model_path = "owner/repo" if inherited_config else "selected.gguf",
        gguf_variant = None if inherited_config else "Q4_K_M",
        llama_cpp_config = None if inherited_config else source,
        gpu_memory_mode = mode,
        gpu_layers = layers,
        speculative_type = "off",
        tensor_parallel = tensor,
    )
    with (
        patch(
            "utils.openai_auto_switch_settings.resolve_override_for_load",
            side_effect = lambda _identifier, _alias, variant: (
                "override",
                {
                    "llama_cpp_config": source
                    if variant
                    else {**source, "ini": source["ini"].replace("ngl=0", "ngl=99")}
                },
            ),
        ),
        patch.object(r, "get_llama_cpp_backend", return_value = backend),
        patch.object(r, "get_inference_backend", return_value = NS(active_model_name = None)),
        patch.object(
            r,
            "_resolve_model_identifier_for_request",
            return_value = ("selected.gguf", "selected.gguf", False),
        ),
        patch.object(r, "resolve_effective_chat_template_override", return_value = None),
        patch.object(r, "ModelConfig", NS(from_identifier = lambda **k: config)),
        patch.object(r, "_classify_diffusion_gguf", return_value = False),
        patch.object(r, "_effective_parallel_slots", return_value = 1),
        patch.object(r, "_preflight_native_audio_placement", side_effect = no_audio),
        patch.object(r, "_guard_chat_load_against_training", return_value = None),
        patch.object(r, "_raise_if_sidecar_swap_in_progress"),
        patch.object(r, "_wait_for_model_switch_idle", side_effect = drained),
        patch.object(r, "_run_gguf_load_attempt", side_effect = failed_attempt),
        patch.object(lc.LlamaCppBackend, "_is_vulkan_backend", return_value = False),
        patch("core.inference.gpu_arbiter.acquire_for", side_effect = acquire),
    ):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                r._load_model_impl(
                    request, NS(app = NS(state = NS(llama_parallel_slots = 1))), current_subject = "review"
                )
            )
        assert caught.value.status_code == (400 if raises else 500 if attempts else 409)
    return events


@pytest.mark.parametrize(
    "mode,layers,ini_layers",
    [("auto", -1, 0), ("manual", 0, 99), ("manual", 0, 0), ("auto", -1, 99)],
)
def test_custom_arbiter_uses_native_placement_not_managed_fields(mode, layers, ini_layers):
    events = route_probe(mode, layers, ini_layers)
    assert events == (["drain_without_acquire"] if ini_layers == 0 else ["acquire_gpu"])


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_custom_route_never_retries_from_managed_tensor_toggle(tensor, raises):
    events = route_probe("auto", -1, 0, attempts = True, tensor = tensor, raises = raises)
    attempted = [event for event in events if isinstance(event, dict)]
    assert len(attempted) == 1
    assert attempted[0]["attempt"].mode == "custom"
    assert attempted[0]["extras"] == ()


def test_uncertain_custom_device_keeps_gpu_handoff():
    assert route_probe("manual", 0, 0, device = None) == ["acquire_gpu"]


def test_auto_selected_variant_replaces_early_bare_custom_override():
    assert route_probe("auto", -1, 0, inherited_config = True) == ["drain_without_acquire"]


@pytest.mark.parametrize("vision,projector", [(True, None), (False, "audio-projector.gguf")])
def test_custom_cpu_main_does_not_skip_companion_handoff(vision, projector):
    assert route_probe("manual", 0, 0, vision = vision, projector = projector) == ["acquire_gpu"]
