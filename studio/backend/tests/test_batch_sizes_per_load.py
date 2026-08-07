# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-class n_batch / n_ubatch load fields (llama-server --batch-size / --ubatch-size).

Compact sibling of test_parallel_slots_per_load.py: pydantic bounds, VRAM-budget
precedence, reload dedupe, shadow stripping and the stored-override mapping.
"""

from __future__ import annotations

import sys
import types as _types
from dataclasses import replace
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import httpx  # noqa: F401

from core.inference.llama_cpp import (
    GgufLoadIntent,
    LlamaCppBackend,
    _extra_args_n_ubatch,
)
from core.inference.llama_server_args import BATCH_MAX, BATCH_MIN, strip_shadowing_flags
from models.inference import LoadRequest, ValidateModelRequest
from utils.openai_auto_switch_settings import (
    model_override_load_kwargs,
    normalize_model_override,
)


@pytest.mark.parametrize("field", ["n_batch", "n_ubatch"])
def test_load_request_bounds(field):
    assert getattr(LoadRequest(model_path = "owner/repo"), field) is None
    assert getattr(LoadRequest(model_path = "owner/repo", **{field: BATCH_MAX}), field) == BATCH_MAX
    for bad in (BATCH_MIN - 1, BATCH_MAX + 1):
        with pytest.raises(ValueError):
            LoadRequest(model_path = "owner/repo", **{field: bad})
    # /validate sizes like /load, so it carries the same field and bounds
    assert getattr(ValidateModelRequest(model_path = "owner/repo", **{field: 256}), field) == 256


@pytest.mark.parametrize("field", ["n_batch", "n_ubatch"])
@pytest.mark.parametrize("model", [LoadRequest, ValidateModelRequest])
def test_batch_fields_reject_json_booleans(model, field):
    # bool is an int subclass, so lax pydantic turns `true` into 1 -> --batch-size 1, which
    # llama-server refuses to start on: the caller saw a 500 instead of a 422. The override
    # store already drops booleans, so this keeps /load and /settings in agreement.
    with pytest.raises(ValueError):
        model(model_path = "owner/repo", **{field: True})
    # numeric strings and plain ints still coerce as before
    assert getattr(model(model_path = "owner/repo", **{field: "4096"}), field) == 4096
    assert getattr(model(model_path = "owner/repo", **{field: 4096}), field) == 4096


def test_effective_ubatch_prefers_first_class_over_env():
    env = {"LLAMA_ARG_BATCH": "128", "LLAMA_ARG_UBATCH": "128"}
    assert _extra_args_n_ubatch(None, env = env, n_batch = 4096, n_ubatch = 1024) == 1024


def test_effective_ubatch_lets_extras_override_first_class():
    # extras are appended after the emitted flags, so they last-wins at launch
    assert _extra_args_n_ubatch(["-ub", "256"], env = {}, n_ubatch = 1024) == 256


def test_effective_ubatch_caps_at_batch_and_ctx():
    assert _extra_args_n_ubatch(None, env = {}, n_batch = 512, n_ubatch = 2048) == 512
    assert _extra_args_n_ubatch(None, env = {}, n_ubatch = 4096, n_ctx = 1024) == 1024
    assert _extra_args_n_ubatch(None, env = {}) is None


def _loaded_backend() -> LlamaCppBackend:
    backend = LlamaCppBackend()
    backend._process = object()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._requested_spec_mode = "auto"
    return backend


def _intent(**kwargs) -> GgufLoadIntent:
    return GgufLoadIntent(
        model_identifier = "owner/repo",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        speculative_type = "auto",
        **kwargs,
    )


def test_dedupe_matches_same_batch_sizes():
    backend = _loaded_backend()
    backend._requested_n_batch = 4096
    matches = backend._runtime_matches_intent(_intent(n_batch = 4096), None)
    assert matches is True


def test_dedupe_reloads_on_batch_change():
    backend = _loaded_backend()
    assert backend._runtime_matches_intent(_intent(n_batch = 4096), None) is False
    backend._requested_n_ubatch = 512
    assert backend._runtime_matches_intent(_intent(n_ubatch = 1024), None) is False


def test_dedupe_ignores_batch_for_diffusion():
    backend = _loaded_backend()
    backend._is_diffusion = True
    backend._diffusion_requested_ngl = None
    backend._gpu_layers = -1
    assert backend._runtime_matches_intent(_intent(n_batch = 4096), None) is True


def test_strip_shadowing_flags_batch_toggles():
    args = ["-b", "4096", "--ubatch-size=256", "--top-k", "20"]
    assert strip_shadowing_flags(args, strip_batch = True) == ["--ubatch-size=256", "--top-k", "20"]
    assert strip_shadowing_flags(args, strip_ubatch = True) == ["-b", "4096", "--top-k", "20"]
    assert strip_shadowing_flags(args) == args


def test_override_store_round_trip():
    entry = normalize_model_override({"n_batch": 4096, "n_ubatch": 1024})
    assert entry == {"n_batch": 4096, "n_ubatch": 1024}
    # out of range or boolean values drop silently, like the other knobs
    assert normalize_model_override({"n_batch": 0, "n_ubatch": True}) == {}
    kwargs = model_override_load_kwargs(entry, is_gguf = True)
    assert kwargs["n_batch"] == 4096 and kwargs["n_ubatch"] == 1024
    assert "n_batch" not in model_override_load_kwargs(entry, is_gguf = False)


def test_fast_path_intent_strips_inherited_batch_flags_when_field_set():
    # the already-loaded dedupe must see the same override the slow path launches
    from routes.inference import _active_gguf_intent

    backend = _loaded_backend()
    backend._extra_args = ["-b", "512", "--top-k", "20"]
    backend._extra_args_source = ("owner/repo", "Q4_K_M")

    kwargs = dict(
        model_identifier = "owner/repo",
        chat_template_override = None,
        n_parallel = 1,
        native_grant_backed = False,
    )
    overriding = _active_gguf_intent(
        LoadRequest(model_path = "owner/repo", n_batch = 4096), backend, **kwargs
    )
    assert overriding.extra_args == ("--top-k", "20")
    assert overriding.extra_args_inherited is False

    inheriting = _active_gguf_intent(LoadRequest(model_path = "owner/repo"), backend, **kwargs)
    assert inheriting.extra_args == ("-b", "512", "--top-k", "20")
    assert inheriting.extra_args_inherited is True


def test_slow_path_intent_strips_inherited_batch_flags_when_field_set(monkeypatch):
    # Sibling of the fast-path case above. A bare repo id with no gguf_variant skips
    # _active_gguf_intent entirely, so without the same bookkeeping here the dedupe
    # compares the LAUNCHED extras (still carrying -b 512) against themselves and reports
    # already_loaded -- leaving the server at effective batch 512 for an Apply that asked
    # for 4096.
    from types import SimpleNamespace

    from routes import inference as route

    backend = _loaded_backend()
    backend._extra_args = ["-b", "512", "--top-k", "20"]
    backend._requested_extra_args = ["-b", "512", "--top-k", "20"]
    backend._extra_args_source = ("owner/repo", "Q4_K_M")
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)

    config = SimpleNamespace(
        identifier = "owner/repo",
        is_gguf = True,
        gguf_hf_repo = "owner/repo",
        gguf_variant = "Q4_K_M",
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        is_vision = False,
    )
    # n_ctx must match the resident one, or the dedupe short-circuits before the extras.
    request = LoadRequest(model_path = "owner/repo", n_batch = 4096, max_seq_length = 8192)
    resolved = route._resolve_inherited_extra_args(request, config, config.identifier, None)
    assert resolved == ["--top-k", "20"]

    intent = route._resolve_gguf_load_intent(
        config,
        request,
        native_grant_backed = False,
        chat_template_override = None,
        extra_args = resolved,
        placement = SimpleNamespace(
            resolved_gpu_ids = None, gpu_ids_are_vulkan_ordinals = None, requested_gpu_ids = None
        ),
        n_parallel = 1,
    )
    assert intent.extra_args_inherited is False
    # Match the resident batch too, so the stripped inherited -b is the ONLY reason left to
    # reload. Without this the batch-field comparison alone satisfies the assert and the
    # extras are never reached.
    backend._requested_n_batch = 4096
    assert backend._runtime_matches_intent(intent, ["--top-k", "20"]) is False
    # Control: with the flag no longer inherited, the same intent dedupes.
    assert (
        backend._runtime_matches_intent(
            replace(intent, extra_args_inherited = True), ["--top-k", "20"]
        )
        is True
    )

    # An Apply that does not name the field still inherits the flag untouched.
    plain = LoadRequest(model_path = "owner/repo", max_seq_length = 8192)
    plain_args = route._resolve_inherited_extra_args(plain, config, config.identifier, None)
    plain_intent = route._resolve_gguf_load_intent(
        config,
        plain,
        native_grant_backed = False,
        chat_template_override = None,
        extra_args = plain_args,
        placement = SimpleNamespace(
            resolved_gpu_ids = None, gpu_ids_are_vulkan_ordinals = None, requested_gpu_ids = None
        ),
        n_parallel = 1,
    )
    assert plain_args == ["-b", "512", "--top-k", "20"]
    assert plain_intent.extra_args_inherited is True


def test_remote_gguf_guard_counts_explicit_micro_batch():
    # a remote gguf has no readable dims, but an explicit ubatch still grows the kq mask
    from types import SimpleNamespace
    from unittest.mock import patch

    from routes import inference as route

    config = SimpleNamespace(
        gguf_file = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_hf_repo = "owner/repo",
        gguf_variant = "Q4_K_M",
    )
    remote_variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
    with (
        patch(
            "utils.models.model_config.list_gguf_variants",
            return_value = ([remote_variant], False),
        ),
        patch.object(route, "_remote_gguf_companion_bytes", return_value = 0),
    ):
        base = route._estimate_gguf_required_gb(config, max_seq_length = 32768)
        big = route._estimate_gguf_required_gb(
            config, max_seq_length = 32768, n_batch = 65536, n_ubatch = 65536
        )
    assert base == pytest.approx(1.0)
    # ctx-capped ubatch (32768) x ctx x 2 x 1.5 mask safety ~= 3 GiB on top
    assert big > base + 2.0

    # auto context still reserves: assume the native one fits a full micro-batch
    with (
        patch(
            "utils.models.model_config.list_gguf_variants",
            return_value = ([remote_variant], False),
        ),
        patch.object(route, "_remote_gguf_companion_bytes", return_value = 0),
    ):
        auto_ctx = route._estimate_gguf_required_gb(
            config, max_seq_length = 0, n_batch = 65536, n_ubatch = 65536
        )
        diffusion = route._estimate_gguf_required_gb(
            config,
            max_seq_length = 32768,
            n_batch = 65536,
            n_ubatch = 65536,
            is_diffusion = True,
        )
    assert auto_ctx > base + 2.0
    # the diffusion runner ignores the llama-server batch flags, so no reserve
    assert diffusion == pytest.approx(1.0)


def _header_reader(**dims):
    """Stand in for _read_gguf_metadata, which would otherwise reset the dims we set.

    _estimate_gguf_kv_gb builds its own LlamaCppBackend and reads the file, so patching
    the instance is not enough; the reader itself has to be replaced.
    """

    def _reader(self, *args, **kwargs):
        for key, value in dims.items():
            setattr(self, f"_{key}", value)

    return _reader


_QWEN3_8B = dict(
    vocab_size = 151936, embedding_length = 4096, n_layers = 36, n_heads = 32,
    n_kv_heads = 8, head_dim = 128, context_length = 262144, architecture = "qwen3",
)


def test_guard_rebuilds_the_compute_buffer_when_the_header_lost_its_vocab():
    # _vocab_size comes only from the tokenizer.ggml.tokens array length, so a truncated
    # header keeps the dims and drops the vocab. Substituting the loader's 5 GiB reserve
    # here charged ~139x the real allocation and 409'd loads that fit.
    from unittest.mock import patch

    from routes import inference as route

    novocab = dict(_QWEN3_8B, vocab_size = None)
    with patch.object(LlamaCppBackend, "_read_gguf_metadata", _header_reader(**_QWEN3_8B)):
        complete = route._estimate_gguf_kv_gb("/x.gguf", 8192, n_parallel = 1)
    with patch.object(LlamaCppBackend, "_read_gguf_metadata", _header_reader(**novocab)):
        rebuilt = route._estimate_gguf_kv_gb("/x.gguf", 8192, n_parallel = 1)
        tensor = route._estimate_gguf_kv_gb(
            "/x.gguf", 8192, n_parallel = 1, n_devices = 4, tensor_parallel = True
        )
    # At one slot the vocab-width term drops out entirely, so the rebuild is exact.
    assert rebuilt == pytest.approx(complete, abs = 0.01)
    # And the 5 GiB per device the reserve would have cost never appears.
    assert tensor < complete + 4.0


def test_guard_floors_a_header_with_no_dimensions_at_all():
    # Nothing to rebuild from: both compute terms are blind, so the loader's flat reserve
    # is the floor. Charged once, not per device, since it is an invented number.
    from unittest.mock import patch

    from routes import inference as route

    blind = dict(_QWEN3_8B, vocab_size = None, embedding_length = None, kv_lora_rank = 512)
    reserve_gb = LlamaCppBackend._TENSOR_PARALLEL_BUFFER_RESERVE_MIB / 1024
    with patch.object(LlamaCppBackend, "_read_gguf_metadata", _header_reader(**blind)):
        one = route._estimate_gguf_kv_gb("/x.gguf", 8192, n_parallel = 1)
        four = route._estimate_gguf_kv_gb(
            "/x.gguf", 8192, n_parallel = 1, n_devices = 4, tensor_parallel = True
        )
    assert one > reserve_gb
    assert four == pytest.approx(one, abs = 0.01)


def test_batch_bounds_stay_standard_json_schema():
    # An Annotated BeforeValidator stops pydantic folding the Field constraints into the
    # int core schema, so they leak out as non-standard ge/le and generated clients drop
    # them. Pin the batch fields against an untouched sibling.
    from models.inference import LoadRequest, ValidateModelRequest

    for model in (LoadRequest, ValidateModelRequest):
        props = model.model_json_schema()["properties"]
        sibling = next(s for s in props["n_parallel"]["anyOf"] if s.get("type") == "integer")
        for field in ("n_batch", "n_ubatch"):
            schema = next(s for s in props[field]["anyOf"] if s.get("type") == "integer")
            assert set(schema) == set(sibling), f"{model.__name__}.{field}: {schema}"
            assert schema["minimum"] == BATCH_MIN
            assert schema["maximum"] == BATCH_MAX


def test_remote_guard_drops_the_split_mask_when_pipelining_is_disabled():
    # The local rule gates the 4x KQ-mask multiplier on _pipeline_parallel_disabled_by_args.
    # Without the same gate remotely, -ot / -ncmoe on a 2-GPU pin is charged 4x while the
    # model is undownloaded and 1x once cached: same model, same flags, two verdicts.
    from types import SimpleNamespace
    from unittest.mock import patch

    from routes import inference as route

    config = SimpleNamespace(
        gguf_file = None, gguf_mmproj_file = None, gguf_mtp_file = None,
        gguf_hf_repo = "owner/repo", gguf_variant = "Q4_K_M",
    )
    remote_variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)

    def _required(extra_args):
        with (
            patch(
                "utils.models.model_config.list_gguf_variants",
                return_value = ([remote_variant], False),
            ),
            patch.object(route, "_remote_gguf_companion_bytes", return_value = 0),
        ):
            return route._estimate_gguf_required_gb(
                config, max_seq_length = 32768, n_ubatch = 2048,
                n_devices = 2, llama_extra_args = extra_args,
            )

    split = _required(None)
    for disabling in (["-ot", r"\.ffn_.*_exps\.=CPU"], ["-ncmoe", "8"], ["--no-kv-offload"]):
        assert _required(disabling) < split, disabling


def test_guard_device_count_follows_the_split_the_loader_would_pick():
    from unittest.mock import patch

    from routes.inference import _guard_device_count

    pool = [(0, 8192, 8192), (1, 8192, 8192), (2, 8192, 8192)]
    with patch.object(
        LlamaCppBackend,
        "_effective_gpu_count",
        lambda ids = None: len(ids) if ids else 0,
    ):
        # tensor mode replicates on every device; a vulkan host counts the probed pool
        assert _guard_device_count(None, pool, tensor_parallel = True) == 3
        assert _guard_device_count([1, 2], pool, tensor_parallel = True) == 2
        assert _guard_device_count(None, [], tensor_parallel = True) == 1
        # a layer split lands on the fewest gpus that fit, so auto placement is one
        assert _guard_device_count(None, pool) == 1
        assert _guard_device_count([1, 2], pool) == 2


def test_override_strips_shadowing_batch_flags():
    kwargs = model_override_load_kwargs(
        {"n_batch": 4096, "llama_extra_args": ["-b", "512", "--top-k", "20"]},
        is_gguf = True,
    )
    assert kwargs["llama_extra_args"] == ["--top-k", "20"]
    # a flag with no first-class field behind it still passes through
    kwargs = model_override_load_kwargs(
        {"llama_extra_args": ["-ub", "256"]},
        is_gguf = True,
    )
    assert kwargs["llama_extra_args"] == ["-ub", "256"]
