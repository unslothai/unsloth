# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GET /api/models/export-size (the Export page size estimate).

The endpoint must never raise and must degrade to nulls when size is unknown, and the local
folder sizer behind it must charge one copy of the weights and no trainer bookkeeping.
"""

import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.hardware.hardware import _get_local_weight_size_bytes

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Real Qwen3.6-35B-A3B: 35.95B params -> ~67 GiB bf16 (UI wrongly showed Q8 ~8.2 GB).
_QWEN35_PARAMS = 35_951_822_704
_QWEN35_FP16_BYTES = _QWEN35_PARAMS * 2


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExportSizeEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_route = _load_route_module(
            "models_route_module_for_export_size_test",
            "routes/models.py",
        )

    def setUp(self):
        self.models_route._EXPORT_SIZE_CACHE.clear()

    def _call(self, model: str = "unsloth/Qwen3.6-35B-A3B"):
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(self.models_route, "resolve_cached_repo_id_case", side_effect = lambda m: m),
        ):
            return asyncio.run(
                self.models_route.get_export_size(
                    model = model, hf_token = None, current_subject = "test-user"
                )
            )

    def test_known_model_returns_bytes_and_params(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (_QWEN35_FP16_BYTES, "safetensors"),
        ):
            resp = self._call()
        self.assertEqual(resp.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(resp.total_params, _QWEN35_PARAMS)
        self.assertEqual(resp.source, "safetensors")
        self.assertEqual(resp.model, "unsloth/Qwen3.6-35B-A3B")

    def test_moe_via_config_fallback(self):
        # MoE sized via the sizer's config path -> source "config".
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (67 * (1024**3), "config"),
        ):
            resp = self._call()
        self.assertEqual(resp.fp16_bytes, 67 * (1024**3))
        self.assertEqual(resp.total_params, (67 * (1024**3)) // 2)
        self.assertEqual(resp.source, "config")

    def test_unknown_size_returns_nulls_not_error(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (None, "unavailable"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertIsNone(resp.total_params)
        self.assertEqual(resp.source, "unavailable")

    def test_zero_size_treated_as_unknown(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (0, "safetensors"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertIsNone(resp.total_params)

    def test_sizer_exception_is_swallowed(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            side_effect = RuntimeError("boom"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")

    def test_result_is_memoized_per_model(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (_QWEN35_FP16_BYTES, "safetensors"),
        ) as mock_sizer:
            first = self._call()
            second = self._call()
        self.assertEqual(first.fp16_bytes, second.fp16_bytes)
        self.assertEqual(mock_sizer.call_count, 1)

    def test_failures_are_not_cached(self):
        # A transient failure must not poison the cache; a later call recovers.
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            side_effect = [(None, "unavailable"), (_QWEN35_FP16_BYTES, "safetensors")],
        ) as mock_sizer:
            first = self._call()
            second = self._call()
        self.assertIsNone(first.fp16_bytes)
        self.assertEqual(second.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(mock_sizer.call_count, 2)

    def test_token_is_forwarded_to_sizer(self):
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(self.models_route, "resolve_cached_repo_id_case", side_effect = lambda m: m),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (_QWEN35_FP16_BYTES, "safetensors"),
            ) as mock_sizer,
        ):
            asyncio.run(
                self.models_route.get_export_size(
                    model = "unsloth/Private",
                    hf_token = "secret-token",
                    current_subject = "test-user",
                )
            )
        self.assertEqual(mock_sizer.call_args.kwargs.get("hf_token"), "secret-token")

    def test_arbitrary_local_path_is_not_scanned(self):
        # Unsafe local paths must not be scanned -> unavailable.
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(self.models_route, "_is_sizable_local_path", return_value = False),
            patch("utils.hardware.hardware.estimate_fp16_model_size_bytes") as mock_sizer,
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = "/etc", hf_token = None, current_subject = "test-user"
                )
            )
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")
        mock_sizer.assert_not_called()

    def test_sizable_local_path_is_sized(self):
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(self.models_route, "_is_sizable_local_path", return_value = True),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                side_effect = lambda m, **_kw: m,
            ),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (_QWEN35_FP16_BYTES, "local"),
            ),
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = "/root/.unsloth/studio/outputs/run",
                    hf_token = None,
                    current_subject = "test-user",
                )
            )
        self.assertEqual(resp.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(resp.source, "local")

    def test_local_adapter_base_escaping_roots_is_rejected(self):
        # A local adapter under a root whose resolved base points outside the
        # roots (e.g. "/") must not be sized: the resolved base is re-validated.
        adapter = "/root/.unsloth/studio/outputs/adapter"
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(
                self.models_route, "_is_sizable_local_path", side_effect = lambda p: p == adapter
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "/",
            ),
            patch("utils.hardware.hardware.estimate_fp16_model_size_bytes") as mock_sizer,
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = adapter, hf_token = None, current_subject = "test-user"
                )
            )
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")
        mock_sizer.assert_not_called()

    def test_is_sizable_local_path_containment(self):
        # Only paths under a trusted root are sizable; '..' can't escape.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            inside = root / "run-1"
            inside.mkdir(parents = True)
            with (
                patch("utils.paths.studio_root", return_value = root),
                patch("utils.paths.outputs_root", return_value = root),
                patch("utils.paths.exports_root", return_value = root),
                patch("utils.paths.storage_roots.cache_root", return_value = root),
            ):
                is_sizable = self.models_route._is_sizable_local_path
                self.assertTrue(is_sizable(str(inside)))
                self.assertTrue(is_sizable(str(root)))
                self.assertFalse(is_sizable(str(root / "missing")))
                self.assertFalse(is_sizable("/etc"))
                self.assertFalse(is_sizable(str(root / ".." / "etc")))
                # A symlink inside a root pointing outside it cannot escape.
                escape = root / "escape"
                os.symlink(tmp, escape)
                self.assertFalse(is_sizable(str(escape)))

    def test_local_weight_size_skips_nested_checkpoints(self):
        # A run dir's intermediate checkpoint-*/global_step* snapshots must not
        # be counted; only the model files at the root are summed.
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "model.safetensors").write_bytes(b"\0" * 1000)
            for sub, size in (("checkpoint-60", 5000), ("global_step10", 7000)):
                d = run / sub
                d.mkdir()
                (d / "model.safetensors").write_bytes(b"\0" * size)
            self.assertEqual(_get_local_weight_size_bytes(str(run)), 1000)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)


def _write_index(path: Path, shards: dict) -> None:
    import json
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(shards.values())},
                "weight_map": {f"layer.{i}.weight": name for i, name in enumerate(sorted(shards))},
            }
        )
    )


def test_dual_format_repo_counts_safetensors_only(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "consolidated.00.pth", 1200)
    _write(tmp_path / "training_args.bin", 500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_full_finetune_output_dir_ignores_bookkeeping(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 4000)
    _write(tmp_path / "model-00002-of-00002.safetensors", 3000)
    _write(tmp_path / "training_args.bin", 500)
    _write(tmp_path / "optimizer.pt", 8000)
    _write(tmp_path / "scheduler.pt", 100)
    _write(tmp_path / "rng_state.pth", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_bin_only_repo_still_sums_shards(tmp_path):
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 4000)
    _write(tmp_path / "pytorch_model-00002-of-00002.bin", 3000)
    _write(tmp_path / "training_args.bin", 500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_diffusers_subfolder_bin_components_are_counted(tmp_path):
    _write(tmp_path / "unet" / "diffusion_pytorch_model.bin", 3440)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.bin", 320)
    _write(tmp_path / "text_encoder" / "pytorch_model.bin", 460)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4220


def test_mixed_format_components_are_all_counted(tmp_path):
    _write(tmp_path / "unet" / "diffusion_pytorch_model.safetensors", 9900)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.bin", 320)
    _write(tmp_path / "text_encoder_2" / "pytorch_model.bin", 9500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 19720


def test_torch_only_tower_beside_safetensors_language_model(tmp_path):
    _write(tmp_path / "model-00001-of-00001.safetensors", 13400)
    _write(tmp_path / "vision_tower" / "pytorch_model.bin", 1700)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 15100


def test_independent_torch_payload_in_the_same_directory_is_counted(tmp_path):
    _write(tmp_path / "model.safetensors", 13400)
    _write(tmp_path / "mm_projector.bin", 1700)
    _write(tmp_path / "projector.pt", 300)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 15400


def test_freely_named_torch_checkpoints_are_counted(tmp_path):
    _write(tmp_path / "open_clip" / "open_clip_pytorch_model.bin", 3900)
    _write(tmp_path / "lora" / "pytorch_lora_weights.bin", 150)
    _write(tmp_path / "esm" / "esm2_t33_650M_UR50D.pt", 2600)
    _write(tmp_path / "legacy" / "weights.pth", 1200)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7850


def test_original_copy_counts_when_it_is_the_only_copy(tmp_path):
    _write(tmp_path / "original" / "consolidated.00.pth", 16000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 16000


def test_root_copy_wins_over_a_larger_original_copy(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "consolidated.00.pth", 1100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_original_only_base_beside_an_external_tower_counts_both(tmp_path):
    _write(tmp_path / "original" / "consolidated.00.pth", 16000)
    _write(tmp_path / "vision_tower" / "pytorch_model.bin", 1700)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 17700


def test_uppercase_extension_is_not_a_loadable_weight(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "stale.SAFETENSORS", 5000)
    _write(tmp_path / "stale.BIN", 5000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_safetensors_optimizer_sidecar_is_bookkeeping(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 4000)
    _write(tmp_path / "model-00002-of-00002.safetensors", 3000)
    _write(tmp_path / "optimizer.safetensors", 8000)
    _write(tmp_path / "scheduler.safetensors", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_sharded_optimizer_state_is_bookkeeping(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 4000)
    _write(tmp_path / "model-00002-of-00002.safetensors", 3000)
    _write(tmp_path / "optimizer-00001-of-00002.bin", 8000)
    _write(tmp_path / "optimizer-00002-of-00002.bin", 8000)
    _write(tmp_path / "rng_state_0.pth", 100)
    _write(tmp_path / "rng_state_1.pth", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7000


def test_precision_variant_beside_canonical_weights_charges_one_copy(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "model.fp16.safetensors", 500)
    _write(tmp_path / "unet" / "diffusion_pytorch_model.safetensors", 3440)
    _write(tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors", 1720)
    _write(tmp_path / "unet" / "diffusion_pytorch_model.bin", 3440)
    _write(tmp_path / "unet" / "diffusion_pytorch_model.fp16.bin", 1720)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4440


def test_sharded_precision_variant_is_held_back_by_the_index(tmp_path):
    shards = {"model-00001-of-00002.safetensors": 600, "model-00002-of-00002.safetensors": 400}
    for name, size in shards.items():
        _write(tmp_path / name, size)
    _write_index(tmp_path / "model.safetensors.index.json", shards)
    _write(tmp_path / "model.fp16-00001-of-00002.safetensors", 300)
    _write(tmp_path / "model.fp16-00002-of-00002.safetensors", 200)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_variant_after_the_shard_counter_is_the_same_archive(tmp_path):
    shards = {"model-00001-of-00002.safetensors": 600, "model-00002-of-00002.safetensors": 400}
    for name, size in shards.items():
        _write(tmp_path / name, size)
    _write_index(tmp_path / "model.safetensors.index.json", shards)
    _write(tmp_path / "model-00001-of-00002.fp16.safetensors", 300)
    _write(tmp_path / "model-00002-of-00002.fp16.safetensors", 200)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_diffusers_sharded_component_is_charged_by_its_index_once(tmp_path):
    # genmo/mochi-1-preview: sharded denoiser with bf16 twins, components with variants and .bin.
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": 3000,
        "diffusion_pytorch_model-00002-of-00002.safetensors": 2000,
    }
    for name, size in shards.items():
        _write(tmp_path / "transformer" / name, size)
    _write_index(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json", shards
    )
    _write(
        tmp_path / "transformer" / "diffusion_pytorch_model-00001-of-00002.bf16.safetensors", 1500
    )
    _write(
        tmp_path / "transformer" / "diffusion_pytorch_model-00002-of-00002.bf16.safetensors", 1000
    )
    _write(tmp_path / "vae" / "diffusion_pytorch_model.safetensors", 320)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.fp16.safetensors", 160)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.bin", 320)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 5320


def test_an_unrelated_index_json_is_never_opened(tmp_path, monkeypatch):
    from utils.hardware import hardware

    _write(tmp_path / "model.safetensors", 1000)
    (tmp_path / "search.index.json").write_text('{"weight_map": {"x": "model.safetensors"}}')
    opened = []
    real = hardware._index_targets

    def spy(index, directory):
        opened.append(index.name)
        return real(index, directory)

    monkeypatch.setattr(hardware, "_index_targets", spy)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000
    assert "search.index.json" not in opened


def test_a_variant_only_spelling_is_held_beside_the_loadable_one(tmp_path):
    _write(tmp_path / "model.fp16.safetensors", 500)
    _write(tmp_path / "pytorch_model.bin", 1000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_same_stem_copy_of_the_archive_is_not_a_component(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "model.pt", 1000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_diffusers_index_outranks_the_direct_file_beside_it(tmp_path):
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": 3000,
        "diffusion_pytorch_model-00002-of-00002.safetensors": 2000,
    }
    for name, size in shards.items():
        _write(tmp_path / "unet" / name, size)
    _write_index(tmp_path / "unet" / "diffusion_pytorch_model.safetensors.index.json", shards)
    _write(tmp_path / "unet" / "diffusion_pytorch_model.safetensors", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 5000


def test_a_transformers_direct_file_still_outranks_a_stale_index(tmp_path):
    shards = {"model-00001-of-00002.safetensors": 3000, "model-00002-of-00002.safetensors": 2000}
    for name, size in shards.items():
        _write(tmp_path / name, size)
    _write_index(tmp_path / "model.safetensors.index.json", shards)
    _write(tmp_path / "model.safetensors", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 100


def test_every_non_default_variant_is_held_with_its_archive(tmp_path):
    component = tmp_path / "transformer"
    _write(component / "diffusion_pytorch_model.safetensors", 4000)
    _write(component / "diffusion_pytorch_model.fp8.safetensors", 2000)
    _write(component / "diffusion_pytorch_model.fp8_e4m3fn-00001-of-00002.safetensors", 900)
    _write(component / "diffusion_pytorch_model.fp8_e4m3fn-00002-of-00002.safetensors", 900)
    _write(component / "diffusion_pytorch_model.non_ema.safetensors", 4000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4000


def test_a_declared_diffusers_component_never_opens_a_stray_model_safetensors(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "config.json").write_text('{"_class_name": "UNet2DConditionModel"}')
    _write(tmp_path / "unet" / "diffusion_pytorch_model.safetensors", 4000)
    _write(tmp_path / "unet" / "model.safetensors", 35)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4000


def test_a_declared_transformers_component_never_opens_a_stray_diffusion_file(tmp_path):
    # linyq/kiwi-edit-5b-instruct-only-diffusers/mllm_encoder: a transformers archive with a
    # 35 MB diffusion_pytorch_model.safetensors beside it.
    component = tmp_path / "mllm_encoder"
    component.mkdir()
    (component / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    shards = {"model-00001-of-00002.safetensors": 4000, "model-00002-of-00002.safetensors": 3510}
    for name, size in shards.items():
        _write(component / name, size)
    _write_index(component / "model.safetensors.index.json", shards)
    _write(component / "diffusion_pytorch_model.safetensors", 35)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 7510


def test_an_undeclared_folder_holding_both_spellings_charges_both_payloads(tmp_path):
    _write(tmp_path / "comp" / "diffusion_pytorch_model.safetensors", 4000)
    _write(tmp_path / "comp" / "model.safetensors", 35)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4035


def test_a_dotdot_in_the_model_path_keeps_the_indexed_shards(tmp_path):
    model = tmp_path / "existing" / "model"
    _write(model / "shard.bin", 600)
    _write(model / "payload", 400)
    (model / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "shard.bin", "b": "payload"}}'
    )
    (tmp_path / "existing" / "other").mkdir()
    dotted = str(tmp_path / "existing" / "other" / ".." / "model")
    assert _get_local_weight_size_bytes(dotted) == 1000


def test_a_diffusers_pickle_index_is_never_opened(tmp_path):
    unet = tmp_path / "unet"
    _write(unet / "diffusion_pytorch_model.bin", 100)
    _write(unet / "diffusion_pytorch_model-00001-of-00001.bin", 5000)
    (unet / "diffusion_pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "diffusion_pytorch_model-00001-of-00001.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 100


def test_nested_indexed_shards_hold_their_obsolete_sibling(tmp_path):
    _write(tmp_path / "weights" / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "weights" / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "weights" / "model-00003-of-00002.safetensors", 999)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "weights/model-00001-of-00002.safetensors",'
        ' "b": "weights/model-00002-of-00002.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_root_bookkeeping_beside_component_archives_is_dropped(tmp_path):
    _write(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors", 4000)
    _write(tmp_path / "vae" / "diffusion_pytorch_model.safetensors", 320)
    _write(tmp_path / "optimizer.pt", 8000)
    _write(tmp_path / "scheduler.pt", 100)
    _write(tmp_path / "training_args.bin", 500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4320


def test_trainer_state_beside_a_freely_named_payload_alone_still_counts(tmp_path):
    # No loadable archive marks weights.pth the model or optimizer.pt its state, so both count.
    _write(tmp_path / "weights.pth", 4000)
    _write(tmp_path / "optimizer.pt", 8000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 12000


def test_variant_only_shards_are_summed(tmp_path):
    _write(tmp_path / "model-00001-of-00002.fp16.safetensors", 300)
    _write(tmp_path / "model-00002-of-00002.fp16.safetensors", 200)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 500


def test_same_directory_dual_format_charges_the_safetensors_copy(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "pytorch_model.bin", 1500)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_sharded_dual_format_charges_the_safetensors_copy(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 650)
    _write(tmp_path / "pytorch_model-00002-of-00002.bin", 400)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_consolidated_beside_sharded_safetensors_charges_one_copy(tmp_path):
    _write(tmp_path / "consolidated.safetensors", 1000)
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_same_stem_dual_format_charges_one_copy(tmp_path):
    _write(tmp_path / "v1-5-pruned.safetensors", 4000)
    _write(tmp_path / "v1-5-pruned.pth", 5000)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 4000


def test_adapter_dual_format_charges_one_copy(tmp_path):
    _write(tmp_path / "adapter_model.safetensors", 300)
    _write(tmp_path / "adapter_model.bin", 400)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 300


def test_an_adapter_is_charged_on_top_of_the_base_model_it_adapts(tmp_path):
    # peft loads the adapter onto a base already resident, so both are needed.
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "adapter_model.safetensors", 50)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1050


def test_an_adapter_never_stands_in_for_bare_shards_beside_it(tmp_path):
    # Nothing opens the bare shards by name; charging the adapter alone reports 50 bytes for 1000.
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "adapter_model.safetensors", 50)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1050


def test_an_unreadable_index_does_not_let_an_adapter_outrank_the_shards(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    (tmp_path / "model.safetensors.index.json").write_text("{ truncated")
    _write(tmp_path / "adapter_model.bin", 50)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1050


def test_an_adapter_beside_legacy_shards_is_charged_with_them(tmp_path):
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 600)
    _write(tmp_path / "pytorch_model-00002-of-00002.bin", 400)
    _write(tmp_path / "adapter_model.safetensors", 50)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1050


def test_index_decides_which_shards_are_loaded(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "model-00003-of-00002.safetensors", 999)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_an_unreadable_index_falls_back_to_the_shards_on_disk(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    (tmp_path / "model.safetensors.index.json").write_text("{ not json")
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_bookkeeping_name_counts_where_it_is_the_only_weight(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "scaler" / "scaler.safetensors", 300)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1300


def test_a_bookkeeping_prefix_is_not_a_bookkeeping_name(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "scheduler_transformer.safetensors", 300)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1300


def test_a_direct_safetensors_file_outranks_a_stale_index(tmp_path):
    _write(tmp_path / "model.safetensors", 100)
    _write(tmp_path / "model-00001-of-00001.safetensors", 1000)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 100


def test_an_index_that_names_the_direct_file_is_not_stale(tmp_path):
    # unsloth/Qwen3.8-27B-NVFP4 ships model.safetensors beside an 0.85 GB MTP head its index
    # names too, so reading the direct file as the whole archive would drop the head.
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "model_mtp.safetensors", 50)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model.safetensors", "b": "model_mtp.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1050


def test_a_direct_pickle_file_outranks_its_own_index(tmp_path):
    _write(tmp_path / "pytorch_model.bin", 100)
    _write(tmp_path / "pytorch_model-00001-of-00001.bin", 1000)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 100


def test_a_loadable_pickle_outranks_safetensors_shards_no_index_names(tmp_path):
    _write(tmp_path / "pytorch_model.bin", 1000)
    _write(tmp_path / "model-00001-of-00002.safetensors", 100)
    _write(tmp_path / "model-00002-of-00002.safetensors", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_pickle_index_is_read_when_no_safetensors_archive_exists(tmp_path):
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 600)
    _write(tmp_path / "pytorch_model-00002-of-00002.bin", 400)
    _write(tmp_path / "pytorch_model-00003-of-00002.bin", 900)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00002.bin",'
        ' "b": "pytorch_model-00002-of-00002.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_safetensors_index_outranks_a_direct_pickle_file(tmp_path):
    _write(tmp_path / "model-00001-of-00001.safetensors", 1000)
    _write(tmp_path / "pytorch_model.bin", 1500)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_an_original_copy_never_outranks_the_weights_the_directory_has(tmp_path):
    _write(tmp_path / "pytorch_model.bin", 1000)
    _write(tmp_path / "original" / "model.safetensors", 100)
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_projector_beside_indexed_weights_is_counted(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "mm_projector.bin", 300)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1300


def test_a_subfolder_only_a_losing_index_names_is_charged_anyway(tmp_path):
    # Nothing tells this folder from a component that is loaded, and hiding a real one is worse.
    _write(tmp_path / "sf" / "model-00001-of-00001.safetensors", 1000)
    _write(tmp_path / "pk" / "pytorch_model-00001-of-00001.bin", 1500)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "sf/model-00001-of-00001.safetensors"}}'
    )
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pk/pytorch_model-00001-of-00001.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 2500


def test_an_index_names_a_shard_that_carries_no_weight_extension(tmp_path):
    _write(tmp_path / "pytorch_model-00001-of-00002.bin", 600)
    _write(tmp_path / "payload", 400)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_an_archive_spelled_entirely_without_weight_suffixes_is_read(tmp_path):
    _write(tmp_path / "payload", 28702)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "payload", "b": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 28702


def test_an_index_under_original_does_not_charge_the_copy_twice(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "consolidated.00.pth", 1200)
    (tmp_path / "original" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "consolidated.00.pth"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_vendor_archive_keeps_the_shard_that_could_not_be_folded_up(tmp_path):
    _write(tmp_path / "original" / "pytorch_model-00001-of-00002.bin", 23143)
    _write(tmp_path / "original" / "payload", 6079)
    (tmp_path / "original" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 29222


def test_a_wholly_unsuffixed_vendor_archive_is_read(tmp_path):
    _write(tmp_path / "original" / "payload", 28702)
    (tmp_path / "original" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 28702


def test_a_vendor_index_does_not_recharge_the_shards_folded_above_it(tmp_path):
    _write(tmp_path / "model-00001-of-00002.safetensors", 600)
    _write(tmp_path / "model-00002-of-00002.safetensors", 400)
    _write(tmp_path / "original" / "consolidated.00.pth", 1100)
    (tmp_path / "original" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "consolidated.00.pth"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_stale_vendor_index_does_not_outrank_the_vendor_file_beside_it(tmp_path):
    _write(tmp_path / "original" / "pytorch_model.bin", 29196)
    _write(tmp_path / "original" / "payload", 28702)
    (tmp_path / "original" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 29196


def test_a_losing_vendor_archive_contributes_nothing_shard_by_shard(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "pytorch_model-00001-of-00002.bin", 23143)
    _write(tmp_path / "original" / "payload", 6079)
    (tmp_path / "original" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_losing_vendor_archive_keeps_an_arbitrary_target_out_too(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "original" / "payload.bin", 5000)
    (tmp_path / "original" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "payload.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_vendor_copy_of_a_nested_component_keeps_its_own_shape(tmp_path):
    _write(tmp_path / "original" / "vision_tower" / "pytorch_model-00001-of-00002.bin", 23143)
    _write(tmp_path / "original" / "vision_tower" / "payload", 6079)
    (tmp_path / "original" / "vision_tower" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00002.bin", "b": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 29222


def test_a_nested_vendor_copy_loses_to_the_component_it_copies(tmp_path):
    _write(tmp_path / "vision_tower" / "model.safetensors", 1000)
    _write(tmp_path / "original" / "vision_tower" / "payload", 28702)
    (tmp_path / "original" / "vision_tower" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_vendor_index_of_the_winning_spelling_is_still_the_same_weights(tmp_path):
    _write(tmp_path / "shard-a.safetensors", 1000)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "shard-a.safetensors"}}'
    )
    _write(tmp_path / "original" / "shard-b.safetensors", 7777)
    (tmp_path / "original" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "shard-b.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_wholly_unsuffixed_vendor_copy_of_a_nested_component_is_read(tmp_path):
    _write(tmp_path / "original" / "vision" / "payload", 28702)
    (tmp_path / "original" / "vision" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 28702


def test_a_vendor_index_holds_a_target_named_under_another_extension(tmp_path):
    _write(tmp_path / "native-shard.safetensors", 1000)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "native-shard.safetensors"}}'
    )
    _write(tmp_path / "original" / "vendor-shard.bin", 5000)
    (tmp_path / "original" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "vendor-shard.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_the_directory_a_weight_index_sits_in_is_tried_before_the_vendor_copy(tmp_path):
    _write(tmp_path / "original" / "native-index-target.safetensors", 1000)
    _write(tmp_path / "original" / "vendor-target.safetensors", 7777)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "original/native-index-target.safetensors"}}'
    )
    (tmp_path / "original" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "vendor-target.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_weight_index_names_its_shards_in_utf8(tmp_path):
    # Under the operator's locale the name stops matching; U+00DF cannot normalise away.
    _write(tmp_path / "model\u00df.safetensors", 1000)
    _write(tmp_path / "model-00001-of-00001.safetensors", 7777)
    (tmp_path / "model.safetensors.index.json").write_bytes(
        b'{"weight_map": {"a": "model\xc3\x9f.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_a_subfolder_shard_with_no_weight_suffix_is_still_found(tmp_path):
    _write(tmp_path / "shards" / "payload", 28702)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"w": "shards/payload"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 28702


def test_a_shard_named_into_a_subfolder_outranks_its_twin_there(tmp_path):
    _write(tmp_path / "weights" / "shard.bin", 1500)
    _write(tmp_path / "weights" / "shard.safetensors", 1000)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "weights/shard.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1500


def test_a_deeper_index_cannot_recharge_a_shard_already_claimed_above_it(tmp_path):
    _write(tmp_path / "weights" / "pytorch_model-00001-of-00001.bin", 1500)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "weights/pytorch_model-00001-of-00001.bin"}}'
    )
    (tmp_path / "weights" / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1500


def test_a_passed_over_index_does_not_swallow_a_component_it_names(tmp_path):
    _write(tmp_path / "model.safetensors", 1000)
    _write(tmp_path / "vision_tower" / "model.safetensors", 700)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "vision_tower/model.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1700


def test_an_index_reaching_into_the_vendor_copy_is_still_one_archive(tmp_path):
    _write(tmp_path / "shard-a.safetensors", 600)
    _write(tmp_path / "original" / "shard-b.safetensors", 400)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "shard-a.safetensors", "b": "original/shard-b.safetensors"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1000


def test_an_index_target_that_walks_back_up_lands_where_the_loader_lands(tmp_path):
    _write(tmp_path / "weights" / "shard.bin", 1500)
    _write(tmp_path / "weights" / "shard.safetensors", 1000)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "weights/../weights/shard.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 1500


def test_a_subfolder_component_the_index_ignores_is_still_counted(tmp_path):
    _write(tmp_path / "weights" / "shard.bin", 1500)
    _write(tmp_path / "vision_tower" / "model.safetensors", 700)
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        '{"weight_map": {"a": "weights/shard.bin"}}'
    )
    assert _get_local_weight_size_bytes(str(tmp_path)) == 2200


if __name__ == "__main__":
    unittest.main()
