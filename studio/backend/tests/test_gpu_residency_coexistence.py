# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for core/inference/gpu_residency.py: pricing image and video loads that stay on a GPU beside chat."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import utils.hardware.hardware as _hw_module
from core.inference import gpu_arbiter as arb
from core.inference import gpu_residency as gr
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_memory import estimate_image_runtime_mib, estimate_video_runtime_mib

MIB = 1024 * 1024


class _GpuCacheResetMixin:
    """Reset module-level GPU caches between tests to prevent state leaks."""

    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


def _parent_ids(ids):
    return patch("utils.hardware.hardware.get_parent_visible_gpu_ids", return_value = list(ids))


def _image_request(model_path, gguf_filename = "z-image-turbo-Q4_K_M.gguf", **kw):
    fields = dict(model_kind = None, base_repo = None, family_override = None, text_encoder_file = None, vae_file = None)
    fields.update(kw)
    return SimpleNamespace(model_path = str(model_path), gguf_filename = gguf_filename, **fields)


def _video_request(model_path, gguf_filename, **kw):
    fields = dict(model_kind = None, base_repo = None, family_override = None, display_repo_id = None)
    fields.update(kw)
    return SimpleNamespace(model_path = str(model_path), gguf_filename = gguf_filename, **fields)


def _sparse(path: Path, mib: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as f:
        f.truncate(mib * MIB)
    return path


class TestMediaLoadNeed(_GpuCacheResetMixin, unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "Z-Image-Turbo-GGUF"
        _sparse(self.root / "z-image-turbo-Q4_K_M.gguf", 4000)
        self._patches = [
            # Native engines are host-dependent; the diffusers path is what these tests price.
            patch("core.inference.diffusion_engine_router.predict_engine", return_value = "diffusers"),
            patch.object(DiffusionBackend, "_companion_cache_bytes", staticmethod(lambda base, *a, **k: 6000 * MIB)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()
        super().tearDown()

    def _image(self, ordinal, parent = (2, 3), request = None):
        with _parent_ids(parent):
            return gr.media_load_need_mib(arb.DIFFUSION, request or _image_request(self.root), ordinal)

    def test_gguf_image_prices_weights_plus_headroom_on_the_physical_gpu(self):
        need = self._image(1)
        self.assertIsNotNone(need)
        weights = int(4000 * 1.05) + 6000
        self.assertEqual(need.weights_mib, weights)
        self.assertEqual(need.kind, "image")
        self.assertEqual(list(need), [3])  # ordinal 1 of parent [2, 3]
        self.assertEqual(need[3], weights + need.headroom_mib)
        self.assertGreaterEqual(need.headroom_mib, 1024)
        self.assertLessEqual(need.headroom_mib, estimate_image_runtime_mib(width = None, height = None))

    def test_uncached_checkpoint_is_unknown(self):
        self.assertIsNone(self._image(0, request = _image_request(self.root, "z-image-turbo-Q8_0.gguf")))

    def test_uncached_companions_are_unknown(self):
        with patch.object(DiffusionBackend, "_companion_cache_bytes", staticmethod(lambda *a, **k: 0)):
            self.assertIsNone(self._image(0))

    def test_unknown_family_is_unknown(self):
        with patch("core.inference.diffusion_families.detect_family_for_pick", return_value = None):
            self.assertIsNone(self._image(0))

    def test_native_engine_is_unknown(self):
        with patch("core.inference.diffusion_engine_router.predict_engine", return_value = "sd_cpp"):
            self.assertIsNone(self._image(0))

    def test_automatic_gpu_with_several_visible_is_unknown(self):
        self.assertIsNone(self._image(None, parent = (0, 1)))

    def test_automatic_gpu_with_one_visible_uses_it(self):
        need = self._image(None, parent = (5,))
        self.assertEqual(list(need), [5])

    def test_masked_gpus_are_unknown(self):
        self.assertIsNone(self._image(0, parent = ()))
        self.assertIsNone(self._image(4, parent = (0, 1)))

    def test_internal_error_is_unknown(self):
        with patch("core.inference.diffusion.resolve_model_kind", side_effect = RuntimeError("boom")):
            self.assertIsNone(self._image(0))
        with patch("utils.hardware.hardware.get_parent_visible_gpu_ids", side_effect = OSError("smi")):
            self.assertIsNone(gr.media_load_need_mib(arb.DIFFUSION, _image_request(self.root), 0))

    def test_unknown_kind_is_unknown(self):
        with _parent_ids((0,)):
            self.assertIsNone(gr.media_load_need_mib("chat", _image_request(self.root), 0))

    def _video_family(self, name = "wan-2.1"):
        return SimpleNamespace(name = name, base_repo = "Wan-AI/Wan2.1-T2V-14B-Diffusers", is_moe = False,
                               resolution_presets = ((832, 480),), default_num_frames = 81)

    def test_video_gguf_prices_checkpoint_plus_companion_table(self):
        root = Path(self._tmp.name) / "Wan-GGUF"
        _sparse(root / "wan-Q4_K_M.gguf", 3000)
        fam = self._video_family()
        with patch("core.inference.video._detect_load_family", return_value = fam), \
                patch("core.inference.diffusion_component_sizes.video_components_gb", return_value = (28.0, 10.0, 0.5)), \
                _parent_ids((2, 3)):
            need = gr.media_load_need_mib(arb.VIDEO, _video_request(root, "wan-Q4_K_M.gguf"), 0)
        companions = int(10.5 * 1000.0**3 / MIB)
        self.assertEqual(need.weights_mib, int(3000 * 1.05) + companions)
        self.assertEqual(need.headroom_mib, estimate_video_runtime_mib(width = 832, height = 480, num_frames = 81))
        self.assertEqual(need.kind, "video")
        self.assertEqual(dict(need), {2: need.weights_mib + need.headroom_mib})

    def test_video_h3_native_is_unknown(self):
        root = Path(self._tmp.name) / "H3-GGUF"
        _sparse(root / "h3-Q4_K_M.gguf", 10)
        with patch("core.inference.video._detect_load_family", return_value = self._video_family("minimax-h3")), \
                patch("core.inference.diffusion_component_sizes.video_components_gb", return_value = (1.0, 1.0, 1.0)), \
                _parent_ids((0,)):
            self.assertIsNone(gr.media_load_need_mib(arb.VIDEO, _video_request(root, "h3-Q4_K_M.gguf"), 0))


if __name__ == "__main__":
    unittest.main()
