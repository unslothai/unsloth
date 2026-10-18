# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for core/inference/gpu_residency.py: pricing image and video loads that stay on a GPU beside chat."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import utils.hardware.hardware as _hw_module
from core.inference import gpu_arbiter as arb
from core.inference import gpu_residency as gr
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_memory import estimate_image_runtime_mib, estimate_video_runtime_mib
from utils.hardware import DeviceType

MIB = 1024 * 1024


class _GpuCacheResetMixin:
    """Reset module-level GPU caches between tests to prevent state leaks."""

    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


def _parent_ids(ids):
    return patch("utils.hardware.hardware.get_parent_visible_gpu_ids", return_value = list(ids))


def _image_request(
    model_path,
    gguf_filename = "z-image-turbo-Q4_K_M.gguf",
    **kw,
):
    fields = dict(
        model_kind = None, base_repo = None, family_override = None, text_encoder_file = None, vae_file = None
    )
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
            patch(
                "core.inference.diffusion_engine_router.predict_engine", return_value = "diffusers"
            ),
            patch.object(
                DiffusionBackend,
                "_companion_cache_bytes",
                staticmethod(lambda base, *a, **k: 6000 * MIB),
            ),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()
        super().tearDown()

    def _image(
        self,
        ordinal,
        parent = (2, 3),
        request = None,
    ):
        with _parent_ids(parent):
            return gr.media_load_need_mib(
                arb.DIFFUSION, request or _image_request(self.root), ordinal
            )

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
        self.assertIsNone(
            self._image(0, request = _image_request(self.root, "z-image-turbo-Q8_0.gguf"))
        )

    def test_uncached_companions_are_unknown(self):
        with patch.object(
            DiffusionBackend, "_companion_cache_bytes", staticmethod(lambda *a, **k: 0)
        ):
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
        with patch(
            "utils.hardware.hardware.get_parent_visible_gpu_ids", side_effect = OSError("smi")
        ):
            self.assertIsNone(gr.media_load_need_mib(arb.DIFFUSION, _image_request(self.root), 0))

    def test_unknown_kind_is_unknown(self):
        with _parent_ids((0,)):
            self.assertIsNone(gr.media_load_need_mib("chat", _image_request(self.root), 0))

    def _video_family(self, name = "wan-2.1"):
        return SimpleNamespace(
            name = name,
            base_repo = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            is_moe = False,
            resolution_presets = ((832, 480),),
            default_num_frames = 81,
        )

    def test_video_gguf_prices_checkpoint_plus_companion_table(self):
        root = Path(self._tmp.name) / "Wan-GGUF"
        _sparse(root / "wan-Q4_K_M.gguf", 3000)
        fam = self._video_family()
        with (
            patch("core.inference.video._detect_load_family", return_value = fam),
            patch(
                "core.inference.diffusion_component_sizes.video_components_gb",
                return_value = (28.0, 10.0, 0.5),
            ),
            _parent_ids((2, 3)),
        ):
            need = gr.media_load_need_mib(arb.VIDEO, _video_request(root, "wan-Q4_K_M.gguf"), 0)
        companions = int(10.5 * 1000.0**3 / MIB)
        self.assertEqual(need.weights_mib, int(3000 * 1.05) + companions)
        self.assertEqual(
            need.headroom_mib, estimate_video_runtime_mib(width = 832, height = 480, num_frames = 81)
        )
        self.assertEqual(need.kind, "video")
        self.assertEqual(dict(need), {2: need.weights_mib + need.headroom_mib})

    def test_video_h3_native_is_unknown(self):
        root = Path(self._tmp.name) / "H3-GGUF"
        _sparse(root / "h3-Q4_K_M.gguf", 10)
        with (
            patch(
                "core.inference.video._detect_load_family",
                return_value = self._video_family("minimax-h3"),
            ),
            patch(
                "core.inference.diffusion_component_sizes.video_components_gb",
                return_value = (1.0, 1.0, 1.0),
            ),
            _parent_ids((0,)),
        ):
            self.assertIsNone(
                gr.media_load_need_mib(arb.VIDEO, _video_request(root, "h3-Q4_K_M.gguf"), 0)
            )


def _util(*devices, index_kind = "physical"):
    rows = [dict(index = i, vram_total_gb = t, vram_used_gb = u) for i, t, u in devices]
    return patch(
        "utils.hardware.get_visible_gpu_utilization",
        return_value = {"index_kind": index_kind, "devices": rows},
    )


def _cuda(device = DeviceType.CUDA):
    return patch("utils.hardware.get_device", return_value = device)


def _need(
    per_gpu,
    weights = 3000,
    headroom = 2000,
    kind = "video",
):
    return gr.MediaNeed(per_gpu, weights_mib = weights, headroom_mib = headroom, kind = kind)


class _ArbiterResetMixin:
    """Empty arbiter state for each test, restored afterwards."""

    _FIELDS = ("_owner", "_residents", "_media_needs", "_accounts")

    def setUp(self):
        super().setUp()
        self._saved = {name: getattr(arb, name) for name in self._FIELDS}
        arb._owner, arb._residents, arb._media_needs, arb._accounts = None, {}, {}, {}

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(arb, name, value)
        super().tearDown()

    def _resident(
        self,
        owner,
        gpus,
        need = None,
    ):
        arb._residents[owner] = frozenset(gpus)
        arb._owner = owner
        if need is not None:
            arb._media_needs[owner] = need


FREE_80_20 = 80 * 1024 - 20 * 1024  # free MiB on an 80 GB card with 20 GB used


class TestCanLoadAlongside(_ArbiterResetMixin, unittest.TestCase):
    def _check(
        self,
        need,
        owner = arb.VIDEO,
        device = DeviceType.CUDA,
        util = None,
    ):
        with _cuda(device), util or _util((0, 80.0, 20.0)):
            return gr.can_load_alongside(owner, need)

    def test_default_deny_reasons(self):
        need = _need({0: 1000})
        self.assertEqual(
            gr.can_load_alongside(arb.VIDEO, None), (False, {"reason": "estimate_unavailable"})
        )
        self.assertEqual(gr.can_load_alongside(arb.VIDEO, {})[1]["reason"], "estimate_unavailable")
        for device in (DeviceType.MLX, DeviceType.CPU, DeviceType.XPU):
            self.assertEqual(
                self._check(need, device = device), (False, {"reason": "non_accelerator"})
            )
        with (
            _cuda(),
            patch("utils.hardware.get_visible_gpu_utilization", side_effect = RuntimeError("smi")),
        ):
            self.assertEqual(
                gr.can_load_alongside(arb.VIDEO, need), (False, {"reason": "probe_error"})
            )
        for kind in ("relative", "unresolved"):
            self.assertEqual(
                self._check(need, util = _util((0, 80.0, 20.0), index_kind = kind))[1]["reason"],
                "unmapped_gpus",
            )
        self.assertEqual(self._check(_need({5: 1000}))[1], {"reason": "gpu_unknown", "gpu": 5})

    def test_unified_memory_is_denied(self):
        need = _need({0: 1000})
        for flag in ("unified_memory", "shared_memory"):
            util = {
                "index_kind": "physical",
                "devices": [{"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 1.0, flag: True}],
            }
            with _cuda(), patch("utils.hardware.get_visible_gpu_utilization", return_value = util):
                self.assertEqual(
                    gr.can_load_alongside(arb.VIDEO, need),
                    (False, {"reason": "unified_memory", "gpu": 0}),
                )
        self.assertEqual(self._check(need, util = _util((0, 0, 0.0)))[1]["reason"], "unified_memory")

    def test_margin_boundary(self):
        fits, info = self._check(_need({0: FREE_80_20 - gr.FIT_MARGIN_MIB}))
        self.assertTrue(fits)
        self.assertEqual(info["reason"], "fits")
        fits, info = self._check(_need({0: FREE_80_20 - gr.FIT_MARGIN_MIB + 1}))
        self.assertFalse(fits)
        self.assertEqual((info["reason"], info["short_mib"]), ("short", {0: 1}))

    def test_one_short_gpu_fails_the_whole_load(self):
        util = _util((0, 80.0, 20.0), (1, 24.0, 20.0))
        fits, info = self._check(_need({0: 1000, 1: 4000}), util = util)
        self.assertFalse(fits)
        self.assertEqual(list(info["short_mib"]), [1])
        self.assertEqual(info["free_mib"], {0: FREE_80_20, 1: 4 * 1024})

    def test_other_resident_headroom_is_subtracted(self):
        self._resident(arb.VIDEO, {0}, _need({0: 5000}, weights = 3000, headroom = 2000))
        _, info = self._check(_need({0: 1000}, kind = "other"), owner = arb.DIFFUSION)
        self.assertEqual(info["free_mib"], {0: FREE_80_20 - 2000})

    def test_a_resident_still_loading_counts_in_full(self):
        # Its weights aren't in live free VRAM yet, so a second media load can't be admitted into that gap.
        self._resident(arb.VIDEO, {0}, _need({0: 5000}, weights = 3000, headroom = 2000))
        with patch.object(gr, "_media_busy", lambda owner: (owner == arb.VIDEO, False)):
            _, info = self._check(_need({0: 1000}, kind = "other"), owner = arb.DIFFUSION)
        self.assertEqual(info["free_mib"], {0: FREE_80_20 - 5000})

    def test_own_record_adds_back_its_weights(self):
        self._resident(arb.VIDEO, {0}, _need({0: 5000}, weights = 3000, headroom = 2000))
        _, info = self._check(_need({0: 1000}), owner = arb.VIDEO)
        self.assertEqual(info["free_mib"], {0: FREE_80_20 + 3000})

    def test_image_that_would_stream_is_denied(self):
        plan = MagicMock(return_value = "plan")
        with (
            _parent_ids((0,)),
            patch("core.inference.diffusion_memory.plan_diffusion_memory", plan),
            patch(
                "core.inference.diffusion_memory.plan_keeps_transformer_resident",
                return_value = False,
            ),
        ):
            fits, info = self._check(_need({0: 1000}, kind = "image"), owner = arb.DIFFUSION)
        self.assertFalse(fits)
        self.assertEqual(info["reason"], "would_stream")
        kw = plan.call_args.kwargs
        self.assertIsNone(kw["requested_mode"])
        self.assertEqual((kw["model_dense_mib"], kw["runtime_headroom_mib"]), (3000, 2000))
        self.assertEqual(
            (kw["device_memory"].device, kw["device_memory"].free_mib), ("cuda:0", FREE_80_20)
        )

    def test_image_kept_resident_fits_and_video_skips_the_planner(self):
        with (
            _parent_ids((0,)),
            patch(
                "core.inference.diffusion_memory.plan_keeps_transformer_resident", return_value = True
            ),
        ):
            self.assertTrue(self._check(_need({0: 1000}, kind = "image"), owner = arb.DIFFUSION)[0])
        with patch(
            "core.inference.diffusion_memory.plan_diffusion_memory",
            side_effect = AssertionError("planner"),
        ):
            self.assertTrue(self._check(_need({0: 1000}, kind = "video"))[0])


class TestMediaHeldMib(_ArbiterResetMixin, unittest.TestCase):
    def test_nothing_resident_is_empty(self):
        self.assertEqual(gr.media_held_mib(), {})
        arb._media_needs[arb.VIDEO] = _need({0: 5000})  # a record with no resident
        self.assertEqual(gr.media_held_mib(), {})

    def test_weights_plus_headroom_summed_per_gpu(self):
        self._resident(arb.CHAT, {0, 1})
        self._resident(arb.VIDEO, {0}, _need({0: 5000}))
        self._resident(arb.DIFFUSION, {0, 1}, _need({0: 7000, 1: 1000}, kind = "image"))
        self.assertEqual(gr.media_held_mib(), {0: 12000, 1: 1000})

    def test_non_resident_record_is_ignored(self):
        self._resident(arb.DIFFUSION, {0}, _need({0: 7000}, kind = "image"))
        arb._media_needs[arb.VIDEO] = _need({0: 5000})
        self.assertEqual(gr.media_held_mib(), {0: 7000})

    def test_corrupt_record_never_raises(self):
        self._resident(arb.VIDEO, {0})
        arb._media_needs[arb.VIDEO] = {0: "lots"}
        self.assertEqual(gr.media_held_mib(), {})


class TestFreeForOrder(_ArbiterResetMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.log, self.evicted = [], []
        self.loading, self.generating, self.foreign, self.fits = (
            set(),
            set(),
            {},
            MagicMock(return_value = (False, {})),
        )
        recorder = lambda name: (lambda: self.log.append(name))
        self.monitor = MagicMock()
        self._patches = [
            patch.dict(
                arb._EVICTORS,
                {
                    arb.CHAT: recorder("chat"),
                    arb.DIFFUSION: recorder("image"),
                    arb.VIDEO: recorder("video"),
                },
            ),
            patch.object(
                arb,
                "resident_slots",
                side_effect = lambda n: [
                    {"kind": gr._MEDIA_KIND.get(n, "chat"), "model": f"{n}-model"}
                ],
            ),
            patch.object(
                gr, "_media_busy", side_effect = lambda n: (n in self.loading, n in self.generating)
            ),
            patch.object(gr, "chat_load_in_flight", side_effect = lambda: arb.CHAT in self.loading),
            patch.object(gr, "foreign_on", side_effect = lambda n, a: self.foreign.get(n, 0)),
            patch.object(gr, "can_load_alongside", self.fits),
            patch.object(
                gr, "_kept_slot_report", return_value = {"kind": "chat", "model": "kept-model"}
            ),
            patch(
                "core.inference.video_minimax_h3.release_h3_native_servers",
                side_effect = lambda reason: self.log.append("h3") or 1,
            ),
            patch(
                "core.inference.model_slots.eviction_victims",
                side_effect = lambda *a: self.log.append("kept") or ["slot"],
            ),
            patch(
                "core.inference.model_slots.evict_these",
                side_effect = lambda v: self.log.append("evict_these"),
            ),
            patch("core.inference.api_monitor.api_monitor", self.monitor),
            patch("utils.account_context.current_account_id", return_value = "me"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        super().tearDown()

    def _free(
        self,
        owner = arb.DIFFUSION,
        gpus = (0,),
    ):
        return gr.free_for(
            owner, gpus, "test", self.evicted, need_mib = _need({0: 1000}, kind = "image")
        )

    def test_idle_media_before_kept_then_active_chat(self):
        self._resident(arb.VIDEO, {0})
        self._resident(arb.CHAT, {0})
        freed = self._free()
        self.assertEqual(self.log, ["h3", "video", "kept", "evict_these", "chat"])
        self.assertEqual(freed, ["video-server", "video-model", "kept-model", "chat-model"])
        self.assertEqual(arb._view(), {})
        self.assertEqual(
            self.evicted,
            [
                {"kind": "video", "model": "video-model"},
                {"kind": "chat", "model": "kept-model"},
                {"kind": "chat", "model": "chat-model"},
            ],
        )
        self.monitor.record_lifecycle.assert_any_call(
            event = "unload", model = "video-model", reason = "evicted"
        )
        self.assertEqual(self.monitor.record_lifecycle.call_count, 3)

    def test_loading_resident_goes_first(self):
        self._resident(arb.VIDEO, {0})
        self._resident(arb.CHAT, {0})
        self.loading = {arb.VIDEO, arb.CHAT}
        self._free()
        self.assertEqual(self.log[:3], ["video", "chat", "h3"])
        self.assertNotIn("kept", self.log)  # CHAT is gone, so no kept slots overlap

    def test_stops_once_it_fits(self):
        self._resident(arb.VIDEO, {0})
        self._resident(arb.CHAT, {0})
        self.fits.side_effect = [(False, {}), (True, {})]
        self.assertEqual(self._free(), ["video-server", "video-model"])
        self.assertEqual(self.log, ["h3", "video"])
        self.assertIn(arb.CHAT, arb._view())

    def test_disjoint_gpus_and_busy_media_are_untouched(self):
        self._resident(arb.VIDEO, {1})
        self._resident(arb.CHAT, {0})
        self._free()
        self.assertNotIn("video", self.log)
        self._resident(arb.VIDEO, {0})
        self.generating = {arb.VIDEO}
        self.log.clear()
        self._resident(arb.CHAT, {0})
        self._free()
        self.assertEqual(self.log, ["h3", "kept", "evict_these", "chat"])
        self.assertIn(arb.VIDEO, arb._view())

    def test_foreign_generation_blocks_the_eviction(self):
        self._resident(arb.CHAT, {0})
        self.foreign = {arb.CHAT: 2}
        with self.assertRaises(arb.GpuBusyForAnotherAccountError) as ctx:
            self._free()
        self.assertEqual((ctx.exception.owner, ctx.exception.active), (arb.CHAT, 2))
        self.assertNotIn("chat", self.log)
        self.assertIn(arb.CHAT, arb._view())


class TestOtherPlannedIncludesMedia(_ArbiterResetMixin, unittest.TestCase):
    def test_media_held_is_added_for_a_registered_backend(self):
        from core.inference import llama_cpp
        from core.inference.llama_cpp import LlamaCppBackend

        mine, other, stray = (LlamaCppBackend(manages_processes = False) for _ in range(3))
        other._process = object()
        other._planned_vram_mib = {0: 4000}
        self._resident(arb.DIFFUSION, {0, 1}, _need({0: 7000, 1: 500}, kind = "image"))
        for backend in (mine, other):
            llama_cpp.register_serving_backend(backend)
        try:
            self.assertEqual(mine._other_planned_vram_mib(), {0: 11000, 1: 500})
            self.assertEqual(stray._other_planned_vram_mib(), {})
        finally:
            for backend in (mine, other):
                llama_cpp.unregister_serving_backend(backend)


class TestChatLoadingRefusal(unittest.TestCase):
    def test_api_shape(self):
        exc = gr.chat_loading_http_exception("/api/inference/images/load")
        self.assertEqual(exc.status_code, 409)
        self.assertEqual(exc.headers, {"Retry-After": "15"})
        self.assertEqual(set(exc.detail), {"error", "message", "retry_after"})
        self.assertEqual((exc.detail["error"], exc.detail["retry_after"]), ("gpu_busy", 15))
        self.assertEqual(gr.ChatLoadInFlightRefusal().as_http_exception().detail, exc.detail)

    def test_openai_shape(self):
        for exc in (
            gr.chat_loading_http_exception("/v1/images/generations"),
            gr.chat_loading_http_exception("/api/x", openai_errors = True),
        ):
            self.assertEqual(exc.headers, {"Retry-After": "15"})
            self.assertEqual(exc.detail["error"]["code"], "gpu_busy")
            message = exc.detail["error"]["message"]
            self.assertNotIn("is set to Always", message)
            self.assertNotIn("is set to keep loaded", message)


if __name__ == "__main__":
    unittest.main()
