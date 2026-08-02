# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
start_training()'s before_spawn hook must run iff a training subprocess is
actually spawned -- i.e. only after ALL synchronous validation (start guards,
config build, GPU-selection) passes. This protects the chat-VRAM unload from
firing for a start that is then refused (e.g. invalid gpu_ids -> 400).
"""

import unittest
from unittest.mock import MagicMock, patch

from core.training.training import TrainingBackend
from utils.hardware import DeviceType


class _DummyProcess:
    pid = 4321

    def start(self):
        return None


class _DummyThread:
    def start(self):
        return None


def _start(backend, hook):
    dummy_queue = object()
    with (
        patch("core.training.training.prepare_gpu_selection", return_value = ([0], {})),
        patch("core.training.training._CTX.Queue", side_effect = [dummy_queue, dummy_queue]),
        patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
        patch("core.training.training.threading.Thread", return_value = _DummyThread()),
    ):
        return backend.start_training(
            job_id = "before-spawn-test",
            before_spawn = hook,
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
        )


class TestBeforeSpawnHook(unittest.TestCase):
    def test_hook_runs_when_training_starts(self):
        backend = TrainingBackend()
        hook = MagicMock()
        ok = _start(backend, hook)
        self.assertTrue(ok)
        hook.assert_called_once()

    def test_hook_skipped_when_subprocess_already_alive(self):
        backend = TrainingBackend()
        backend._proc = MagicMock()
        backend._proc.is_alive.return_value = True
        hook = MagicMock()
        ok = _start(backend, hook)
        self.assertFalse(ok)
        hook.assert_not_called()  # never free chat VRAM for a refused start

    def test_hook_skipped_when_pump_thread_will_not_die(self):
        backend = TrainingBackend()
        stuck = MagicMock()
        stuck.is_alive.return_value = True
        stuck.join.return_value = None
        backend._pump_thread = stuck
        hook = MagicMock()
        ok = _start(backend, hook)
        self.assertFalse(ok)
        hook.assert_not_called()

    def test_hook_failure_does_not_block_start(self):
        backend = TrainingBackend()
        hook = MagicMock(side_effect = RuntimeError("boom"))
        ok = _start(backend, hook)
        self.assertTrue(ok)  # training still starts despite a hook error
        hook.assert_called_once()

    def test_hook_skipped_when_gpu_selection_rejects(self):
        # Invalid gpu_ids raise in prepare_gpu_selection (before the spawn), so the
        # hook must NOT run -- a refused start frees no chat/export VRAM.
        backend = TrainingBackend()
        hook = MagicMock()
        with (
            patch("utils.hardware.hardware.DEVICE", DeviceType.CUDA),
            patch(
                "core.training.training.prepare_gpu_selection",
                side_effect = ValueError("Invalid gpu_ids [99]"),
            ),
            patch("core.training.training._CTX.Process") as process_mock,
        ):
            with self.assertRaisesRegex(ValueError, "Invalid gpu_ids"):
                backend.start_training(
                    job_id = "before-spawn-test",
                    before_spawn = hook,
                    model_name = "unsloth/test",
                    training_type = "LoRA/QLoRA",
                    gpu_ids = [99],
                )
        hook.assert_not_called()
        process_mock.assert_not_called()

    def test_auto_placement_runs_after_hook(self):
        # Auto-selection ranks GPUs by free VRAM, so it must run AFTER the hook
        # frees export/chat -- otherwise training could be pinned onto a freed GPU
        # (or onto a GPU holding a chat model the probe decided to keep).
        order = []
        backend = TrainingBackend()
        hook = MagicMock(side_effect = lambda: order.append("hook"))

        def _placement(gpu_ids, **kwargs):
            order.append("placement")
            return ([0], {})

        with (
            patch("utils.hardware.hardware.DEVICE", DeviceType.CUDA),
            patch("core.training.training.prepare_gpu_selection", side_effect = _placement),
            patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
            patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
            patch("core.training.training.threading.Thread", return_value = _DummyThread()),
        ):
            ok = backend.start_training(
                job_id = "before-spawn-test",
                before_spawn = hook,
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
            )  # gpu_ids omitted -> auto mode
        self.assertTrue(ok)
        self.assertEqual(order, ["hook", "placement"])

    def test_explicit_placement_validated_before_hook(self):
        # Explicit gpu_ids are validated before the hook (so an invalid set 400s
        # without teardown); explicit placement is VRAM-independent.
        order = []
        backend = TrainingBackend()
        hook = MagicMock(side_effect = lambda: order.append("hook"))

        def _placement(gpu_ids, **kwargs):
            order.append("placement")
            return (list(gpu_ids), {})

        with (
            patch("utils.hardware.hardware.DEVICE", DeviceType.CUDA),
            patch("core.training.training.prepare_gpu_selection", side_effect = _placement),
            patch("core.training.training._CTX.Queue", side_effect = [object(), object()]),
            patch("core.training.training._CTX.Process", return_value = _DummyProcess()),
            patch("core.training.training.threading.Thread", return_value = _DummyThread()),
        ):
            ok = backend.start_training(
                job_id = "before-spawn-test",
                before_spawn = hook,
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                gpu_ids = [5],
            )
        self.assertTrue(ok)
        self.assertEqual(order, ["placement", "hook"])


if __name__ == "__main__":
    unittest.main()
