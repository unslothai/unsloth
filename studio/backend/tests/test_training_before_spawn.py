# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
start_training()'s before_spawn hook must run iff a training subprocess is
actually spawned -- i.e. only after the start guards (no live subprocess, no
lingering pump thread) pass. This protects the chat-VRAM unload from firing for
a start that is then refused.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.training.training import TrainingBackend


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


if __name__ == "__main__":
    unittest.main()
