# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Starting training frees the GPU by stopping the export worker. Another account's running export is not
stopped: the start is refused the way a running video restyle refuses it. The caller's own export, and a
loaded but idle export checkpoint, are freed as before."""

from __future__ import annotations

import inspect

import pytest


class _Export:
    def __init__(self, active):
        self.active = active
        self.current_checkpoint = "some/model"
        self.is_vision = self.is_peft = False
        self.shut_down = False

    def is_export_active(self):
        return self.active

    def _shutdown_subprocess(self):
        self.shut_down = True


class _Freed(Exception):
    pass


def _teardown(monkeypatch, export, foreign):
    import core.export
    import core.training.account_jobs as account_jobs
    from routes import training

    monkeypatch.setattr(core.export, "get_export_backend", lambda: export)
    monkeypatch.setattr(account_jobs, "job_is_foreign", lambda service: foreign)

    def stop_after_the_export():
        raise _Freed

    monkeypatch.setattr(training, "_free_restyle_for_training", stop_after_the_export)
    return training


def test_another_accounts_running_export_refuses_the_start(monkeypatch):
    from routes.training_vram import ManagedEngineStillRunning

    export = _Export(active = True)
    training = _teardown(monkeypatch, export, foreign = True)
    with pytest.raises(ManagedEngineStillRunning, match = "Another account's export"):
        training._free_gpu_for_diffusion_training()
    assert export.shut_down is False


@pytest.mark.parametrize("active, foreign", [(True, False), (False, True), (False, False)])
def test_own_export_or_an_idle_checkpoint_is_freed_as_before(monkeypatch, active, foreign):
    export = _Export(active = active)
    training = _teardown(monkeypatch, export, foreign = foreign)
    with pytest.raises(_Freed):
        training._free_gpu_for_diffusion_training()
    assert export.shut_down is True


def test_llm_training_start_runs_the_same_check_before_freeing_anything():
    from routes import training

    source = inspect.getsource(training.start_training)
    hook = source[source.index("def _free_vram_for_training") :]
    assert hook.index("_refuse_a_foreign_export()") < hook.index("_shutdown_subprocess")
