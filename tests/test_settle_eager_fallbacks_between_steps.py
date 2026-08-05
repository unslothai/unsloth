# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A deferred compile-mode switch must be settled between training steps.

unsloth_zoo defers the switch to eager when torch.compile runs out of
recompile cache, because switching mid-call splits a non-reentrant activation
checkpoint region across two compile modes and the backward then dies with
"Something went unexpectedly wrong in activation checkpoint". Somebody has to
settle that debt where no region is half-packed, and the top of
`Trainer.training_step` is that point.
"""
import pytest

pytest.importorskip("transformers")
utils = pytest.importorskip("unsloth_zoo.temporary_patches.utils")

from unsloth.models._utils import patch_gradient_accumulation_fix

pytestmark = pytest.mark.skipif(
    not hasattr(utils, "apply_pending_eager_fallbacks"),
    reason = "unsloth_zoo without the deferred compile-mode switch",
)


@pytest.fixture
def FakeTrainer():
    """Enough of a Trainer for the patch to wrap and for the wrapper to call.

    `training_step` deliberately has no `num_items_in_batch`, so the gradient
    accumulation source rewrite skips it and the test sees only the settler.
    A fresh class per test keeps the patch's install-once flag honest.
    """
    class _FakeTrainer:
        def training_step(self, model, inputs):
            return "stepped"
    return _FakeTrainer


def test_training_step_settles_pending_eager_fallbacks(FakeTrainer, monkeypatch):
    calls = []
    monkeypatch.setattr(
        utils, "apply_pending_eager_fallbacks", lambda: calls.append(1),
    )
    patch_gradient_accumulation_fix(FakeTrainer)

    assert getattr(FakeTrainer, "_unsloth_settles_eager_fallbacks", False)
    assert FakeTrainer().training_step("model", "inputs") == "stepped"
    assert calls == [1], "every step must settle the pending switch exactly once"


def test_a_settler_failure_never_breaks_the_step(FakeTrainer, monkeypatch):
    def _boom():
        raise RuntimeError("no")
    monkeypatch.setattr(utils, "apply_pending_eager_fallbacks", _boom)
    patch_gradient_accumulation_fix(FakeTrainer)
    assert FakeTrainer().training_step("model", "inputs") == "stepped"


def test_the_settler_is_installed_only_once(FakeTrainer):
    patch_gradient_accumulation_fix(FakeTrainer)
    first = FakeTrainer.training_step
    patch_gradient_accumulation_fix(FakeTrainer)
    assert FakeTrainer.training_step is first, "re-patching must not stack wrappers"
