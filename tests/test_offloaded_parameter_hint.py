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

"""A save that fails because the model was offloaded should say so.

Offloaded parameters sit on the meta device, and saving then dies inside
accelerate with an error that names neither the model nor the offload. The
hint is appended to that error, never substituted for it, and stays empty
unless a meta parameter is really present.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from unsloth.save import _offloaded_parameter_hint  # noqa: E402


class _Model:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params)


def _p(device):
    return torch.nn.Parameter(torch.zeros(2, device = device), requires_grad = False)


def test_a_meta_parameter_produces_a_hint():
    m = _Model([("model.layers.0.mlp.down_proj.weight", _p("meta"))])
    hint = _offloaded_parameter_hint(m)
    assert hint
    assert "meta device" in hint


def test_the_hint_names_the_offending_parameter():
    """So the reader can tell which part of the model was offloaded."""
    m = _Model([("model.layers.31.mlp.experts.w1", _p("meta"))])
    assert "model.layers.31.mlp.experts.w1" in _offloaded_parameter_hint(m)


def test_the_hint_states_the_remedy():
    m = _Model([("a", _p("meta"))])
    hint = _offloaded_parameter_hint(m)
    assert "did not fit" in hint
    assert "device_map" in hint or "large enough" in hint


def test_it_reports_a_few_names_not_all_of_them():
    """A 30B MoE has thousands of offloaded tensors; pasting them all would
    bury the actual error."""
    m = _Model([(f"layer.{i}.weight", _p("meta")) for i in range(500)])
    hint = _offloaded_parameter_hint(m)
    assert hint.count("layer.") <= 3
    assert len(hint) < 600


def test_a_mix_of_real_and_meta_still_fires():
    """Partial offload is the normal case -- only some layers move."""
    m = _Model([("good", _p("cpu")), ("bad", _p("meta"))])
    assert _offloaded_parameter_hint(m)


def test_a_fully_resident_model_gets_no_hint():
    """The mislabelling risk. An unrelated save failure must not be blamed
    on an offload that never happened."""
    m = _Model([("a", _p("cpu")), ("b", _p("cpu"))])
    assert _offloaded_parameter_hint(m) == ""


def test_a_model_with_no_parameters_gets_no_hint():
    assert _offloaded_parameter_hint(_Model([])) == ""


def test_a_model_without_named_parameters_gets_no_hint():
    class Odd:
        pass

    assert _offloaded_parameter_hint(Odd()) == ""


def test_none_gets_no_hint():
    assert _offloaded_parameter_hint(None) == ""


def test_a_raising_named_parameters_gets_no_hint():
    """A diagnostic must never replace the real error with its own."""

    class Boom:
        def named_parameters(self):
            raise RuntimeError("model is in a bad state")

    assert _offloaded_parameter_hint(Boom()) == ""


def test_a_parameter_with_no_device_does_not_crash():
    class NoDevice:
        device = None

    m = _Model([("weird", NoDevice())])
    assert _offloaded_parameter_hint(m) == ""


SRC = (ROOT / "unsloth" / "save.py").read_text(encoding = "utf-8")


def test_both_save_failure_paths_use_it():
    """The GGUF export can fail at the merge step or at the plain save step,
    and offloading breaks both."""
    assert SRC.count("_offloaded_parameter_hint(self)") == 2


def test_the_original_error_is_still_reported():
    """The hint is added TO the error, never instead of it.

    Anchored on the message text alone, not on a whole string literal: the
    repo's ruff hook may merge the hint into the same f-string or split it out
    again, and either shape satisfies what this is actually checking.
    """
    for anchor in ("Failed to save/merge model: ", "Failed to save model: "):
        # All occurrences, not the first:
        windows = []
        i = SRC.find(anchor)
        assert i != -1, anchor
        while i != -1:
            windows.append(SRC[i : i + 200])
            i = SRC.find(anchor, i + 1)
        # `{e}` is empty when the exception has no args, so the type-leading form counts too.
        assert any(
            ("{e}" in w or "_describe_exception(e)" in w) and "_offloaded_parameter_hint" in w
            for w in windows
        ), anchor


def test_it_still_raises_runtimeerror():
    """Callers catch RuntimeError; changing the type would break them."""
    i = SRC.index("_offloaded_parameter_hint(self)")
    assert "raise RuntimeError(" in SRC[max(0, i - 300) : i]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
