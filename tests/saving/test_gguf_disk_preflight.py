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

"""The disk preflight in front of a GGUF export, and its wiring.

Three notebooks -- Gemma4 (26B A4B) Vision, Gemma4 (31B) Vision and Qwen3 32B
-- trained, ran inference and wrote a complete `merged_16bit`, then died
partway through a GGUF shard with the VM out of space. The size estimate in
front of them counted the model twice. The real peak is four artefacts on one
filesystem: the pre-warmed base in the Hugging Face cache, the 16-bit merge,
the intermediate GGUF and the quants.

The arithmetic itself is unsloth_zoo's, and is tested in
`unsloth_zoo/tests/test_disk_utils_kaggle.py`. What is tested here is
unsloth's use of it: which of the three outcomes it picks (proceed / drop the
pre-warm / refuse), that it never blocks on an unmeasurable number, and that
nothing it computes leaks into the kwargs of a save.

The sizing functions are stubbed rather than exercised, so this runs on CPU,
in milliseconds, against any installed unsloth_zoo.
"""

import os

import pytest

from unsloth import save as S

GB = 1024**3


class _FakeModel:
    """Not a PeftModel; the preflight is called with an explicit needs_merge."""

    pass


@pytest.fixture
def stub_sizing(monkeypatch):
    """Replace the size and free-space calls with numbers a test dictates.

    Returns a setter. `need` is the export without the cached base, and
    `need_with_cache` the export with it.
    """
    state = {"need": 0, "need_with_cache": 0, "free": None, "redirect": (None, None)}

    def fake_estimate(**kwargs):
        return state["need_with_cache"] if kwargs.get("base_cache_copy") else state["need"]

    def fake_free(path):
        return state["free"]

    def fake_redirect(save_directory, need_bytes = 0, what = "export"):
        target, message = state["redirect"]
        if message is None:
            return save_directory, None
        return target, message

    monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
    monkeypatch.setattr(S, "free_bytes", fake_free)
    monkeypatch.setattr(S, "kaggle_tmp_redirect", fake_redirect)
    monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
    monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)

    def configure(**kwargs):
        state.update(kwargs)
        return state

    return configure


class TestNormalizeQuantizationMethods:
    def test_friendly_aliases(self):
        assert S._normalize_quantization_methods("fast_quantized") == ["q8_0"]
        assert S._normalize_quantization_methods("quantized") == ["q4_k_m"]
        assert S._normalize_quantization_methods("not_quantized") == ["f16"]

    def test_list_and_case(self):
        assert S._normalize_quantization_methods(["Q4_K_M", "Q8_0"]) == ["q4_k_m", "q8_0"]

    def test_none_entries_become_the_default(self):
        assert S._normalize_quantization_methods([None]) == ["q8_0"]

    def test_none_and_junk_are_empty(self):
        assert S._normalize_quantization_methods(None) == []
        assert S._normalize_quantization_methods(object()) == []


class TestPreflightOutcomes:
    def test_plenty_of_room_changes_nothing(self, stub_sizing):
        stub_sizing(need = 30 * GB, need_with_cache = 44 * GB, free = 500 * GB)
        directory, prewarm_ok = S._preflight_gguf_disk(
            _FakeModel(), "model", "q4_k_m", first_conversion = "f16"
        )
        assert directory == "model"
        assert prewarm_ok is True

    def test_fits_only_without_the_prewarm(self, stub_sizing, capsys):
        """The Gemma4 31B case: drop the cache, keep the export."""
        stub_sizing(need = 148 * GB, need_with_cache = 206 * GB, free = 174 * GB)
        directory, prewarm_ok = S._preflight_gguf_disk(
            _FakeModel(), "model", "q8_0", first_conversion = "f16"
        )
        assert directory == "model"
        assert prewarm_ok is False, "refusing here would decline an export that fits"
        out = capsys.readouterr().out
        assert "pre-warm" in out

    def test_does_not_fit_at_all_raises_with_the_numbers(self, stub_sizing):
        stub_sizing(need = 40 * GB, need_with_cache = 54 * GB, free = 19 * GB)
        with pytest.raises(RuntimeError) as excinfo:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", first_conversion = "f16")
        message = str(excinfo.value)
        assert "40.0GB" in message and "19.0GB" in message
        # Actionable, not merely correct.
        assert "push_to_hub_gguf" in message
        assert "UNSLOTH_DISK_PREFLIGHT=0" in message
        assert "q4_k_m" in message

    def test_unmeasurable_model_never_blocks(self, stub_sizing, monkeypatch):
        """A guard that blocks on a guess is worse than no guard.

        With no size to work from there is also nothing to justify moving the
        files, so the Kaggle redirect must not even be consulted -- otherwise
        an unmeasurable model on Kaggle gets relocated on the strength of
        `need_bytes = 0`.
        """
        redirect_calls = []
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda *a, **k: redirect_calls.append((a, k)) or (a[0], None),
        )
        stub_sizing(need = 0, need_with_cache = 0, free = 0)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)
        assert redirect_calls == []

    def test_unmeasurable_disk_never_blocks(self, stub_sizing):
        stub_sizing(need = 10**15, need_with_cache = 10**15, free = None)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    def test_sizing_that_raises_never_blocks(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", boom)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
    def test_kill_switch(self, stub_sizing, monkeypatch, value):
        stub_sizing(need = 10**15, need_with_cache = 10**15, free = 1)
        monkeypatch.setenv("UNSLOTH_DISK_PREFLIGHT", value)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    def test_prewarm_already_off_is_not_offered_as_a_saving(self, stub_sizing, monkeypatch):
        """With the pre-warm already disabled there is no cache copy to drop."""
        monkeypatch.setenv("UNSLOTH_PREWARM_HUB_CACHE", "0")
        stub_sizing(need = 148 * GB, need_with_cache = 206 * GB, free = 174 * GB)
        directory, prewarm_ok = S._preflight_gguf_disk(
            _FakeModel(), "model", "q8_0", first_conversion = "f16"
        )
        assert (directory, prewarm_ok) == ("model", True)

    def test_non_peft_export_has_no_merge_and_no_cache_copy(self, stub_sizing, monkeypatch):
        seen = {}

        def fake_estimate(**kwargs):
            seen.update(kwargs)
            return 1

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        stub_sizing(free = 10 * GB)
        S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", needs_merge = False)
        assert seen["needs_merge"] is False
        # No merge means no pre-warm, so no cache copy is ever priced in.
        assert seen.get("base_cache_copy") in (None, False)


class TestKaggleRedirectWiring:
    def test_redirect_is_taken_and_announced_once(self, stub_sizing, capsys):
        stub_sizing(
            need = 34 * GB,
            need_with_cache = 48 * GB,
            free = 1000 * GB,
            redirect = ("/tmp/unsloth_saves/model", "Unsloth: moved, /tmp is not saved"),
        )
        directory, prewarm_ok = S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert directory == "/tmp/unsloth_saves/model"
        out = capsys.readouterr().out
        assert out.count("Unsloth: moved") == 1

    def test_the_redirected_directory_is_what_gets_size_checked(self, stub_sizing, monkeypatch):
        """A redirect that is not re-measured would report the old disk."""
        probed = []

        def fake_free(path):
            probed.append(path)
            return 1000 * GB

        stub_sizing(
            need = 34 * GB,
            need_with_cache = 48 * GB,
            redirect = ("/tmp/unsloth_saves/model", "moved"),
        )
        monkeypatch.setattr(S, "free_bytes", fake_free)
        S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert probed == ["/tmp/unsloth_saves/model"]

    def test_merge_preflight_never_rewrites_a_repo_id(self, monkeypatch):
        """`push_to_hub=True` makes save_directory "user/model", not a path."""
        called = []
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: called.append(a) or ("/tmp/x", "moved")
        )
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", lambda **k: 10 * GB)
        result = S._preflight_merge_disk(
            _FakeModel(), "danielhanchen/my-model", "merged_16bit", push_to_hub = True
        )
        assert result == "danielhanchen/my-model"
        assert called == []

    @pytest.mark.parametrize("save_method", ["lora", "merged_4bit", "fp8"])
    def test_merge_preflight_only_applies_to_full_16bit_exports(self, monkeypatch, save_method):
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/x", "moved")
        )
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", lambda **k: 10 * GB)
        assert S._preflight_merge_disk(_FakeModel(), "model", save_method) == "model"

    def test_merge_preflight_takes_the_redirect(self, monkeypatch, capsys):
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/unsloth_saves/model", "moved")
        )
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", lambda **k: 10 * GB)
        assert S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit") == (
            "/tmp/unsloth_saves/model"
        )
        assert "moved" in capsys.readouterr().out

    def test_merge_preflight_never_raises(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", boom)
        assert S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit") == "model"


class TestPrewarmContextManager:
    def test_disables_and_restores_a_previous_value(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PREWARM_HUB_CACHE", "1")
        with S._hub_cache_prewarm_disabled(True):
            assert os.environ["UNSLOTH_PREWARM_HUB_CACHE"] == "0"
        assert os.environ["UNSLOTH_PREWARM_HUB_CACHE"] == "1"

    def test_removes_the_variable_it_invented(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        with S._hub_cache_prewarm_disabled(True):
            assert os.environ["UNSLOTH_PREWARM_HUB_CACHE"] == "0"
        assert "UNSLOTH_PREWARM_HUB_CACHE" not in os.environ

    def test_restores_on_an_exception(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PREWARM_HUB_CACHE", "1")
        with pytest.raises(ValueError):
            with S._hub_cache_prewarm_disabled(True):
                raise ValueError("merge blew up")
        assert os.environ["UNSLOTH_PREWARM_HUB_CACHE"] == "1"

    def test_no_op_when_not_disabling(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        with S._hub_cache_prewarm_disabled(False):
            assert "UNSLOTH_PREWARM_HUB_CACHE" not in os.environ


class TestNoLeakIntoSaveKwargs:
    def test_preflight_locals_never_reach_unsloth_generic_save(self):
        """`arguments = dict(locals())` snapshots every local in scope.

        The preflight adds one, and `unsloth_generic_save` takes named
        keyword arguments, so forgetting the matching `del` turns every GGUF
        export into `TypeError: unexpected keyword argument`. Read the
        function rather than run it: reaching this line for real needs a
        loaded 4-bit model and a llama.cpp build.
        """
        import inspect

        source = inspect.getsource(S.unsloth_save_pretrained_gguf)
        assert "_gguf_prewarm_ok" in source
        assert 'del arguments["_gguf_prewarm_ok"]' in source

        accepted = set(inspect.signature(S.unsloth_generic_save).parameters)
        deleted = set()
        for line in source.splitlines():
            line = line.strip()
            if line.startswith('del arguments["'):
                deleted.add(line.split('"')[1])
        # Every local that exists at the snapshot point and is not a parameter
        # of unsloth_generic_save has to be deleted before the call.
        introduced = {"self", "base_model_name", "model_name", "is_vlm", "is_processor",
                      "is_gpt_oss", "_gguf_prewarm_ok", "quantization_method",
                      "first_conversion", "imatrix_file"}
        for name in introduced - accepted:
            assert name in deleted, f"{name} would be passed to unsloth_generic_save"
