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

import math
import os

import pytest

from unsloth import save as S

GB = 1024**3


def _with_merge_headroom(n_bytes):
    """What the merge preflight asks for, given a raw output size.

    unsloth_zoo's merge guard compares against `int(free * 0.95)`, so the
    preflight has to ask for the same effective figure. The 0.95 is written
    out rather than read from `S`, so dropping the headroom fails here.
    """
    return math.ceil(n_bytes / 0.95)


class _FakeModel:
    """Not a PeftModel; the preflight is called with an explicit needs_merge."""


class _ModelWithEmbeddings:
    """Answers the two embedding getters a weight-only export leaves alone."""

    class _Weight:
        def __init__(self, numel):
            self._numel = numel

        def numel(self):
            return self._numel

    class _Embedding:
        def __init__(self, weight):
            self.weight = weight

    def __init__(
        self,
        input_numel,
        output_numel = 0,
        tied = False,
    ):
        self._input = self._Embedding(self._Weight(input_numel))
        if tied:
            self._output = self._input
        elif output_numel:
            self._output = self._Embedding(self._Weight(output_numel))
        else:
            self._output = None

    def get_input_embeddings(self):
        return self._input

    def get_output_embeddings(self):
        return self._output


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

    def fake_redirect(
        save_directory,
        need_bytes = 0,
        what = "export",
    ):
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

    @pytest.mark.parametrize("save_method", ["lora", "merged_4bit"])
    def test_merge_preflight_only_applies_to_exports_that_write_16bit(
        self, monkeypatch, save_method
    ):
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/x", "moved"))
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        assert S._preflight_merge_disk(_FakeModel(), "model", save_method) == "model"

    def test_merge_preflight_takes_the_redirect(self, monkeypatch, capsys):
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/unsloth_saves/model", "moved")
        )
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        assert S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit") == (
            "/tmp/unsloth_saves/model"
        )
        assert "moved" in capsys.readouterr().out

    def test_merge_preflight_never_raises(self, monkeypatch):
        def boom(model):
            raise RuntimeError("no")

        monkeypatch.setattr(S, "model_16bit_bytes", boom)
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
        introduced = {
            "self",
            "base_model_name",
            "model_name",
            "is_vlm",
            "is_processor",
            "is_gpt_oss",
            "_gguf_prewarm_ok",
            "quantization_method",
            "first_conversion",
            "imatrix_file",
        }
        for name in introduced - accepted:
            assert name in deleted, f"{name} would be passed to unsloth_generic_save"


# The unsloth_zoo.disk_utils signatures, transcribed. The fixtures above accept
# `**kwargs`, which is convenient and hides the one failure that matters: an
# argument the real function does not take raises TypeError, and every caller
# here swallows exceptions, so the guard silently turns itself off.
# `test_reference_signatures_match_the_installed_zoo` keeps these honest.
def _zoo_estimate_gguf_export_bytes(
    model = None,
    quantization_methods = (),
    first_conversion = "f16",
    needs_merge = True,
    n_parameters = None,
    base_cache_copy = False,
):
    return 10 * GB


def _zoo_model_16bit_bytes(model):
    return 10 * GB


def _zoo_kaggle_tmp_redirect(
    save_directory,
    need_bytes = 0,
    what = "export",
    subdirectory = "unsloth_saves",
):
    return "/tmp/unsloth_saves/model", "moved"


@pytest.fixture
def zoo_api(monkeypatch):
    """Bind the preflights against the real zoo signatures, not `**kwargs`."""
    monkeypatch.setattr(S, "estimate_gguf_export_bytes", _zoo_estimate_gguf_export_bytes)
    monkeypatch.setattr(S, "model_16bit_bytes", _zoo_model_16bit_bytes)
    monkeypatch.setattr(S, "kaggle_tmp_redirect", _zoo_kaggle_tmp_redirect)
    monkeypatch.setattr(S, "free_bytes", lambda path: 1000 * GB)
    monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
    monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)


class TestCallsBindToTheZooApi:
    """A guard that cannot call its own sizing function is not a guard.

    `_preflight_merge_disk` passed `keep_intermediate_gguf`, which
    `unsloth_zoo.disk_utils.estimate_gguf_export_bytes` does not accept, so
    every call raised TypeError into the surrounding `except Exception` and
    no `save_pretrained_merged` was ever redirected on Kaggle.
    """

    def test_merge_preflight_reaches_the_redirect(self, zoo_api):
        assert (
            S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit")
            == "/tmp/unsloth_saves/model"
        )

    def test_gguf_preflight_reaches_the_redirect(self, zoo_api):
        directory, _ = S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert directory == "/tmp/unsloth_saves/model"

    def test_reference_signatures_match_the_installed_zoo(self):
        import inspect
        disk_utils = pytest.importorskip("unsloth_zoo.disk_utils")
        for reference, name in (
            (_zoo_estimate_gguf_export_bytes, "estimate_gguf_export_bytes"),
            (_zoo_model_16bit_bytes, "model_16bit_bytes"),
            (_zoo_kaggle_tmp_redirect, "kaggle_tmp_redirect"),
        ):
            assert inspect.signature(reference) == inspect.signature(
                getattr(disk_utils, name)
            ), f"{name} drifted; the stubs above no longer prove anything"


class TestMergeSizing:
    """A merge writes the model once, and a compressed export writes it twice."""

    @pytest.fixture
    def sized(self, monkeypatch):
        asked = []
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        return asked

    def test_a_plain_merge_is_two_bytes_per_parameter(self, sized):
        S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit")
        # Not the GGUF estimate, which would add an intermediate conversion
        # this export never writes.
        assert sized == [_with_merge_headroom(10 * GB)]

    @pytest.mark.parametrize("save_method", ["merged 16bit", "MERGED_16BIT", " merged-16bit "])
    def test_supported_spellings_are_measured_too(self, sized, save_method):
        """`unsloth_save_model` normalizes spaces, so these are the same export."""
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert sized == [_with_merge_headroom(10 * GB)]

    @pytest.mark.parametrize(
        "save_method,expected_gb",
        [
            ("fp8", 15),  # 16-bit merge + an 8-bit sibling
            ("mxfp8", 15),
            ("int8", 15),
            ("mxfp4", 12.5),  # 16-bit merge + a 4-bit sibling
            ("nvfp4", 12.5),
            ("w4a16", 12.5),
        ],
    )
    def test_every_compressed_export_sizes_its_sibling(self, sized, save_method, expected_gb):
        """`_unsloth_save_compressed_tensors` keeps the merge AND the sibling."""
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert sized == [pytest.approx(_with_merge_headroom(expected_gb * GB))]

    def test_an_unsupported_near_miss_is_left_to_its_own_error(self, sized):
        """`_normalize_compressed_method` raises on these; the message is downstream."""
        assert S._preflight_merge_disk(_FakeModel(), "model", "fp4_banana") == "model"
        assert sized == []

    @pytest.mark.parametrize(
        "save_method", ["torchao_fp8", "torchao_int8", "portable_fp8", "portable-int8"]
    )
    def test_a_torchao_export_is_sized_by_its_sibling_alone(self, sized, save_method):
        """The torchao path merges into a temp dir, not into `save_directory`.

        So `save_directory` holds the 8-bit sibling only, and pricing the
        16-bit merge there as well would move an export that fits.
        """
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert sized == [_with_merge_headroom(5 * GB)]

    def test_the_embeddings_are_not_priced_as_quantized(self, sized):
        """Weight-only schemes quantize `Linear` only.

        The input embeddings and an untied lm_head stay 16-bit in the sibling,
        so a model that is a quarter embeddings costs more than half the merge.
        """
        model = _ModelWithEmbeddings(input_numel = 1024**3, output_numel = 1024**3 // 2)
        # 10GB merge, 3GB of it embeddings -> 7GB at 8 bits + 3GB copied.
        S._preflight_merge_disk(model, "model", "fp8")
        assert sized == [_with_merge_headroom(10 * GB + 3 * GB + int(3.5 * GB))]

    def test_tied_embeddings_are_counted_once(self, sized):
        model = _ModelWithEmbeddings(input_numel = 1024**3, tied = True)
        S._preflight_merge_disk(model, "model", "fp8")
        assert sized == [_with_merge_headroom(10 * GB + 2 * GB + 4 * GB)]

    def test_a_model_that_does_not_answer_is_sized_as_before(self, sized):
        """The old whole-model arithmetic, so this can only ever ask for more."""
        S._preflight_merge_disk(_FakeModel(), "model", "fp8")
        assert sized == [_with_merge_headroom(15 * GB)]

    def test_the_torchao_merge_really_is_staged_elsewhere(self):
        """The sizing above is only right while this stays true."""
        import inspect

        source = inspect.getsource(S._unsloth_save_torchao)
        assert "mkdtemp" in source
        assert "save_directory = staging" in source
        assert 'out_dir = base + "-" + suffix' in source


class TestTorchaoStagingSharesTheRedirectDestination:
    """The torchao staging merge is on /tmp, and so is the redirect target.

    `_unsloth_save_torchao` merges into `tempfile.mkdtemp()` and deletes it
    only after quantization, so on a Kaggle kernel both artefacts are on /tmp
    at once. Sending the sibling to a /tmp that cannot hold the staging merge
    as well turns an export that fit in /kaggle/working into a disk-full
    failure, so a destination that is too small for both is not used.

    The staging bytes are deliberately NOT added to `need_bytes`: nothing
    stages in the working directory, and charging it there would relocate
    exports that fit into /tmp, which is not kept as notebook output.
    """

    @pytest.fixture
    def redirected(self, monkeypatch, tmp_path):
        """A redirect that always fires, onto a real directory the test owns."""
        target = str(tmp_path)
        free = {"bytes": 0}
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (target, "moved"),
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: free["bytes"])
        monkeypatch.setattr(S, "_same_filesystem", lambda left, right: True)
        return target, free

    # 10GB merge -> a 5GB sibling, asked for with the 5% merge headroom.
    _SIBLING = _with_merge_headroom(5 * GB)

    def test_room_for_the_sibling_alone_is_not_enough(self, redirected):
        target, free = redirected
        free["bytes"] = self._SIBLING + 10 * GB - 1
        assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == "model"

    def test_room_for_both_takes_the_redirect(self, redirected):
        target, free = redirected
        free["bytes"] = self._SIBLING + 10 * GB
        assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == target

    def test_a_separate_staging_filesystem_is_not_charged(self, redirected, monkeypatch):
        """Off Kaggle `tempfile` can be its own mount; then only the sibling lands here."""
        target, free = redirected
        free["bytes"] = self._SIBLING
        monkeypatch.setattr(S, "_same_filesystem", lambda left, right: False)
        assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == target

    @pytest.mark.parametrize("save_method", ["merged_16bit", "fp8", "mxfp4"])
    def test_an_export_that_stages_nothing_is_never_cancelled(self, redirected, save_method):
        """Only torchao stages a second copy; the rest write into `save_directory`."""
        target, free = redirected
        free["bytes"] = 0
        assert S._preflight_merge_disk(_FakeModel(), "model", save_method) == target

    def test_an_unmeasurable_destination_takes_the_redirect(self, redirected):
        target, free = redirected
        free["bytes"] = None
        assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == target

    def test_the_staging_is_not_charged_to_the_working_directory(self, monkeypatch):
        """What `kaggle_tmp_redirect` is asked for stays the sibling alone."""
        asked = []
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        assert asked == [self._SIBLING]


class TestMergeHeadroomMatchesTheZooGuard:
    """A working directory that is "just big enough" is not big enough.

    `merge_and_overwrite_lora` compares the save against `int(free * 0.95)`,
    so a 30GB merge with 31GB free was left in /kaggle/working by the redirect
    and then refused outright by the merge itself.
    """

    @pytest.fixture
    def kaggle(self, monkeypatch):
        """A `kaggle_tmp_redirect` that answers like the real one."""
        free_working = 31 * GB

        def fake_redirect(
            save_directory,
            need_bytes = 0,
            what = "export",
        ):
            if need_bytes <= 0 or free_working >= need_bytes:
                return save_directory, None
            return "/tmp/unsloth_saves/model", "moved"

        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 30 * GB)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", fake_redirect)

    def test_a_merge_that_only_just_fits_is_still_redirected(self, kaggle):
        assert S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit") == (
            "/tmp/unsloth_saves/model"
        )

    def test_the_ask_clears_the_guard_that_comes_next(self, monkeypatch):
        asked = []
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 30 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        S._preflight_merge_disk(_FakeModel(), "model", "merged_16bit")
        # Free space that satisfies this preflight also satisfies the 5% the
        # merge reserves; 31GB satisfies neither.
        assert int(asked[0] * 0.95) >= 30 * GB
        assert int(31 * GB * 0.95) < 30 * GB


class TestSixteenBitCheckpointDetection:
    """`needs_merge` is "does a 16-bit checkpoint get written", not "is it PEFT"."""

    class _NonPeft:
        class config:
            _name_or_path = "unsloth/Qwen3-32B"

    class _NonPeftFromDisk:
        def __init__(self, directory):
            self.config = type("cfg", (), {"_name_or_path": directory})()

    def test_a_hub_id_still_writes_a_checkpoint(self):
        """The non-PEFT fallback `save_pretrained`s the whole model first."""
        assert S._gguf_writes_16bit_checkpoint(self._NonPeft()) is True

    def test_an_existing_local_checkpoint_is_reused(self, tmp_path):
        assert S._gguf_writes_16bit_checkpoint(self._NonPeftFromDisk(str(tmp_path))) is False

    def test_no_config_at_all_is_counted(self):
        assert S._gguf_writes_16bit_checkpoint(_FakeModel()) is True


class TestFallbackCheckpointDtype:
    """The non-PEFT fallback `save_pretrained`s the model at its own dtype.

    The estimator budgets two bytes per parameter for that checkpoint, so an
    fp32 model (`dtype = torch.float32` is a supported load) writes twice what
    was measured and can fill a disk this called big enough.
    """

    @pytest.fixture
    def sized_from_parameters(self, monkeypatch):
        import torch
        monkeypatch.setattr(
            S,
            "model_16bit_bytes",
            lambda model: sum(p.numel() for p in model.parameters()) * 2,
        )
        return torch

    def test_float32_parameters_cost_the_difference(self, sized_from_parameters):
        torch = sized_from_parameters
        model = torch.nn.Linear(8, 8, dtype = torch.float32)
        n_parameters = sum(p.numel() for p in model.parameters())
        assert S._fallback_checkpoint_extra_bytes(model) == n_parameters * 2

    def test_a_sixteen_bit_model_adds_nothing(self, sized_from_parameters):
        torch = sized_from_parameters
        model = torch.nn.Linear(8, 8, dtype = torch.bfloat16)
        assert S._fallback_checkpoint_extra_bytes(model) == 0

    def test_a_reused_checkpoint_on_disk_adds_nothing(self, sized_from_parameters, tmp_path):
        """Nothing is written, whatever dtype the model is in memory."""
        torch = sized_from_parameters
        model = torch.nn.Linear(8, 8, dtype = torch.float32)
        model.config = type("cfg", (), {"_name_or_path": str(tmp_path)})()
        assert S._fallback_checkpoint_extra_bytes(model) == 0

    def test_an_unmeasurable_model_adds_nothing(self):
        assert S._fallback_checkpoint_extra_bytes(_FakeModel()) == 0

    def test_the_gguf_preflight_asks_for_it(self, sized_from_parameters, monkeypatch):
        torch = sized_from_parameters
        asked = []
        model = torch.nn.Linear(8, 8, dtype = torch.float32)
        n_parameters = sum(p.numel() for p in model.parameters())
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", lambda **kwargs: 100 * GB)
        monkeypatch.setattr(S, "free_bytes", lambda path: 1000 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        S._preflight_gguf_disk(model, "model", "q4_k_m", needs_merge = True)
        assert asked == [100 * GB + n_parameters * 2]


class TestDisabledImatrixIsNotSizedAsAnImatrix:
    @pytest.mark.parametrize("value", [None, False])
    def test_agrees_with_the_resolver(self, value):
        assert S._imatrix_is_enabled(value) is False
        assert S._resolve_imatrix_file(_FakeModel(), value, None, "unused") is None

    def test_a_real_path_is_enabled(self, tmp_path):
        path = tmp_path / "imatrix.gguf"
        path.write_bytes(b"x")
        assert S._imatrix_is_enabled(str(path)) is True
        assert S._resolve_imatrix_file(_FakeModel(), str(path), None, str(tmp_path)) == str(path)

    def test_a_disabled_imatrix_keeps_the_single_pass_conversion(self):
        """The point of the flag: q8_0 alone converts straight to q8_0."""
        assert S._choose_first_conversion(["q8_0"], "f16", has_imatrix = False) == "q8_0"
        assert S._choose_first_conversion(["q8_0"], "f16", has_imatrix = True) == "f16"


class TestKaggleNeverPricesACacheCopy:
    """`_prewarm_base_model_hub_cache` returns before it runs on Kaggle and Colab.

    Pricing a cache that cannot exist sends an export that fits in
    /kaggle/working to /tmp, which is not kept as notebook output.
    """

    @pytest.fixture
    def asked(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            S,
            "estimate_gguf_export_bytes",
            lambda **kwargs: seen.append(kwargs)
            or (44 * GB if kwargs.get("base_cache_copy") else 30 * GB),
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: 1000 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                seen.append({"need_bytes": need_bytes}) or (save_directory, None)
            ),
        )
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return seen

    @pytest.mark.parametrize("environment", ["IS_KAGGLE_ENVIRONMENT", "IS_COLAB_ENVIRONMENT"])
    def test_the_redirect_is_priced_without_the_cache(self, asked, monkeypatch, environment):
        monkeypatch.setattr(S, environment, True)
        S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", needs_merge = True)
        assert not any(call.get("base_cache_copy") for call in asked)
        assert asked[-1]["need_bytes"] == 30 * GB

    def test_an_ordinary_machine_still_prices_it(self, asked, monkeypatch):
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", needs_merge = True)
        assert any(call.get("base_cache_copy") for call in asked)
        assert asked[-1]["need_bytes"] == 44 * GB
