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

# http://www.apache.org/licenses/LICENSE-2.0
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

import contextlib
import math
import os

import pytest

from unsloth import save as S

GB = 1024**3


def _with_merge_headroom(n_bytes):
    """What `merge_and_overwrite_lora` needs free to write `n_bytes` of merge.

    unsloth_zoo's merge guard compares against `int(free * 0.95)`, so the
    preflight has to ask for the same effective figure. The 0.95 is written
    out rather than read from `S`, so dropping the headroom fails here.
    """
    return math.ceil(n_bytes / 0.95)


def _merge_preflight_ask(total_bytes, merge_bytes):
    """What the merge preflight asks for, given what lands where.

    The reserve belongs on the merge alone, because that is the only artefact
    `merge_and_overwrite_lora` writes and the only one its guard measures. A
    quantized sibling beside it, a torchao sibling with the merge staged in a
    temp directory, and a full-model `"lora"` save written straight through
    `save_pretrained` are all charged at face value. Reserving around the whole
    estimate instead moves an export that fits, and on Kaggle "moves" means
    into a /tmp the kernel does not keep as notebook output.
    """
    return max(math.ceil(total_bytes), _with_merge_headroom(merge_bytes))


class _FakeModel:
    """Not a PeftModel; the preflight is called with an explicit needs_merge."""


class _ModelWithLayers:
    """The `.model.layers` layout `unsloth_save_model` rebuilds a dict for.

    Anything else takes its generic fallback, which hands the caller's
    dictionary to `save_pretrained` untouched.
    """

    class _Inner:
        layers = ()

    def __init__(self):
        self.model = self._Inner()


class _FakeAdapterModel:
    """Stands in for a PeftModel: monkeypatch `S.PeftModel` onto this class.

    Building a real one needs a base model and an adapter config, and the
    preflight only ever asks `isinstance`.
    """


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
        # directory the intermediate conversion is written to. What must never
        # The `_gguf` sibling is measured as well, and so is the working directory the intermediate conversion is
        assert probed[0] == "/tmp/unsloth_saves/model"
        assert "model" not in probed
        assert set(probed) <= {
            "/tmp/unsloth_saves/model",
            "/tmp/unsloth_saves/model_gguf",
            os.getcwd(),
        }

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

    def test_merge_preflight_skips_an_export_that_writes_no_checkpoint(self, monkeypatch):
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/x", "moved"))
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        assert S._preflight_merge_disk(_FakeModel(), "model", "merged_4bit") == "model"

    def test_merge_preflight_skips_adapters(self, monkeypatch):
        """`lora` on a real PeftModel writes adapters, which are megabytes."""
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/x", "moved"))
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        assert S._preflight_merge_disk(_FakeAdapterModel(), "model", "lora") == "model"

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
        # Every local that exists at the snapshot point and is not a parameter of unsloth_generic_save has to be
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


# The unsloth_zoo.disk_utils signatures, transcribed.
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
        assert sized == [_merge_preflight_ask(10 * GB, 0)]

    @pytest.mark.parametrize("save_method", ["merged 16bit", "MERGED_16BIT", " merged-16bit "])
    def test_supported_spellings_are_measured_too(self, sized, save_method):
        """`unsloth_save_model` normalizes spaces, so these are the same export."""
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        # this export never writes. No headroom either: `_FakeModel` has no
        # Not the GGUF estimate, which would add an intermediate conversion this export never writes.
        assert sized == [_merge_preflight_ask(10 * GB, 0)]

    @pytest.mark.parametrize(
        "save_method,expected_gb",
        [
            ("fp8", 15),
            ("mxfp8", 15),
            ("int8", 15),
            ("mxfp4", 12.5),
            ("nvfp4", 12.5),
            ("w4a16", 12.5),
        ],
    )
    def test_every_compressed_export_sizes_its_sibling(self, sized, save_method, expected_gb):
        """`_unsloth_save_compressed_tensors` keeps the merge AND the sibling."""
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert sized == [pytest.approx(_merge_preflight_ask(expected_gb * GB, 10 * GB))]

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
        16-bit merge there as well would move an export that fits. No merge
        guard runs against this filesystem either, so no reserve is charged.
        """
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert sized == [_merge_preflight_ask(5 * GB, 0)]

    def test_the_embeddings_are_not_priced_as_quantized(self, sized):
        """Weight-only schemes quantize `Linear` only.

        The input embeddings and an untied lm_head stay 16-bit in the sibling,
        so a model that is a quarter embeddings costs more than half the merge.
        """
        model = _ModelWithEmbeddings(input_numel = 1024**3, output_numel = 1024**3 // 2)
        S._preflight_merge_disk(model, "model", "fp8")
        assert sized == [_merge_preflight_ask(10 * GB + 3 * GB + int(3.5 * GB), 10 * GB)]

    def test_tied_embeddings_are_counted_once(self, sized):
        model = _ModelWithEmbeddings(input_numel = 1024**3, tied = True)
        # 10GB merge, 3GB of it embeddings -> 7GB at 8 bits + 3GB copied.
        S._preflight_merge_disk(model, "model", "fp8")
        assert sized == [_merge_preflight_ask(10 * GB + 2 * GB + 4 * GB, 10 * GB)]

    def test_a_model_that_does_not_answer_is_sized_as_before(self, sized):
        """The old whole-model arithmetic, so this can only ever ask for more."""
        S._preflight_merge_disk(_FakeModel(), "model", "fp8")
        assert sized == [_merge_preflight_ask(15 * GB, 10 * GB)]

    def test_the_torchao_merge_really_is_staged_elsewhere(self):
        """The sizing above is only right while this stays true."""
        import inspect

        source = inspect.getsource(S._unsloth_save_torchao)
        assert "mkdtemp" in source
        assert "save_directory = staging" in source
        assert 'out_dir = base + "-" + suffix' in source


class _FakeParameter:
    """A parameter the sizing can measure without torch."""

    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


class _FakeModule:
    """A module tree answering the two calls the sizing makes.

    `parameters()` recurses, exactly as `torch.nn.Module.parameters()` does,
    so a parent that matches an ignore pattern reports its children's weights
    too and the deduplication has something to do.
    """

    def __init__(
        self,
        parameters = (),
        children = None,
    ):
        self._parameters = list(parameters)
        self._children = dict(children or {})

    def parameters(self):
        for parameter in self._parameters:
            yield parameter
        for child in self._children.values():
            yield from child.parameters()

    def named_modules(self, prefix = ""):
        yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)


class _ShapedModel(_FakeModule):
    """A named module tree plus the embedding getters and a config."""

    def __init__(
        self,
        children,
        config,
        input_embeddings = None,
        output_embeddings = None,
    ):
        super().__init__(children = children)
        self.config = config
        self._input = input_embeddings
        self._output = output_embeddings

    def get_input_embeddings(self):
        return self._input

    def get_output_embeddings(self):
        return self._output


class _Config:
    def __init__(self, **kwargs):
        self.model_type = ""
        for key, value in kwargs.items():
            setattr(self, key, value)


def _embedding(numel):
    """A module whose single weight is also what the getters return."""
    parameter = _FakeParameter(numel)
    module = _FakeModule(parameters = [parameter])
    module.weight = parameter
    return module


def _sibling_bytes(merge_bytes, unquantized_bytes, weight_bits):
    """The arithmetic `_quantized_sibling_bytes` does, written out."""
    return int((merge_bytes - unquantized_bytes) * weight_bits / 16) + unquantized_bytes


class TestTheRecipesIgnoredModulesStay16Bit:
    """Everything `_compressed_quantize`'s recipe refuses to quantize is copied at 16 bits.

    The recipe ignores `lm_head`, every module under a `linear_attn` or a
    `visual`, anything named `*mtp*`, and on an MoE the router gates. Only the
    embeddings were priced at 16 bits here, so a VLM's vision tower or a
    hybrid's linear attention was charged 4 or 8 bits for bytes the export
    writes at 16 - an under-count, which is the direction that loses the
    Kaggle redirect an export needed.
    """

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

    def _vlm(self):
        """A 10GB VLM: 2GB of input embeddings and a 1GB vision tower."""
        embeddings = _embedding(1024**3)
        visual = _FakeModule(
            children = {
                "blocks": _FakeModule(
                    children = {
                        "0": _FakeModule(
                            children = {
                                "attn": _FakeModule(parameters = [_FakeParameter(1024**3 // 2)]),
                            }
                        ),
                    }
                ),
            }
        )
        return _ShapedModel(
            children = {
                "model": _FakeModule(children = {"embed_tokens": embeddings, "visual": visual}),
            },
            config = _Config(model_type = "qwen3_vl"),
            input_embeddings = embeddings,
        )

    def test_a_vision_tower_is_not_priced_as_quantized(self, sized):
        S._preflight_merge_disk(self._vlm(), "model", "fp8")
        # 2GB embeddings + 1GB vision tower stay 16-bit;
        # `gate_proj` is 2GB and quantizes.
        expected = 10 * GB + _sibling_bytes(10 * GB, 3 * GB, 8)
        assert sized == [_merge_preflight_ask(expected, 10 * GB)]
        # And strictly more than the embeddings-only figure this replaces, which is the whole point:
        assert sized[0] > _merge_preflight_ask(
            10 * GB + _sibling_bytes(10 * GB, 2 * GB, 8), 10 * GB
        )

    def test_the_vision_tower_is_counted_once(self):
        """`re:.*\\.visual\\..*` matches the tower's children, not just the tower."""
        model = self._vlm()
        patterns = S._compressed_ignore_patterns(model)
        assert S._unquantized_parameter_bytes(model, patterns) == 3 * GB

    def test_a_torchao_export_is_charged_none_of_them(self, sized):
        """torchao quantizes with no ignore list, so charging these over-counts."""
        S._preflight_merge_disk(self._vlm(), "model", "torchao_fp8")
        assert sized == [_merge_preflight_ask(_sibling_bytes(10 * GB, 2 * GB, 8), 0)]

    def test_an_lm_head_module_is_not_counted_twice(self):
        """`lm_head` is both `get_output_embeddings()` and an ignored name."""
        embeddings = _embedding(1024**3)
        head = _embedding(1024**3)
        model = _ShapedModel(
            children = {"model": _FakeModule(children = {"embed_tokens": embeddings}), "lm_head": head},
            config = _Config(model_type = "llama"),
            input_embeddings = embeddings,
            output_embeddings = head,
        )
        patterns = S._compressed_ignore_patterns(model)
        assert "lm_head" in patterns
        assert S._unquantized_parameter_bytes(model, patterns) == 4 * GB

    def test_the_moe_gates_are_counted_and_gate_proj_is_not(self):
        """`re:.*\\.gate$` is anchored: it must not swallow `gate_proj`."""
        embeddings = _embedding(1024**3)
        layer = _FakeModule(
            children = {
                "mlp": _FakeModule(
                    children = {
                        "gate": _FakeModule(parameters = [_FakeParameter(1024**3 // 2)]),
                        "gate_proj": _FakeModule(parameters = [_FakeParameter(1024**3)]),
                    }
                ),
            }
        )
        model = _ShapedModel(
            children = {
                "model": _FakeModule(
                    children = {
                        "embed_tokens": embeddings,
                        "layers": _FakeModule(children = {"0": layer}),
                    },
                ),
            },
            config = _Config(model_type = "qwen3_moe", num_experts = 128),
            input_embeddings = embeddings,
        )
        patterns = S._compressed_ignore_patterns(model)
        assert "re:.*\\.gate$" in patterns
        # 2GB embeddings + 1GB router gate.
        assert S._unquantized_parameter_bytes(model, patterns) == 3 * GB

    def test_a_dense_model_gets_no_gate_patterns(self):
        model = _ShapedModel(children = {}, config = _Config(model_type = "llama"))
        assert "re:.*\\.gate$" not in S._compressed_ignore_patterns(model)

    def test_a_model_whose_modules_cannot_be_walked_keeps_the_old_figure(self):
        """Degrade to today's behaviour rather than raise."""
        model = self._vlm()

        def explode():
            raise RuntimeError("no modules here")

        model.named_modules = explode
        assert S._unquantized_parameter_bytes(model, S._compressed_ignore_patterns(model)) == 2 * GB

    def test_a_missing_recipe_symbol_leaves_the_estimate_alone(self, monkeypatch):
        """A renamed or deleted `compressed_ignore_patterns` must not raise."""
        import unsloth._compressed_quantize as Q

        monkeypatch.delattr(Q, "compressed_ignore_patterns")
        assert S._compressed_ignore_patterns(self._vlm()) == []

    def test_an_unparseable_pattern_matches_nothing(self):
        module = _FakeModule(parameters = [_FakeParameter(1024**3)])
        assert not S._matches_ignore_pattern("model.visual.attn", module, ["re:*["])

    def test_the_matcher_mirrors_compressed_tensors(self):
        """`re:` is `re.match` (start-anchored, not full); plain is an exact name."""
        module = _FakeModule()
        assert S._matches_ignore_pattern("visual.blocks.0", module, ["re:visual.*"])
        # Start-anchored: a mid-string match is not one.
        assert not S._matches_ignore_pattern("model.visual.blocks", module, ["re:visual.*"])
        assert S._matches_ignore_pattern("model.visual.blocks", module, ["re:.*\\.visual\\..*"])
        assert S._matches_ignore_pattern("model.mtp.layers.0", module, ["re:.*mtp.*"])
        # Plain entries are exact, never substrings.
        assert S._matches_ignore_pattern("lm_head", module, ["lm_head"])
        assert not S._matches_ignore_pattern("model.lm_head", module, ["lm_head"])
        # Plain entries also match a parent class name, as `_match_class` does.
        assert S._matches_ignore_pattern("anything", module, ["_FakeModule"])

    def test_the_recipe_is_read_from_the_runner_not_restated(self):
        """The drift this fixes is two copies of one list; keep there being one."""
        import inspect

        source = inspect.getsource(S._compressed_ignore_patterns)
        assert "from unsloth._compressed_quantize import compressed_ignore_patterns" in source
        import unsloth._compressed_quantize as Q

        main_source = inspect.getsource(Q.main)
        assert "compressed_ignore_patterns(config)" in main_source
        assert "re:.*mtp.*" not in main_source, "main() must not rebuild the list"


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

    # 10GB merge -> a 5GB sibling.
    _SIBLING = 5 * GB

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

    def test_a_cancelled_redirect_says_the_filesystem_is_short(self, monkeypatch, capsys):
        """The one outcome nothing downstream measures.

        The redirect fired because /kaggle/working could not hold the 5GB
        sibling, and it was cancelled because /tmp cannot hold that sibling
        and the 10GB staging merge together. The export is handed back a
        filesystem with 4GB free, the torchao merge guard only ever measures
        the staging disk, and the sibling is written at the end of a long
        quantization, so this has to be said up front.
        """
        import tempfile

        working, scratch = 4 * GB, 12 * GB
        on_tmp = lambda path: str(path).startswith("/tmp") or str(path) == tempfile.gettempdir()
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                ("/tmp/unsloth_saves/model", "moved")
                if working < need_bytes <= scratch
                else (save_directory, None)
            ),
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: scratch if on_tmp(path) else working)
        monkeypatch.setattr(
            S, "_same_filesystem", lambda left, right: on_tmp(left) == on_tmp(right)
        )
        assert (
            S._preflight_merge_disk(_FakeModel(), "kaggle/working/model", "torchao_fp8")
            == "kaggle/working/model"
        )
        printed = capsys.readouterr().out
        assert "4.0GB free" in printed
        assert "5.0GB" in printed

    def test_a_cancelled_redirect_onto_a_roomy_filesystem_is_silent(self, monkeypatch, capsys):
        """`UNSLOTH_KAGGLE_USE_TMP=1` moves without measuring, so silence is right."""
        import tempfile

        on_tmp = lambda path: str(path).startswith("/tmp") or str(path) == tempfile.gettempdir()
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                "/tmp/unsloth_saves/model",
                "moved",
            ),
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: 12 * GB if on_tmp(path) else 500 * GB)
        monkeypatch.setattr(
            S, "_same_filesystem", lambda left, right: on_tmp(left) == on_tmp(right)
        )
        assert (
            S._preflight_merge_disk(_FakeModel(), "kaggle/working/model", "torchao_fp8")
            == "kaggle/working/model"
        )
        assert capsys.readouterr().out == ""

    def test_a_real_stat_of_the_destination_cancels_the_redirect(self, monkeypatch):
        """The same cancellation, with `_same_filesystem` left unstubbed.

        Cancelling is the one outcome the helper cannot reach by accident:
        every failure inside it, `os.stat` included, returns True and takes
        the redirect. So a real destination that is really rejected is proof
        that both stats resolved and compared equal.
        """
        import shutil
        import tempfile

        # Under the tempfile default, so it shares that mount by construction.
        destination = tempfile.mkdtemp(prefix = "unsloth-staging-test-")
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (destination, "moved"),
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: self._SIBLING + 10 * GB - 1)
        try:
            assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == "model"
        finally:
            shutil.rmtree(destination, ignore_errors = True)

    def test_the_redirect_creates_its_target_before_returning_it(self, monkeypatch, tmp_path):
        """Which is why stat-ing the destination is safe on a first export.

        `kaggle_tmp_redirect` returns a message only after `os.makedirs`
        succeeded, and the `unsloth.disk_utils` fallback never returns one, so
        the helper is unreachable with a destination that does not exist.
        """
        from unsloth.disk_utils import HAS_ZOO_DISK_UTILS

        if not HAS_ZOO_DISK_UTILS:
            assert S.kaggle_tmp_redirect("model", need_bytes = GB)[1] is None
            return

        import unsloth_zoo.disk_utils as zoo

        working = tmp_path / "kaggle" / "working"
        scratch = tmp_path / "tmp"
        working.mkdir(parents = True)
        scratch.mkdir()
        monkeypatch.setattr(zoo, "KAGGLE_WORKING", str(working))
        monkeypatch.setattr(zoo, "KAGGLE_TMP", str(scratch))
        monkeypatch.setenv("UNSLOTH_IS_KAGGLE", "1")
        monkeypatch.delenv("UNSLOTH_KAGGLE_USE_TMP", raising = False)
        monkeypatch.setattr(
            zoo,
            "free_bytes",
            lambda path: 1 if os.path.abspath(str(path)) == str(working) else 1000 * GB,
        )
        monkeypatch.chdir(working)
        target, message = zoo.kaggle_tmp_redirect(
            "model",
            need_bytes = GB,
            what = "16-bit merge",
        )
        assert message is not None, "the redirect never fired, so nothing was proven"
        assert os.path.isdir(target)


class TestASeparateStagingFilesystemIsStillMeasured:
    """The other half: `_destination_holds_torchao_staging` asks nothing there.

    When `tempfile` resolves onto its own mount that helper returns True and
    the staging filesystem is never measured at all, so a 4GB tmpfs is handed
    a 60GB merge and `_unsloth_save_torchao` dies inside `tempfile.mkdtemp`
    without saying which disk ran out.

    A warning, not a refusal. Cancelling the redirect cannot help: the staging
    merge goes to TMPDIR either way, so declining the move leaves the same
    failure and puts the output on the smaller disk too.
    """

    _STAGING = 10 * GB

    @pytest.fixture
    def separate(self, monkeypatch, tmp_path):
        """A TMPDIR on its own mount, with free space the test dictates."""
        import tempfile

        staging = tmp_path / "tmp"
        staging.mkdir()
        # `tempfile.gettempdir()` caches its answer on first use, so setting the variable alone would leave the
        monkeypatch.setattr(tempfile, "tempdir", str(staging))
        free = {"staging": 0, "other": 1000 * GB}
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: self._STAGING)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: (a[0], None))
        monkeypatch.setattr(S, "_same_filesystem", lambda left, right: False)
        # The working directory is full and the scratch overlay is not, which is the only shape that moves anything.
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: free["staging"] if str(path) == str(staging) else free["other"],
        )
        monkeypatch.setenv("TMPDIR", str(staging))
        return free

    def test_a_short_staging_filesystem_is_named(self, separate, capsys):
        separate["staging"] = _with_merge_headroom(self._STAGING) - 1
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        out = capsys.readouterr().out
        assert "TMPDIR" in out
        assert "10.5GB" in out

    def test_a_staging_filesystem_with_room_says_nothing(self, separate, capsys):
        separate["staging"] = _with_merge_headroom(self._STAGING)
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        assert capsys.readouterr().out == ""

    def test_the_warning_carries_the_merge_guards_own_reserve(self, separate, capsys):
        """Exactly the merge's size on TMPDIR is not enough to write it there.

        `merge_and_overwrite_lora` compares the staging save against
        `int(free * 0.95)` and raises "Failed saving - no disk space left"
        when it is short, and on a separate TMPDIR its own /tmp fallback is
        that same filesystem, so there is no recovery. Between `staging_bytes`
        and `staging_bytes / 0.95` the merge therefore dies and only this
        warning could have named the disk, so it has to use the same figure
        `_preflight_merge_disk` asks the redirect for.
        """
        separate["staging"] = self._STAGING
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        assert "TMPDIR" in capsys.readouterr().out

        separate["staging"] = self._STAGING - 1
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        assert "TMPDIR" in capsys.readouterr().out

    @pytest.mark.parametrize("save_method", ["merged_16bit", "fp8", "mxfp4"])
    def test_an_export_that_stages_nothing_is_silent(self, separate, capsys, save_method):
        """Only torchao writes a second full copy outside `save_directory`."""
        separate["staging"] = 0
        S._preflight_merge_disk(_FakeModel(), "model", save_method)
        assert capsys.readouterr().out == ""

    def test_a_shared_staging_filesystem_is_left_to_the_other_helper(
        self, separate, capsys, monkeypatch
    ):
        """On Kaggle both are /tmp, and that case is already charged for."""
        separate["staging"] = 0
        monkeypatch.setattr(S, "_same_filesystem", lambda left, right: True)
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")
        assert capsys.readouterr().out == ""

    def test_it_warns_rather_than_refusing(self, separate):
        """The preflight still returns the directory it was given."""
        separate["staging"] = 0
        assert S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8") == "model"

    def test_an_unmeasurable_staging_filesystem_says_nothing(self, separate, monkeypatch):
        monkeypatch.setattr(S, "free_bytes", lambda path: None)
        S._preflight_merge_disk(_FakeModel(), "model", "torchao_fp8")

    def test_the_staging_really_is_a_full_second_copy(self):
        """The warning is only worth printing while this stays true."""
        import inspect

        source = inspect.getsource(S._unsloth_save_torchao)
        assert 'tempfile.mkdtemp(prefix = "unsloth-torchao-")' in source
        assert "save_pretrained_merged" in source or "merge" in source.lower()


class TestTheStagingWarningSurvivesAFreshDestination:
    """The destination of a first export does not exist yet.

    `_preflight_merge_disk` runs before anything is written, so on a first
    export `save_directory` is a name and not a directory. Comparing devices
    by stat-ing it directly raised, the helper's broad handler swallowed the
    whole probe, and the undersized TMPDIR went unmentioned in exactly the
    case the warning exists for. The nearest existing ancestor is the
    filesystem the write lands on, and is what `free_bytes` already measures.

    `os.stat` is faked rather than `_same_filesystem` monkeypatched, because
    the bug is inside `_same_filesystem` and a stub for it cannot show it.
    Existence still comes from the real `os.stat`, so the missing directory
    is genuinely missing.
    """

    _STAGING = 10 * GB

    @pytest.fixture
    def fresh(self, monkeypatch, tmp_path):
        import tempfile

        staging = tmp_path / "tmp"
        staging.mkdir()
        monkeypatch.chdir(tmp_path)
        # `tempfile.gettempdir()` caches its answer, so the attribute and the variable both have to move.
        monkeypatch.setattr(tempfile, "tempdir", str(staging))
        monkeypatch.setenv("TMPDIR", str(staging))

        class _Device:
            def __init__(self, st_dev):
                self.st_dev = st_dev

        devices = {str(staging): 20, str(tmp_path): 10}
        real_stat = os.stat

        def fake_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            device = devices.get(str(path), None)
            return result if device is None else _Device(device)

        monkeypatch.setattr(os, "stat", fake_stat)
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: self._STAGING)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: (a[0], None))
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 0 if str(path) == str(staging) else 1000 * GB,
        )
        return tmp_path

    def test_an_absent_destination_still_gets_the_warning(self, fresh, capsys):
        destination = str(fresh / "model")
        assert not os.path.exists(destination)
        S._preflight_merge_disk(_FakeModel(), destination, "torchao_fp8")
        assert "TMPDIR" in capsys.readouterr().out

    def test_a_shared_filesystem_is_still_silent(self, fresh, capsys, monkeypatch):
        """Resolving the ancestor must not turn every fresh path into a warning."""
        monkeypatch.setattr(S, "_filesystem_id", lambda path: 10)
        S._preflight_merge_disk(_FakeModel(), str(fresh / "model"), "torchao_fp8")
        assert capsys.readouterr().out == ""

    def test_an_unidentifiable_path_cancels_the_probe(self, fresh, capsys, monkeypatch):
        """Unmeasurable stays "cannot tell", which is silence and not a guess."""
        monkeypatch.setattr(S, "_filesystem_id", lambda path: None)
        S._preflight_merge_disk(_FakeModel(), str(fresh / "model"), "torchao_fp8")
        assert capsys.readouterr().out == ""


class TestMergeHeadroomMatchesTheZooGuard:
    """A working directory that is "just big enough" is not big enough.

    `merge_and_overwrite_lora` compares the save against `int(free * 0.95)`,
    so a 30GB merge with 31GB free was left in /kaggle/working by the redirect
    and then refused outright by the merge itself.

    An adapter merged by `unsloth_generic_save`, because that is the only
    writer here that calls the guarded function at all.
    """

    @pytest.fixture(autouse = True)
    def adapter(self, monkeypatch):
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)

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
        assert S._preflight_merge_disk(
            _FakeAdapterModel(),
            "model",
            "merged_16bit",
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        ) == ("/tmp/unsloth_saves/model")

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
        S._preflight_merge_disk(
            _FakeAdapterModel(),
            "model",
            "merged_16bit",
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        )
        # Free space that satisfies this preflight also satisfies the 5% the merge reserves;
        # 31GB satisfies neither.
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


class TestFullModelSavedAsLora:
    """`lora` on a model with no adapter writes the WHOLE model.

    `unsloth_generic_save` and `unsloth_save_model` both fall back to
    `save_pretrained` there, so the checkpoint is the size of the model and
    fills /kaggle/working exactly like a merge. Skipping the preflight on the
    method name alone let that through.
    """

    @pytest.fixture
    def sized(self, monkeypatch):
        asked = []
        # Deliberately not the size of the checkpoint:
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        return asked

    @staticmethod
    def _float32_model():
        import torch
        return torch.nn.Linear(8, 8, dtype = torch.float32)

    def test_a_full_model_lora_save_is_sized(self, sized):
        model = self._float32_model()
        expected = sum(p.numel() * p.element_size() for p in model.parameters())
        S._preflight_merge_disk(model, "model", "lora")
        assert sized == [_merge_preflight_ask(expected, 0)]

    def test_it_is_sized_at_the_tensors_own_dtype(self, sized):
        """Four bytes per parameter for fp32, not the two a merge writes."""
        model = self._float32_model()
        n_parameters = sum(p.numel() for p in model.parameters())
        assert S._full_model_checkpoint_bytes(model) == n_parameters * 4
        S._preflight_merge_disk(model, "model", "lora")
        assert sized == [_merge_preflight_ask(n_parameters * 4, 0)]

    def test_a_sixteen_bit_model_is_sized_at_two_bytes(self):
        import torch

        model = torch.nn.Linear(8, 8, dtype = torch.bfloat16)
        n_parameters = sum(p.numel() for p in model.parameters())
        assert S._full_model_checkpoint_bytes(model) == n_parameters * 2

    def test_an_adapter_save_is_still_skipped(self, sized, monkeypatch):
        """A real PeftModel writes adapters only; moving those buys nothing."""
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        assert S._preflight_merge_disk(_FakeAdapterModel(), "model", "lora") == "model"
        assert sized == []

    @pytest.mark.parametrize("save_method", ["LoRA", " lora "])
    def test_the_supported_spellings_are_measured_too(self, sized, save_method):
        S._preflight_merge_disk(self._float32_model(), "model", save_method)
        assert len(sized) == 1

    def test_an_unmeasurable_model_is_left_alone(self, sized):
        assert S._preflight_merge_disk(_FakeModel(), "model", "lora") == "model"
        assert sized == []

    def test_the_redirect_is_taken(self, monkeypatch, capsys):
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/unsloth_saves/model", "moved")
        )
        assert S._preflight_merge_disk(self._float32_model(), "model", "lora") == (
            "/tmp/unsloth_saves/model"
        )
        assert "moved" in capsys.readouterr().out

    def test_the_message_does_not_call_it_a_merge(self, monkeypatch):
        described = []
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                described.append(what) or (save_directory, None)
            ),
        )
        S._preflight_merge_disk(self._float32_model(), "model", "lora")
        assert described == ["model checkpoint"]

    def test_the_fallback_really_writes_the_whole_model(self):
        """The sizing above is only right while this stays true."""
        import inspect

        source = inspect.getsource(S.unsloth_generic_save)
        assert "_is_peft = isinstance(model, PeftModel)" in source
        assert "model.save_pretrained(save_directory, **_save_kwargs)" in source

    def test_a_caller_state_dict_is_what_gets_measured(self, sized):
        """`save_pretrained` writes the dict it was handed, not the model.

        Only `"16bit" in save_method` casts that dict, so a `"lora"` save can
        forward an fp32 one over fp16 resident parameters and the export is
        twice what sizing the model says. Undercounting is not a crash, it is
        a missed redirect, which is how a 20GB Kaggle working directory fills.
        """
        import torch

        model = torch.nn.Linear(8, 8, dtype = torch.float16)
        state_dict = {name: tensor.to(torch.float32) for name, tensor in model.state_dict().items()}
        n_parameters = sum(p.numel() for p in model.parameters())
        assert S._full_model_checkpoint_bytes(model, state_dict) == n_parameters * 4
        S._preflight_merge_disk(model, "model", "lora", state_dict = state_dict)
        assert sized == [_merge_preflight_ask(n_parameters * 4, 0)]

    def test_no_state_dict_measures_the_model(self, sized):
        """`None` is the only spelling of "the caller passed nothing"."""
        model = self._float32_model()
        n_parameters = sum(p.numel() for p in model.parameters())
        assert S._full_model_checkpoint_bytes(model, None) == n_parameters * 4
        S._preflight_merge_disk(model, "model", "lora", state_dict = None)
        assert sized == [_merge_preflight_ask(n_parameters * 4, 0)]

    def test_an_explicitly_empty_state_dict_is_not_no_state_dict(self, sized):
        """`{}` is a caller's answer, and the answer is "write nothing".

        `unsloth_generic_save` forwards the dict on `state_dict is not None`,
        so an empty one reaches `save_pretrained` and no model tensors are
        written at all. This selected on truthiness and priced the resident
        model instead, which on Kaggle moves a save that writes nothing off
        persistent storage and into a /tmp the kernel does not keep.
        """
        model = self._float32_model()
        assert S._full_model_checkpoint_bytes(model, {}) == 0
        assert S._preflight_merge_disk(model, "model", "lora", state_dict = {}) == "model"
        assert sized == [], "nothing is written, so nothing is asked for"

    def test_the_writer_really_forwards_an_empty_dict(self):
        """The sizing above is only right while this stays true."""
        import inspect

        source = inspect.getsource(S.unsloth_generic_save)
        assert (
            'if state_dict is not None:\n            _save_kwargs["state_dict"] = state_dict'
            in source
        )

    def test_both_call_sites_forward_the_dict(self, monkeypatch):
        """A parameter nothing passes measures nothing.

        Both of these accept a `state_dict` and both document `"lora"`, so
        both have to hand it on. Driven rather than read, because the string
        `state_dict = state_dict` also appears where they forward it to
        `save_pretrained`, which would pass on a body that never wires it in.
        """
        import torch

        seen = []
        monkeypatch.setattr(
            S,
            "_preflight_merge_disk",
            lambda *args, **kwargs: seen.append(kwargs.get("state_dict", "MISSING")) or args[1],
        )
        state_dict = {"weight": torch.zeros(4)}
        for function in (
            S.unsloth_save_pretrained_merged,
            S.unsloth_generic_save_pretrained_merged,
        ):
            with contextlib.suppress(Exception):
                function(
                    self._float32_model(),
                    "model",
                    tokenizer = None,
                    save_method = "lora",
                    state_dict = state_dict,
                )
        assert seen == [state_dict, state_dict]


class TestASuppliedDictIsWhatASixteenBitSaveWrites:
    """`unsloth_generic_save` writes the dictionary it was handed, cast.

    It only reaches for `model.state_dict()` when it was given none, so a
    caller-supplied one decides the size of the checkpoint: `{}` or a subset
    writes less than the resident model, and a dictionary carrying more writes
    more. Sizing the model either moves a nearly empty save off persistent
    Kaggle storage for nothing, or leaves a bigger one to fill the 20GB
    working directory it should have been redirected out of.

    `unsloth_save_model` is the other writer and rebuilds the dictionary from
    the merged layers, dropping whatever it was passed, so only the generic
    call site says the dict is followed.
    """

    @pytest.fixture
    def sized(self, monkeypatch):
        asked = []
        # Deliberately not the size of the dict below:
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        return asked

    @staticmethod
    def _model():
        import torch
        return torch.nn.Linear(8, 8, dtype = torch.float16)

    @staticmethod
    def _dict(numel = 4096):
        import torch
        return {"model.embed_tokens.weight": torch.zeros(numel, dtype = torch.float32)}

    def test_floats_are_charged_at_two_bytes_whatever_they_arrived_as(self):
        """The writer casts every floating entry to bf16/fp16 before saving."""
        assert S._cast_16bit_state_dict_bytes(self._dict(4096)) == 4096 * 2

    def test_an_integer_entry_keeps_its_own_width(self):
        import torch
        state_dict = {"buffer": torch.zeros(16, dtype = torch.int64)}
        assert S._cast_16bit_state_dict_bytes(state_dict) == 16 * 8

    def test_the_dict_is_measured_and_not_the_model(self, sized):
        state_dict = self._dict(8 * 1024**3 // 2)
        expected = S._cast_16bit_state_dict_bytes(state_dict)
        S._preflight_merge_disk(
            self._model(),
            "model",
            "merged_16bit",
            state_dict = state_dict,
            forwards_state_dict = True,
        )
        assert expected == 8 * GB
        # No adapter, so `unsloth_generic_save` casts this dictionary and writes it with a bare `save_pretrained`.
        assert sized == [_merge_preflight_ask(expected, 0)]

    def test_an_empty_dict_writes_nothing(self, sized):
        """`{}` reaches `save_pretrained` and no model tensor is written."""
        assert (
            S._preflight_merge_disk(
                self._model(),
                "model",
                "merged_16bit",
                state_dict = {},
                forwards_state_dict = True,
            )
            == "model"
        )
        assert sized == [], "nothing is written, so nothing is asked for"

    def test_no_dict_measures_the_model(self, sized):
        """`None` is when the writer builds the dictionary itself."""
        S._preflight_merge_disk(
            self._model(),
            "model",
            "merged_16bit",
            state_dict = None,
            forwards_state_dict = True,
        )
        assert sized == [_merge_preflight_ask(10 * GB, 0)]

    def test_an_adapter_merge_ignores_the_dict(self, sized, monkeypatch):
        """A PeftModel goes to `merge_and_overwrite_lora`, which takes none."""
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        S._preflight_merge_disk(
            _FakeAdapterModel(),
            "model",
            "merged_16bit",
            state_dict = self._dict(),
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        )
        # And that is the one writer whose guard is real, so this keeps its 5%.
        assert sized == [_merge_preflight_ask(10 * GB, 10 * GB)]

    def test_the_other_writer_rebuilds_the_dict(self, sized):
        """`unsloth_save_model` merges the layers itself, so the model is sized."""
        S._preflight_merge_disk(
            self._model(),
            "model",
            "merged_16bit",
            state_dict = self._dict(),
        )
        # Still a bare `save_pretrained`, so still no reserve.
        # It writes the merged shards itself and runs no zoo guard, so the model is sized at two bytes a parameter and
        assert sized == [_merge_preflight_ask(10 * GB, 0)]

    def test_each_call_site_says_what_its_writer_does_with_the_dict(self, monkeypatch):
        """Driven rather than read, so a body that never wires it in fails.

        `unsloth_save_model` rebuilds the dictionary only on the path that
        walks `.model.layers`, so that is the model this asks about.
        """
        import torch

        seen = []
        monkeypatch.setattr(
            S,
            "_preflight_merge_disk",
            lambda *args, **kwargs: (
                seen.append(
                    (
                        kwargs.get("forwards_state_dict", False),
                        kwargs.get("writes_model_verbatim", False),
                        kwargs.get("writer_runs_merge_guard", False),
                    )
                )
                or args[1]
            ),
        )
        for function in (
            S.unsloth_save_pretrained_merged,
            S.unsloth_generic_save_pretrained_merged,
        ):
            with contextlib.suppress(Exception):
                function(
                    _ModelWithLayers(),
                    "model",
                    tokenizer = None,
                    save_method = "merged_16bit",
                    state_dict = {"weight": torch.zeros(4)},
                )
        # The third flag is the merge guard:
        assert seen == [(False, False, False), (True, False, True)]

    def test_the_writers_really_differ(self):
        """The split above is only right while these two stay as they are."""
        import inspect

        generic = inspect.getsource(S.unsloth_generic_save)
        assert '("16bit" in save_method or is_qwen3_5_vlm) and state_dict is None' in generic
        assert "v.to(dtype = _target_dtype) if v.is_floating_point() else v" in generic
        # The other writer overwrites the caller's dictionary with its own.
        assert "state_dict = OrderedDict()" in inspect.getsource(S.unsloth_save_model)


class TestTheGgufSiblingIsMeasuredToo:
    """The GGUF files land in `save_directory + "_gguf"`, a SIBLING.

    So they are on the PARENT's filesystem, which is the same disk as
    `save_directory` unless that path is itself a mount point or a symlink
    onto another one. Measuring only `save_directory` then passes an export
    whose quants fill a filesystem nobody looked at.

    Only genuinely separate filesystems change anything, so a single
    filesystem behaves exactly as before.
    """

    @pytest.fixture
    def split(self, monkeypatch):
        """Free space per path, and the estimate split into its two halves."""
        state = {"free": 1000 * GB, "sibling_free": 1000 * GB, "separate": False}

        def fake_free(path):
            return state["sibling_free"] if str(path).endswith("_gguf") else state["free"]

        def fake_estimate(**kwargs):
            if kwargs.get("base_cache_copy"):
                return 48 * GB
            return 34 * GB if kwargs.get("needs_merge", True) else 18 * GB

        monkeypatch.setattr(S, "free_bytes", fake_free)
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(
            S,
            "_on_separate_filesystems",
            lambda left, right: state["separate"]
            if (str(left), str(right)) == ("model", "model_gguf")
            else False,
        )
        monkeypatch.setattr(S, "_shares_filesystem", lambda left, right: False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def test_a_tighter_sibling_filesystem_refuses(self, split):
        """`model` is a symlink onto a big disk; `model_gguf` is not."""
        split.update(free = 1000 * GB, sibling_free = 10 * GB, separate = True)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "model_gguf" in str(error.value)

    @pytest.mark.parametrize(
        "free_gb,expected",
        [(1000, ("model", True)), (40, ("model", False))],
    )
    def test_one_filesystem_is_unchanged(self, split, free_gb, expected):
        """Both probes return the same figure, so nothing new can fire."""
        split.update(free = free_gb * GB, sibling_free = free_gb * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == expected

    def test_a_roomier_sibling_is_not_a_refusal(self, split):
        """5GB holds neither half, so this is the checkpoint's own refusal."""
        split.update(free = 5 * GB, sibling_free = 1000 * GB, separate = True)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "model_gguf" not in str(error.value)

    def test_the_sibling_is_sized_without_the_checkpoint(self, split):
        """20GB holds the quants but not the merge, and only the quants go there."""
        split.update(free = 1000 * GB, sibling_free = 20 * GB, separate = True)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    def test_an_unmeasurable_sibling_leaves_the_decision_alone(self, split, monkeypatch):
        split["separate"] = True
        monkeypatch.setattr(
            S, "free_bytes", lambda path: None if str(path).endswith("_gguf") else 1000 * GB
        )
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    def test_an_unmeasurable_save_directory_leaves_the_decision_alone(self, split, monkeypatch):
        split["separate"] = True
        monkeypatch.setattr(
            S, "free_bytes", lambda path: 1 * GB if str(path).endswith("_gguf") else None
        )
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)

    def test_the_sibling_measured_is_the_redirect_target(self, split, monkeypatch):
        probed = []
        # Only the save-directory / sibling pair is split.
        monkeypatch.setattr(
            S, "kaggle_tmp_redirect", lambda *a, **k: ("/tmp/unsloth_saves/model", "moved")
        )
        monkeypatch.setattr(S, "free_bytes", lambda path: probed.append(path) or 1000 * GB)
        S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert probed == ["/tmp/unsloth_saves/model", "/tmp/unsloth_saves/model_gguf"]

    def test_an_estimator_that_cannot_size_the_sibling_keeps_the_main_guard(
        self, split, monkeypatch
    ):
        """The new call must not be able to switch the whole preflight off."""

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                raise TypeError("older unsloth_zoo")
            return 48 * GB if kwargs.get("base_cache_copy") else 34 * GB

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        split.update(free = 10 * GB, sibling_free = 10 * GB)
        with pytest.raises(RuntimeError):
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")

    def test_the_export_writes_where_the_preflight_measured(self):
        """One definition, so the two cannot drift apart."""
        import inspect

        assert S._gguf_output_directory("model") == "model_gguf"
        source = inspect.getsource(S.unsloth_save_pretrained_gguf)
        assert "gguf_directory = _gguf_output_directory(save_directory)" in source
        assert 'f"{save_directory}_gguf"' not in inspect.getsource(S._preflight_gguf_disk)


class TestEachFilesystemIsChargedForWhatItHolds:
    """Two filesystems, and each one is charged only for what lands on it.

    The other half of the sibling problem. Measuring the sibling caught the
    case where it is too tight; the aggregate estimate was still charged in
    full to the filesystem holding `save_directory`, which refuses a split
    export whose checkpoint fits on one disk and whose quants fit on the
    other. A guard that blocks an export that works is worse than no guard.

    Numbers throughout: the export is 34GB (16GB checkpoint + 18GB of
    conversion and quants), 48GB with a pre-warmed base cache.
    """

    CHECKPOINT = 16 * GB
    SIBLING = 18 * GB
    AGGREGATE = 34 * GB
    AGGREGATE_WITH_CACHE = 48 * GB

    @pytest.fixture
    def split(self, monkeypatch):
        state = {"free": 1000 * GB, "sibling_free": 1000 * GB, "separate": True}

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                return self.SIBLING
            if kwargs.get("base_cache_copy"):
                return self.AGGREGATE_WITH_CACHE
            return self.AGGREGATE

        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: state["sibling_free"] if str(path).endswith("_gguf") else state["free"],
        )
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(
            S,
            "_on_separate_filesystems",
            lambda left, right: state["separate"]
            if (str(left), str(right)) == ("model", "model_gguf")
            else False,
        )
        monkeypatch.setattr(S, "_shares_filesystem", lambda left, right: False)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        # The reserve below belongs to `merge_and_overwrite_lora`, which only a PEFT model reaches, so the tests that
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def test_a_split_export_that_fits_is_not_refused(self, split):
        """20GB holds the 16GB checkpoint; the 18GB of quants go elsewhere."""
        split.update(free = 20 * GB, sibling_free = 1000 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", False)

    @pytest.mark.parametrize("free_gb,fits", [(17, True), (16, False)])
    def test_the_checkpoint_portion_is_the_boundary(self, split, free_gb, fits):
        """`need - need_sibling`, not the aggregate, decides.

        The boundary is the 16GB checkpoint over the 0.95 reserve
        `merge_and_overwrite_lora` applies, so 16.85GB rather than 16GB.
        Exactly 16GB free is the case that motivates it: the checkpoint
        nominally fits, and the merge refuses it anyway a moment later.
        """
        split.update(free = free_gb * GB, sibling_free = 1000 * GB)
        if fits:
            assert S._preflight_gguf_disk(_FakeAdapterModel(), "model", "q4_k_m") == (
                "model",
                False,
            )
        else:
            with pytest.raises(RuntimeError) as error:
                S._preflight_gguf_disk(_FakeAdapterModel(), "model", "q4_k_m")
            assert "about 16.8GB" in str(error.value)

    def test_the_cache_copy_is_charged_here_too(self, split):
        """The pre-warm goes to the HF cache, not the sibling, so it stays here.

        48GB with the cache minus the 18GB sibling half is 30GB: at 30GB the
        cache is affordable, at 29GB it is dropped rather than refused.
        """
        split.update(free = 30 * GB, sibling_free = 1000 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", True)
        split.update(free = 29 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", False)

    def test_the_split_carries_the_merge_guards_own_reserve(self, split):
        """16GB of checkpoint on 16GB of disk is what the merge itself refuses.

        `merge_and_overwrite_lora` will not write a merge unless `free * 0.95`
        covers it, and the aggregate branch never had to think about that
        because it charges the quants too. Charging the checkpoint alone
        removes that cover, so the reserve has to come back with it or this
        passes an export the merge kills seconds later.
        """
        split.update(free = 16 * GB, sibling_free = 1000 * GB)
        with pytest.raises(RuntimeError):
            S._preflight_gguf_disk(_FakeAdapterModel(), "model", "q4_k_m")
        assert S.free_bytes("model") * S._MERGE_FREE_SPACE_RESERVE < self.CHECKPOINT

    def test_only_a_lora_merge_is_charged_the_reserve(self, split):
        """A non-PEFT checkpoint is written by `save_pretrained`, which reserves nothing.

        `needs_merge` is true for a non-PEFT model with no reusable local
        `_name_or_path` as well, because the GGUF path still has to write a
        checkpoint - but it writes it with a bare `self.save_pretrained`,
        which never consults `merge_and_overwrite_lora` and never applies its
        `free * 0.95`. Charging the reserve there refuses 16GB of checkpoint
        on 16GB of disk that the writer would have accepted.
        """
        split.update(free = 16 * GB, sibling_free = 1000 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", False)

    def test_the_non_peft_writer_really_has_no_guard(self):
        """The reserve is only skippable while this stays true."""
        import inspect

        source = inspect.getsource(S.unsloth_save_pretrained_gguf)
        fallback = source.split("Saving directly without LoRA merge")[1]
        assert "self.save_pretrained(save_directory)" in fallback
        assert "merge_and_overwrite_lora" not in fallback

    def test_an_export_writing_no_merge_is_not_charged_the_reserve(self, split):
        """`needs_merge = False` reaches no merge guard, so it pays for none.

        Nothing else is on this filesystem then, so the checkpoint portion is
        zero, the whole export is the sibling's, and 1GB is enough here.
        """
        split.update(free = 1 * GB, sibling_free = 1000 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", needs_merge = False) == (
            "model",
            True,
        )

    def test_the_reserve_never_exceeds_the_aggregate(self, split, monkeypatch):
        """A sibling small next to the checkpoint must not add a refusal.

        1GB of sibling leaves 33GB of checkpoint, and 33 / 0.95 is 34.7GB:
        more than the 34GB aggregate the single-filesystem branch charges. The
        split may cancel a redirect, never cause a refusal the aggregate would
        have allowed, so the reserved figure is clamped at `need`.
        """
        monkeypatch.setattr(
            S,
            "estimate_gguf_export_bytes",
            lambda **kwargs: (
                1 * GB
                if not kwargs.get("needs_merge", True)
                else (
                    self.AGGREGATE_WITH_CACHE if kwargs.get("base_cache_copy") else self.AGGREGATE
                )
            ),
        )
        split.update(free = 34 * GB, sibling_free = 1000 * GB)
        assert S._preflight_gguf_disk(_FakeAdapterModel(), "model", "q4_k_m") == ("model", False)

    def test_a_short_sibling_roomier_than_the_checkpoint_disk_still_refuses(self, split):
        """The hole the split would leave if the refusal still needed a TIGHTER sibling.

        10GB on the sibling holds neither the 18GB of quants nor anything
        else, and it is more than the 5GB here, so "tighter than `free`"
        would miss it, and the aggregate comparison that used to catch it is
        gone once the checkpoint is charged only its own portion.
        """
        split.update(free = 5 * GB, sibling_free = 10 * GB)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "model_gguf" in str(error.value)

    @pytest.mark.parametrize(
        "free_gb,expected",
        [
            (5, "raises"),
            (16, "raises"),
            (20, "raises"),
            (34, ("model", False)),
            (40, ("model", False)),
            (48, ("model", True)),
            (1000, ("model", True)),
        ],
    )
    def test_one_filesystem_is_unchanged_across_the_range(self, split, free_gb, expected):
        """Every outcome is the aggregate one, at every level of free space.

        This is the whole of the previous behaviour: refuse below 34GB, drop
        the pre-warm between 34GB and 48GB, proceed at 48GB. Nothing about
        the split may move any of it.
        """
        split.update(free = free_gb * GB, sibling_free = free_gb * GB, separate = False)
        if expected == "raises":
            with pytest.raises(RuntimeError) as error:
                S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
            assert "about 34.0GB" in str(error.value)
            assert "model_gguf" not in str(error.value)
        else:
            assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == expected

    def test_one_filesystem_that_moved_between_the_probes_is_still_one(self, split):
        """Two `disk_usage` calls on ONE filesystem can disagree.

        Something else writing a single block between them is not a second
        filesystem, and reading it as one would charge this export the larger
        half, 16GB, instead of the 34GB sum. The predicate is the device id,
        so the difference in free space cannot decide it.
        """
        split.update(free = 20 * GB, sibling_free = 20 * GB - 4096, separate = False)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "about 34.0GB" in str(error.value)

    def test_two_filesystems_with_the_same_free_space_are_still_two(self, split):
        """And the converse: equal figures are not evidence of one filesystem."""
        split.update(free = 20 * GB, sibling_free = 20 * GB, separate = True)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m") == ("model", False)

    def test_an_unmeasurable_sibling_charges_the_aggregate(self, split, monkeypatch):
        monkeypatch.setattr(
            S, "free_bytes", lambda path: None if str(path).endswith("_gguf") else 20 * GB
        )
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "about 34.0GB" in str(error.value)

    def test_a_sibling_the_estimator_cannot_size_charges_the_aggregate(self, split, monkeypatch):
        """`need_sibling` falls back to 0, so `need - need_sibling` is `need`."""

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                raise TypeError("older unsloth_zoo")
            return self.AGGREGATE_WITH_CACHE if kwargs.get("base_cache_copy") else self.AGGREGATE

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        split.update(free = 20 * GB, sibling_free = 1000 * GB)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
        assert "about 34.0GB" in str(error.value)

    def _outcome_before(self, free, sibling_free):
        """What the previous revision did: sibling refusal, then the aggregate."""
        if sibling_free < free and sibling_free < self.SIBLING:
            return "raises"
        if free >= self.AGGREGATE_WITH_CACHE:
            return ("model", True)
        if free >= self.AGGREGATE:
            return ("model", False)
        return "raises"

    def test_the_split_never_refuses_where_the_aggregate_allowed(self, split):
        """A guard may cancel a redirect, never cause one, and may not add refusals.

        Over a grid of both filesystems: every refusal the split produces was
        already a refusal before it, so no working export is newly blocked.
        """
        for free_gb in (1, 5, 16, 17, 20, 34, 48, 100):
            for sibling_gb in (1, 5, 17, 18, 19, 50, 1000):
                split.update(free = free_gb * GB, sibling_free = sibling_gb * GB)
                try:
                    now = S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m")
                except RuntimeError:
                    now = "raises"
                before = self._outcome_before(free_gb * GB, sibling_gb * GB)
                if now == "raises":
                    assert before == "raises", (free_gb, sibling_gb)

    def test_the_probe_matches_free_bytes_on_one_filesystem(self, tmp_path):
        """The real helper, real paths, no monkeypatching.

        A directory and its not-yet-created sibling under one tmp dir are one
        filesystem, so the split is inert on every ordinary setup.
        """
        directory = tmp_path / "model"
        directory.mkdir()
        assert S._on_separate_filesystems(str(directory), f"{directory}_gguf") is False
        assert S._filesystem_id(str(directory)) == S._filesystem_id(f"{directory}_gguf")

    def test_the_probe_sees_a_real_second_filesystem(self, tmp_path):
        """A symlinked save directory on another mount: the reachable case."""
        other = "/dev/shm"
        if not os.path.isdir(other) or os.stat(other).st_dev == os.stat(tmp_path).st_dev:
            pytest.skip("no second filesystem available here")
        directory = tmp_path / "model"
        directory.symlink_to(other)
        assert S._on_separate_filesystems(str(directory), f"{directory}_gguf") is True

    def test_an_unidentifiable_path_is_not_split(self):
        """Unmeasurable is never read as two filesystems."""

        class _Unusable:
            def __str__(self):
                raise ValueError("not a path")

        assert S._filesystem_id(_Unusable()) is None
        assert S._on_separate_filesystems(_Unusable(), "model_gguf") is False

    def test_a_zero_device_id_is_unmeasurable(self, tmp_path, monkeypatch):
        """Windows fills `st_dev` from the volume serial; zero means it did not."""
        directory = tmp_path / "model"
        directory.mkdir()

        class _Zero:
            st_dev = 0

        real_stat = os.stat
        monkeypatch.setattr(
            S.os,
            "stat",
            lambda path, *a, **k: _Zero()
            if str(path) == str(directory)
            else real_stat(path, *a, **k),
        )
        assert S._filesystem_id(str(directory)) is None
        assert S._on_separate_filesystems(str(directory), f"{directory}_gguf") is False


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


class _Packed4bitParameter:
    """A `Params4bit`: uint8 storage, with the real shape on `quant_state`.

    `numel()` is the packed count, which bitsandbytes fits two 4-bit
    parameters into. `logical_numel` is what a helper that reads
    `quant_state.shape` returns for it, and what the merge is sized with.
    """

    class _QuantState:
        def __init__(self, shape):
            self.shape = shape

    def __init__(self, logical):
        self.logical_numel = logical
        self.quant_state = self._QuantState((logical,))

    def numel(self):
        return self.logical_numel // 2


class _NamedModule(_FakeModule):
    """A `_FakeModule` that can also name its parameters, as torch does."""

    def named_parameters(self, prefix = ""):
        for index, parameter in enumerate(self._parameters):
            yield f"{prefix}weight{index}", parameter
        for name, child in self._children.items():
            yield from child.named_parameters(f"{prefix}{name}.")


class TestIgnoredModulesAreSizedFromLogicalShapes:
    """An ignored module on a 4-bit model is priced at its LOGICAL size.

    `model_16bit_bytes` sizes the merge through unsloth_zoo's `logical_numel`,
    which reads `quant_state.shape`. Sizing the ignored subtree that is
    subtracted from it with `numel()` instead prices packed uint8 storage --
    about half -- so the two disagree and the quantized sibling comes out
    small. On a tight Kaggle filesystem that is a redirect that never happens.
    """

    @pytest.fixture
    def logical(self, monkeypatch):
        """Route `logical_numel` at the fake parameters' declared logical size.

        Deliberately not a copy of the zoo's implementation: what is under
        test is that save.py ASKS, and passes the name along, not that it can
        re-derive an answer the zoo already owns.
        """
        seen = []

        def fake_logical_numel(parameter, name = ""):
            seen.append(name)
            return getattr(parameter, "logical_numel", parameter.numel())

        monkeypatch.setattr(S, "logical_numel", fake_logical_numel)
        return seen

    def _model(self):
        """Qwen3-Next shaped: a 4-bit `linear_attn` the recipe will not touch."""
        return _ShapedModel(
            children = {
                "model": _NamedModule(
                    children = {
                        "layers": _NamedModule(
                            children = {
                                "0": _NamedModule(
                                    children = {
                                        "linear_attn": _NamedModule(
                                            children = {
                                                "in_proj_qkvz": _NamedModule(
                                                    parameters = [_Packed4bitParameter(4 * GB)],
                                                ),
                                            },
                                        ),
                                    },
                                ),
                            },
                        ),
                    },
                ),
            },
            config = _Config(model_type = "qwen3_next"),
        )

    def test_a_four_bit_ignored_module_is_charged_its_logical_size(self, logical):
        model = self._model()
        patterns = S._compressed_ignore_patterns(model)
        # 4e9 logical parameters at 2 bytes, not the 2e9 packed ones.
        assert S._unquantized_parameter_bytes(model, patterns) == 8 * GB

    def test_the_packed_count_is_not_what_is_used(self, monkeypatch):
        """Without the logical lookup the same subtree prices at half."""
        monkeypatch.setattr(S, "logical_numel", lambda parameter, name = "": parameter.numel())
        model = self._model()
        patterns = S._compressed_ignore_patterns(model)
        assert S._unquantized_parameter_bytes(model, patterns) == 4 * GB

    def test_the_parameter_name_reaches_the_helper(self, logical):
        """MXFP4 packing has no `quant_state`; only the name gives it away."""
        model = self._model()
        S._unquantized_parameter_bytes(model, S._compressed_ignore_patterns(model))
        assert "weight0" in logical

    def test_the_embeddings_go_through_it_too(self, logical):
        model = _ShapedModel(
            children = {},
            config = _Config(),
            input_embeddings = _embedding(1 * GB),
        )
        assert S._unquantized_parameter_bytes(model) == 2 * GB
        assert logical, "the embedding weight was measured without the helper"

    def test_a_module_that_cannot_name_its_parameters_still_counts(self, logical):
        """`_FakeModule` has no `named_parameters`, and must still be sized."""
        model = _ShapedModel(
            children = {
                "visual": _FakeModule(parameters = [_FakeParameter(1 * GB)]),
            },
            config = _Config(),
        )
        assert S._unquantized_parameter_bytes(model, ["re:.*visual.*"]) == 2 * GB

    def test_the_helper_is_the_one_the_merge_estimate_uses(self):
        """One definition, so the two sizings cannot drift apart."""
        from unsloth import disk_utils

        assert S.logical_numel is disk_utils.logical_numel
        assert "logical_numel" in disk_utils.__all__

    def test_a_helper_that_raises_leaves_the_estimate_standing(self, monkeypatch):
        def boom(parameter, name = ""):
            raise RuntimeError("no")

        monkeypatch.setattr(S, "logical_numel", boom)
        model = self._model()
        assert S._unquantized_parameter_bytes(model, S._compressed_ignore_patterns(model)) == 0


class TestADisposableMergeIsNotChargedForAllThreeAtOnce:
    """`_free_merge_if_disk_is_tight` deletes the merge before the quants run.

    Nemotron-3-Nano-30B-A3B on a 132GB disk: a 63GB merge, a 60GB BF16
    intermediate and an 18GB Q4_K_M. The two phases peak at 123GB and 78GB,
    and the export runs. Summing all three gives 141GB and refuses it -- the
    same export the reclamation in this PR was written to make work.
    """

    AGGREGATE = 141 * GB
    AGGREGATE_WITH_CACHE = 204 * GB
    MERGE_PHASE = 123 * GB
    QUANT_PHASE = 78 * GB

    @pytest.fixture
    def phases(self, monkeypatch, tmp_path):
        state = {"free": 132 * GB, "separate": False}

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                return self.QUANT_PHASE if kwargs.get("quantization_methods") else 60 * GB
            if not kwargs.get("quantization_methods"):
                return self.MERGE_PHASE
            return self.AGGREGATE_WITH_CACHE if kwargs.get("base_cache_copy") else self.AGGREGATE

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 1000 * GB if str(path).endswith("_gguf") else state["free"],
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: (a[0], None))
        monkeypatch.setattr(
            S,
            "_on_separate_filesystems",
            lambda left, right: state["separate"] and str(right).endswith("_gguf"),
        )
        monkeypatch.setattr(S, "_shares_filesystem", lambda left, right: False)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        # A disposable merge is a LoRA merge, so the model is a PEFT one and the split branch's reserve applies to it.
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        state["directory"] = str(tmp_path / "model")
        return state

    def _preflight(self, phases, **kwargs):
        kwargs.setdefault("merge_is_disposable", True)
        return S._preflight_gguf_disk(
            _FakeAdapterModel(),
            phases["directory"],
            "q4_k_m",
            first_conversion = "bf16",
            **kwargs,
        )

    def test_the_export_the_reclamation_makes_work_is_not_refused(self, phases):
        assert self._preflight(phases) == (phases["directory"], False)

    def test_a_merge_that_is_not_disposable_still_needs_all_three(self, phases):
        """The SentenceTransformer export keeps its merge, so it pays for it."""
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases, merge_is_disposable = False)
        assert "141.0GB" in str(error.value)

    def test_the_default_is_the_aggregate(self, phases):
        """Every caller that does not opt in behaves exactly as before."""
        with pytest.raises(RuntimeError):
            S._preflight_gguf_disk(
                _FakeModel(), phases["directory"], "q4_k_m", first_conversion = "bf16"
            )

    @pytest.mark.parametrize("free_gb,fits", [(123, True), (122, False)])
    def test_the_larger_phase_is_the_boundary(self, phases, free_gb, fits):
        phases.update(free = free_gb * GB)
        if fits:
            assert self._preflight(phases) == (phases["directory"], False)
        else:
            with pytest.raises(RuntimeError) as error:
                self._preflight(phases)
            assert "123.0GB" in str(error.value)

    def test_the_cache_copy_rides_on_top_of_the_peak(self, phases):
        """204 - 141 = 63GB of cached base, on top of the 123GB peak."""
        phases.update(free = 186 * GB)
        assert self._preflight(phases) == (phases["directory"], True)
        phases.update(free = 185 * GB)
        assert self._preflight(phases) == (phases["directory"], False)

    def test_an_export_with_no_merge_is_unchanged(self, phases):
        """Nothing was written here to reclaim, so the sibling total stands."""
        phases.update(free = 70 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases, needs_merge = False)
        assert "78.0GB" in str(error.value)

    def test_a_single_pass_export_gets_no_relief(self, phases):
        """No quantize pass follows, so there is nothing to reclaim for."""
        phases.update(free = 132 * GB)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(
                _FakeModel(),
                phases["directory"],
                "bf16",
                first_conversion = "bf16",
                merge_is_disposable = True,
            )
        assert "141.0GB" in str(error.value)

    def test_a_reused_output_directory_gets_no_relief(self, phases):
        """The reclamation never deletes weights it did not write."""
        directory = phases["directory"]
        os.makedirs(directory, exist_ok = True)
        with open(os.path.join(directory, "model.safetensors"), "w") as handle:
            handle.write("x")
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases)
        assert "141.0GB" in str(error.value)

    def test_an_unrelated_file_in_the_output_directory_is_not_a_merge(self, phases):
        """A training `output_dir` holding an optimizer state still qualifies."""
        directory = phases["directory"]
        os.makedirs(directory, exist_ok = True)
        with open(os.path.join(directory, "optimizer.pt"), "w") as handle:
            handle.write("x")
        assert self._preflight(phases) == (directory, False)

    def test_an_unreadable_output_directory_gets_no_relief(self, phases, monkeypatch):
        def boom(path):
            raise PermissionError("no")

        monkeypatch.setattr(S.os, "listdir", boom)
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases)
        assert "141.0GB" in str(error.value)

    def test_split_storage_charges_each_side_instead(self, phases):
        """The reclamation declines across filesystems, so the relief must too."""
        phases.update(separate = True, free = 62 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases)
        assert "66.3GB" in str(error.value)

    def test_it_can_only_ever_lower_the_figure(self, phases, monkeypatch):
        """An estimator whose phases exceed the aggregate changes nothing."""
        # here, so it is never charged to the save directory's disk.
        # Only the save-directory / sibling pair can be split.
        monkeypatch.setattr(
            S,
            "estimate_gguf_export_bytes",
            lambda **kwargs: 300 * GB
            if kwargs.get("base_cache_copy")
            else (200 * GB if not kwargs.get("quantization_methods") else 141 * GB),
        )
        phases.update(free = 132 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases)
        assert "141.0GB" in str(error.value)

    def test_an_estimator_that_cannot_size_a_phase_charges_the_aggregate(self, phases, monkeypatch):
        real = S.estimate_gguf_export_bytes

        def fake_estimate(**kwargs):
            if kwargs.get("needs_merge", True) and not kwargs.get("quantization_methods"):
                raise RuntimeError("no")
            return real(**kwargs)

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        # 141 aggregate - 78 sibling = 63GB of checkpoint, 66.3GB once the merge's own 0.95 reserve is on it, and 62GB
        with pytest.raises(RuntimeError) as error:
            self._preflight(phases)
        assert "141.0GB" in str(error.value)

    def _redirect_ask(self, phases, monkeypatch, **kwargs):
        """What `kaggle_tmp_redirect` is asked for, with the move declined.

        On Kaggle, which is the only environment the redirect fires in, and
        where nothing is ever charged for a cache copy.
        """
        asked = []
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", True)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        try:
            self._preflight(phases, **kwargs)
        except RuntimeError:
            # The ask is recorded before the refusal, and it is the ask this is about:
            pass
        return asked

    def test_the_redirect_is_asked_for_the_peak_and_not_the_aggregate(self, phases, monkeypatch):
        """Or Kaggle relocates an export that fits to a /tmp it does not keep.

        The refusal below reads the 123GB peak, so the redirect above it has
        to as well: asked for the 141GB aggregate, a 132GB /kaggle/working
        looks too small and the export is moved off notebook storage.
        """
        assert self._redirect_ask(phases, monkeypatch) == [self.MERGE_PHASE]

    def test_a_merge_that_is_not_disposable_still_asks_the_aggregate(self, phases, monkeypatch):
        asked = self._redirect_ask(phases, monkeypatch, merge_is_disposable = False)
        assert asked == [self.AGGREGATE]

    def test_split_storage_asks_the_aggregate(self, phases, monkeypatch):
        """The reclamation declines across filesystems, so the ask must too."""
        phases.update(separate = True)
        asked = self._redirect_ask(phases, monkeypatch)
        assert asked == [self.AGGREGATE]

    def _reuse(self, phases):
        """Leave a previous export's checkpoint in the output directory."""
        directory = phases["directory"]
        os.makedirs(directory, exist_ok = True)
        with open(os.path.join(directory, "model.safetensors"), "w") as handle:
            handle.write("x")
        return directory

    def test_a_reused_output_directory_asks_the_aggregate_first(self, phases, monkeypatch):
        """Then asks the peak, because the move writes into a fresh directory.

        The reclamation this directory cannot offer is available at the
        redirect target, so a declined move is worth a second ask: 132GB free
        here is already less than the 141GB aggregate, and the export really
        does peak at 123GB once it is relocated.
        """
        self._reuse(phases)
        assert self._redirect_ask(phases, monkeypatch) == [self.AGGREGATE, self.MERGE_PHASE]

    def test_a_reused_output_directory_with_room_here_is_asked_once(self, phases, monkeypatch):
        """Nothing is refused here, so there is nothing a move could rescue."""
        self._reuse(phases)
        phases.update(free = self.AGGREGATE)
        assert self._redirect_ask(phases, monkeypatch) == [self.AGGREGATE]

    def _zoo_redirect(self, phases, monkeypatch, tmp_path, working_free, tmp_free):
        """Run the preflight against `kaggle_tmp_redirect`'s own move rule.

        Returns `(asks, directory)`, with `directory` the one the export ends
        up writing to.
        """
        target = str(tmp_path / "overlay" / "unsloth_saves" / "model")
        asked = []

        def redirect(
            save_directory,
            need_bytes = 0,
            what = "export",
        ):
            asked.append(need_bytes)
            if tmp_free <= working_free or need_bytes <= 0:
                return save_directory, None
            if working_free >= need_bytes or tmp_free < need_bytes:
                return save_directory, None
            os.makedirs(target, exist_ok = True)
            return target, f"Unsloth: moved to {target}"

        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", True)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", redirect)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: tmp_free if str(path).startswith(target) else working_free,
        )
        directory, _ = self._preflight(phases)
        return asked, directory

    def test_a_reused_directory_is_relocated_instead_of_refused(
        self, phases, monkeypatch, tmp_path, capsys
    ):
        """100GB here, 130GB on the overlay: the 123GB peak fits after the move.

        Asked only the 141GB aggregate, the overlay declines it too and the
        export is refused on a filesystem it never had to use.
        """
        self._reuse(phases)
        phases.update(free = 100 * GB)
        asked, directory = self._zoo_redirect(
            phases, monkeypatch, tmp_path, working_free = 100 * GB, tmp_free = 130 * GB
        )
        assert asked == [self.AGGREGATE, self.MERGE_PHASE]
        assert directory == str(tmp_path / "overlay" / "unsloth_saves" / "model")
        assert "moved to" in capsys.readouterr().out

    def test_a_move_the_aggregate_would_have_made_is_not_cancelled(
        self, phases, monkeypatch, tmp_path
    ):
        """130GB here holds the peak but not the aggregate, and no merge is

        reclaimable here, so the refusal would read 141GB. Asking the peak
        outright keeps the export on a filesystem that cannot run it; the
        second ask only ever follows a move this directory could not avoid.
        """
        self._reuse(phases)
        phases.update(free = 130 * GB)
        asked, directory = self._zoo_redirect(
            phases, monkeypatch, tmp_path, working_free = 130 * GB, tmp_free = 200 * GB
        )
        assert asked == [self.AGGREGATE]
        assert directory == str(tmp_path / "overlay" / "unsloth_saves" / "model")

    def test_save_to_gguf_passes_the_flag_through(self):
        """The preflight and the reclamation must agree about disposability."""
        import inspect

        source = inspect.getsource(S.unsloth_save_pretrained_gguf)
        assert "merge_is_disposable = merge_is_disposable" in source
        assert source.index("_preflight_gguf_disk(") < source.index(
            'del arguments["merge_is_disposable"]'
        )


class TestTheConversionWorkingDirectoryIsMeasured:
    """The intermediate GGUF is written to the process CWD, then moved.

    `unsloth_zoo.llama_cpp.convert_to_gguf` passes a bare `--outfile`, which
    llama.cpp resolves against the CWD; only an unwritable CWD falls back to
    the input folder. So a Kaggle export redirected to /tmp still writes its
    largest staging artefact into the 20GB /kaggle/working, and that was the
    one filesystem this preflight never measured.
    """

    TMP = "/tmp/unsloth_saves/model"
    WORKING = "/kaggle/working"

    @pytest.fixture
    def kaggle(self, monkeypatch):
        state = {"cwd_free": 19 * GB, "conversion": self.WORKING}

        def fake_estimate(**kwargs):
            total = 60 * GB if kwargs.get("needs_merge", True) else 0
            total += 30 * GB
            if kwargs.get("quantization_methods"):
                total += 9 * GB
            return total

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 1000 * GB if str(path).startswith("/tmp") else state["cwd_free"],
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: (self.TMP, "moved to /tmp"))
        monkeypatch.setattr(
            S,
            "_on_separate_filesystems",
            lambda left, right: str(left).startswith("/tmp") != str(right).startswith("/tmp"),
        )
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: state["conversion"])
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self):
        return S._preflight_gguf_disk(
            _FakeModel(), "/kaggle/working/model", "q4_k_m", first_conversion = "bf16"
        )

    def test_a_redirect_that_cannot_hold_the_conversion_is_refused(self, kaggle):
        with pytest.raises(RuntimeError) as error:
            self._preflight()
        message = str(error.value)
        assert self.WORKING in message and f"{self.TMP}_gguf" in message
        assert "19.0GB free" in message and "30.0GB" in message
        assert "UNSLOTH_DISK_PREFLIGHT=0" in message

    def test_a_working_directory_with_room_changes_nothing(self, kaggle):
        kaggle.update(cwd_free = 31 * GB)
        assert self._preflight() == (self.TMP, True)

    def test_the_boundary_is_the_conversion_alone(self, kaggle):
        """Not the whole export: only the intermediate passes through here."""
        kaggle.update(cwd_free = 30 * GB)
        assert self._preflight() == (self.TMP, True)
        kaggle.update(cwd_free = 29 * GB)
        with pytest.raises(RuntimeError):
            self._preflight()

    def test_a_working_directory_on_the_same_filesystem_is_not_charged(self, kaggle):
        """Then the move is a rename and the bytes are already counted."""
        kaggle.update(conversion = "/tmp/somewhere", cwd_free = 1 * GB)
        assert self._preflight() == (self.TMP, True)

    def test_an_unmeasurable_working_directory_leaves_the_decision_alone(self, kaggle):
        kaggle.update(cwd_free = None)
        assert self._preflight() == (self.TMP, True)

    def test_a_working_directory_that_cannot_be_identified_is_not_charged(self, kaggle):
        kaggle.update(conversion = None)
        assert self._preflight() == (self.TMP, True)

    def test_it_is_not_reached_when_the_preflight_is_disabled(self, kaggle, monkeypatch):
        monkeypatch.setenv("UNSLOTH_DISK_PREFLIGHT", "0")
        assert self._preflight() == ("/kaggle/working/model", True)


class TestWhereTheConversionWrites:
    """`_gguf_conversion_directory` mirrors `convert_to_gguf`'s own rule."""

    def test_a_writable_working_directory_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert S._gguf_conversion_directory("model") == os.getcwd()

    def test_an_unwritable_working_directory_falls_back_to_the_model(self, monkeypatch):
        monkeypatch.setattr(S, "_directory_is_writable", lambda directory: False)
        assert S._gguf_conversion_directory("model") == "model"

    def test_an_unreadable_working_directory_is_unmeasurable(self, monkeypatch):
        def boom():
            raise OSError("no")

        monkeypatch.setattr(S.os, "getcwd", boom)
        assert S._gguf_conversion_directory("model") is None

    def test_the_probe_answers_for_real_directories(self, tmp_path):
        assert S._directory_is_writable(str(tmp_path)) is True
        assert S._directory_is_writable(str(tmp_path / "nope")) is False

    def test_the_probe_leaves_nothing_behind(self, tmp_path):
        S._directory_is_writable(str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestAColocatedConversionIsChargedWithTheCheckpoint:
    """Split storage, and the conversion lands on the checkpoint's disk.

    `save_directory` is a mount, so the `_gguf` sibling is on another
    filesystem and the split branch charges this one for the checkpoint alone.
    The intermediate conversion goes to the working directory and is only
    moved to the sibling afterwards, so when that working directory is on the
    same filesystem as `save_directory` the two sit there together - and the
    conversion check charged this disk for the conversion alone. A 60GB
    checkpoint and a 60GB conversion each passed on 100GB, and then it filled.

    Numbers: 120GB aggregate, 60GB of it the sibling's, so a 60GB checkpoint
    and a 60GB conversion.
    """

    CHECKPOINT = 60 * GB
    SIBLING = 60 * GB
    CONVERSION = 60 * GB

    @pytest.fixture
    def colocated(self, monkeypatch):
        # One device map rather than a stub per pair, so the three paths cannot describe a machine that does not exist
        state = {"free": 100 * GB, "devices": {"model": 1, "model_gguf": 2, "work": 1}}

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                return self.SIBLING if kwargs.get("quantization_methods") else self.CONVERSION
            return self.CHECKPOINT + self.SIBLING

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 1000 * GB if str(path).endswith("_gguf") else state["free"],
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: "work")
        # `model` is the mount and `model_gguf` is not, so the export is split;
        monkeypatch.setattr(S, "_filesystem_id", lambda path: state["devices"].get(str(path)))
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self):
        return S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", first_conversion = "bf16")

    def test_both_artefacts_are_charged_to_the_one_filesystem(self, colocated):
        """100GB holds either alone and not the pair."""
        with pytest.raises(RuntimeError) as error:
            self._preflight()
        assert "120.0GB" in str(error.value)

    def test_room_for_the_pair_is_not_refused(self, colocated):
        colocated.update(free = 120 * GB)
        assert self._preflight() == ("model", True)

    def test_a_conversion_elsewhere_is_charged_only_once(self, colocated):
        """The working directory on the sibling's disk leaves this one alone."""
        colocated["devices"]["work"] = 2
        assert self._preflight() == ("model", True)

    def test_an_unmeasurable_working_directory_charges_nothing_extra(self, colocated, monkeypatch):
        """`_shares_filesystem` says no to what it cannot see, so this is unchanged."""
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: None)
        assert self._preflight() == ("model", True)

    def test_the_predicate_never_guesses(self, monkeypatch):
        monkeypatch.setattr(S, "_filesystem_id", lambda path: None)
        assert S._shares_filesystem("a", "b") is False
        assert S._on_separate_filesystems("a", "b") is False


class TestTheMergeGuardAndTheConversionAreTwoPhases:
    """The same colocated split, with a LoRA merge in front of it.

    `merge_and_overwrite_lora` runs first and wants the checkpoint covered by
    `free * 0.95`, with nothing else on the disk yet. The conversion is
    written afterwards, next to the finished checkpoint and against no guard
    at all. Adding the conversion to the reserved figure charges both at once
    and asks for a peak that never exists.

    Numbers: a 60GB checkpoint and a 60GB conversion. The merge phase wants
    63.2GB and the conversion phase 120GB, so 122GB of disk clears both, and
    the sum asks 123.2GB and refuses it.
    """

    CHECKPOINT = 60 * GB
    SIBLING = 60 * GB
    CONVERSION = 60 * GB

    @pytest.fixture
    def colocated_merge(self, monkeypatch):
        state = {"free": 122 * GB, "conversion": self.CONVERSION}

        def fake_estimate(**kwargs):
            if not kwargs.get("needs_merge", True):
                return self.SIBLING if kwargs.get("quantization_methods") else state["conversion"]
            return self.CHECKPOINT + self.SIBLING

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", fake_estimate)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 1000 * GB if str(path).endswith("_gguf") else state["free"],
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: "work")
        monkeypatch.setattr(
            S,
            "_filesystem_id",
            lambda path: 2 if str(path) == "model_gguf" else 1,
        )
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self):
        return S._preflight_gguf_disk(
            _FakeAdapterModel(), "model", "q4_k_m", first_conversion = "bf16"
        )

    def test_the_taller_phase_decides_and_not_their_sum(self, colocated_merge):
        """122GB clears the 63.2GB merge and then the 120GB pair."""
        assert self._preflight() == ("model", True)

    def test_the_pair_is_still_charged_in_full(self, colocated_merge):
        colocated_merge.update(free = 119 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight()
        assert "120.0GB" in str(error.value)

    def test_the_merge_reserve_survives_a_small_conversion(self, colocated_merge):
        """A 1GB conversion leaves the merge phase the taller of the two.

        61GB holds the checkpoint and the conversion together and is still
        less than the 63.2GB `merge_and_overwrite_lora` insists on, so this
        has to refuse: taking the maximum must not drop the reserve.
        """
        colocated_merge.update(conversion = 1 * GB, free = 62 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight()
        assert "63.2GB" in str(error.value)

    def test_a_non_peft_export_is_unchanged(self, colocated_merge):
        """No merge guard, so the requirement is the pair and nothing more."""
        colocated_merge.update(free = 120 * GB)
        assert S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", first_conversion = "bf16") == (
            "model",
            True,
        )

    def test_the_guard_really_runs_before_the_conversion(self):
        """The two phases are only separate while the merge is written first."""
        import inspect

        source = inspect.getsource(S.unsloth_save_pretrained_gguf)
        assert source.index("unsloth_generic_save(**arguments)") < source.index("save_to_gguf(")
        assert "merge_and_overwrite_lora" in inspect.getsource(S.unsloth_generic_save)


class TestTheFallbackFollowsTheReusedCheckpoint:
    """An unwritable CWD sends the conversion to the folder the converter READS.

    `convert_to_gguf` redirects a bare `--outfile` into `input_folder`, and for
    a non-PEFT model with a local `_name_or_path` that folder is the reused
    checkpoint, not the requested output: `unsloth_save_pretrained_gguf`
    reassigns `save_directory` to it before the conversion runs. Probing the
    requested output there measures a filesystem nothing is written to, while
    the intermediate fills the checkpoint's.
    """

    class _NonPeftFromDisk:
        def __init__(self, directory):
            self.config = type("cfg", (), {"_name_or_path": directory})()

    def test_the_reused_checkpoint_is_the_input_folder(self, tmp_path):
        model = self._NonPeftFromDisk(str(tmp_path))
        assert S._gguf_model_input_directory(model, "model") == str(tmp_path)

    def test_a_merged_model_still_reads_the_output(self, tmp_path, monkeypatch):
        """A PEFT merge writes the checkpoint into `save_directory` and converts it."""
        # Only a PeftModel reaches the merge guard whose reserve this is about.
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        assert S._gguf_model_input_directory(_FakeAdapterModel(), "model") == "model"

    def test_a_hub_id_is_not_a_directory(self):
        model = self._NonPeftFromDisk("unsloth/Qwen3-32B")
        assert S._gguf_model_input_directory(model, "model") == "model"

    def test_a_model_with_no_config_reads_the_output(self):
        assert S._gguf_model_input_directory(_FakeModel(), "model") == "model"

    def test_the_probe_measures_the_checkpoints_filesystem(self, tmp_path, monkeypatch):
        """End to end: the refusal names the checkpoint, not the requested output."""
        checkpoint = tmp_path / "base"
        checkpoint.mkdir()
        model = self._NonPeftFromDisk(str(checkpoint))
        monkeypatch.setattr(S, "_directory_is_writable", lambda directory: False)
        # `model` and the working directory are one mount, the `_gguf` sibling another, so the export is split and the
        monkeypatch.setattr(
            S,
            "estimate_gguf_export_bytes",
            lambda **kwargs: 30 * GB if kwargs.get("quantization_methods") else 20 * GB,
        )
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: 1 * GB if str(path) == str(checkpoint) else 1000 * GB,
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: (a[0], None))
        monkeypatch.setattr(
            S,
            "_on_separate_filesystems",
            lambda left, right: str(left) != str(right),
        )
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        with pytest.raises(RuntimeError) as error:
            S._preflight_gguf_disk(
                model, "output", "q4_k_m", first_conversion = "bf16", needs_merge = False
            )
        message = str(error.value)
        assert str(checkpoint) in message
        assert "1.0GB free" in message and "20.0GB" in message

    def test_the_converter_really_falls_back_to_its_input_folder(self):
        """The redirect is only worth following while llama_cpp does this."""
        import inspect

        from unsloth_zoo import llama_cpp

        source = inspect.getsource(llama_cpp.convert_to_gguf)
        assert "os.path.abspath(input_folder)" in source
        assert '"--outfile": output_file' in source


class TestWhetherTheMergeCanBeReclaimed:
    """`_merge_reclamation_is_possible` asks the question the reclamation asks."""

    def test_a_directory_that_does_not_exist_yet(self, tmp_path):
        assert S._merge_reclamation_is_possible(str(tmp_path / "new")) is True

    def test_an_empty_directory(self, tmp_path):
        assert S._merge_reclamation_is_possible(str(tmp_path)) is True

    def test_a_directory_already_holding_a_checkpoint(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("x")
        assert S._merge_reclamation_is_possible(str(tmp_path)) is False

    def test_a_directory_holding_only_training_artifacts(self, tmp_path):
        (tmp_path / "optimizer.pt").write_text("x")
        (tmp_path / "training_args.bin").write_text("x")
        assert S._merge_reclamation_is_possible(str(tmp_path)) is True

    def test_an_unreadable_directory(self, monkeypatch):
        def boom(path):
            raise PermissionError("no")

        monkeypatch.setattr(S.os, "listdir", boom)
        assert S._merge_reclamation_is_possible("model") is False


class TestTheGenericFallbackCopiesWhatItHolds:
    """`unsloth_save_model` rebuilds a dictionary for ONE architecture layout.

    Everything else -- GPT-2 style, custom heads, anything without
    `.model.layers` -- falls to the generic branch, which calls
    `save_pretrained(**save_pretrained_settings)` with the caller's dictionary
    still in it and with no cast at all. So the checkpoint is that
    dictionary's own bytes at its own dtypes, and a supplied fp32 dictionary
    is twice what a 16-bit merge would have been.
    """

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

    @staticmethod
    def _fp32_dict(numel = 4 * 1024**3 // 4):
        import torch
        return {"model.embed_tokens.weight": torch.zeros(numel, dtype = torch.float32)}

    def test_the_dispositions_split_on_the_layout(self):
        import torch
        assert S._merge_writer_disposition(torch.nn.Linear(8, 8), "merged_16bit") == (True, True)
        assert S._merge_writer_disposition(_ModelWithLayers(), "merged_16bit") == (False, False)

    def test_a_supplied_dict_is_written_uncast(self, sized):
        """4GB of fp32 stays 4GB here, where a cast would have called it 2GB."""
        import torch

        state_dict = self._fp32_dict()
        S._preflight_merge_disk(
            torch.nn.Linear(8, 8),
            "model",
            "merged_16bit",
            state_dict = state_dict,
            forwards_state_dict = True,
            writes_model_verbatim = True,
        )
        assert S._cast_16bit_state_dict_bytes(state_dict) == 2 * GB
        assert sized == [_merge_preflight_ask(4 * GB, 0)]

    def test_no_dict_is_the_resident_model_at_its_own_dtype(self, sized):
        """A bare `save_pretrained` casts nothing, so fp32 parameters cost four bytes."""
        import torch

        model = torch.nn.Linear(1024, 1024, bias = False, dtype = torch.float32)
        S._preflight_merge_disk(
            model,
            "model",
            "merged_16bit",
            forwards_state_dict = True,
            writes_model_verbatim = True,
        )
        assert sized == [_merge_preflight_ask(1024 * 1024 * 4, 0)]

    def test_nothing_is_reserved_for_a_merge_that_never_runs(self, sized):
        """No `merge_and_overwrite_lora` here, so no `free * 0.95` either."""
        import torch

        S._preflight_merge_disk(
            torch.nn.Linear(8, 8),
            "model",
            "merged_16bit",
            state_dict = self._fp32_dict(),
            forwards_state_dict = True,
            writes_model_verbatim = True,
        )
        assert sized == [4 * GB], "a reserve would round this up"

    def test_an_adapter_is_left_exactly_as_it_was(self, sized, monkeypatch):
        """A PeftModel in that branch writes adapters, not a checkpoint."""
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        S._preflight_merge_disk(
            _FakeAdapterModel(),
            "model",
            "merged_16bit",
            state_dict = self._fp32_dict(),
            forwards_state_dict = True,
            writes_model_verbatim = True,
        )
        # Sized from the model rather than the dictionary, and unreserved: the writer these two flags describe is
        # `unsloth_save_model`, which merges and writes the shards itself with no `merge_and_overwrite_lora` anywhere
        # behind it.
        assert sized == [_merge_preflight_ask(10 * GB, 0)]
        assert S._merge_writer_disposition(_FakeAdapterModel(), "merged_16bit") == (False, False)

    def test_the_writer_really_still_forwards_it(self):
        """The split above is only right while `unsloth_save_model` stays as it is."""
        import inspect

        source = inspect.getsource(S.unsloth_save_model)
        assert (
            'or (not hasattr(model, "model") or not hasattr(internal_model.model, "layers"))'
            in source
        )
        assert "model.save_pretrained(**save_pretrained_settings)" in source


class TestASpecialExportStagesFromTheSuppliedDict:
    """compressed-tensors and torchao merge through `unsloth_generic_save` too.

    Both hand it `save_method = "merged_16bit"` and the caller's dictionary, so
    that dictionary is what the kept (compressed) or staged (torchao) merge
    costs. Sizing the resident model there prices a checkpoint that is not the
    one being written.
    """

    @pytest.fixture
    def sized(self, monkeypatch):
        asked = []
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: 10 * GB)
        monkeypatch.setattr(S, "_unquantized_parameter_bytes", lambda model, patterns = (): 0)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        return asked

    @staticmethod
    def _dict(gigabytes = 8):
        import torch
        return {
            "model.embed_tokens.weight": torch.zeros(gigabytes * 1024**3 // 2, dtype = torch.float16)
        }

    def test_a_compressed_export_measures_the_dict(self, sized):
        """8GB of merge plus its fp8 sibling, not the 10GB resident model."""
        import torch

        S._preflight_merge_disk(
            torch.nn.Linear(8, 8),
            "model",
            "fp8",
            state_dict = self._dict(8),
            forwards_state_dict = True,
        )
        # so the 5% band never showed up in this figure anyway.
        # Unreserved: with no adapter, `unsloth_generic_save` casts the dict and writes it, and the sibling is a
        assert sized == [_merge_preflight_ask(8 * GB + 4 * GB, 0)]

    def test_a_torchao_export_measures_the_dict(self, sized):
        """The staging merge is the dict; only the 8-bit sibling lands here."""
        import torch

        S._preflight_merge_disk(
            torch.nn.Linear(8, 8),
            "model",
            "torchao_fp8",
            state_dict = self._dict(8),
            forwards_state_dict = True,
        )
        assert sized == [_merge_preflight_ask(4 * GB, 0)]

    def test_the_special_methods_forward_without_casting_twice(self):
        import torch
        for method in ("fp8", "mxfp4", "torchao_fp8", "torchao_int8"):
            assert S._merge_writer_disposition(torch.nn.Linear(8, 8), method) == (
                True,
                False,
            ), method

    def test_both_special_writers_really_pass_the_dict_on(self):
        import inspect

        for function in (S._unsloth_save_compressed_tensors, S._unsloth_save_torchao):
            source = inspect.getsource(function)
            assert "unsloth_generic_save(" in source
            assert "merge_kwargs" in source
        entrypoint = inspect.getsource(S.unsloth_save_pretrained_merged)
        assert entrypoint.count("state_dict = state_dict,") >= 2


class TestOnlyTheGuardedWriterIsCharged:
    """The 5% reserve belongs to `merge_and_overwrite_lora` and to nothing else.

    One function in this module calls it: `unsloth_generic_save`, and only on
    its adapter branch. Every other writer that lands a 16-bit checkpoint at
    `save_directory` is a bare `save_pretrained` that reserves nothing, so
    charging it `1 / 0.95` moves an export off persistent Kaggle storage that
    its writer would have accepted.

    The reserve is decided apart from the sizing on purpose: a compressed
    export IS cast to two bytes by that writer and keeps that sizing, adapter
    or not.
    """

    MERGE = 10 * GB

    @pytest.fixture
    def sized(self, monkeypatch):
        asked = []
        monkeypatch.setattr(S, "model_16bit_bytes", lambda model: self.MERGE)
        monkeypatch.setattr(S, "_unquantized_parameter_bytes", lambda model, patterns = (): 0)
        monkeypatch.setattr(
            S,
            "kaggle_tmp_redirect",
            lambda save_directory, need_bytes = 0, what = "export": (
                asked.append(need_bytes) or (save_directory, None)
            ),
        )
        return asked

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setattr(S, "PeftModel", _FakeAdapterModel)
        return _FakeAdapterModel()

    def test_a_generic_sixteen_bit_save_with_no_adapter_reserves_nothing(self, sized):
        """`unsloth_generic_save` casts the dict and writes it, with no merge.

        The reserve here redirected a checkpoint that fits in the 5% band.
        """
        S._preflight_merge_disk(
            _FakeModel(),
            "model",
            "merged_16bit",
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        )
        assert sized == [self.MERGE]
        assert sized != [_with_merge_headroom(self.MERGE)]

    @pytest.mark.parametrize("save_method", ["torchao_fp8", "torchao_int8"])
    def test_a_torchao_export_reserves_nothing_here(self, sized, adapter, save_method):
        """It merges into a temp dir, so no guard runs against THIS filesystem.

        True with an adapter as well, which is the case that has a guard at
        all; it just runs somewhere else.
        """
        S._preflight_merge_disk(
            adapter,
            "model",
            save_method,
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        )
        assert sized == [self.MERGE // 2], "the sibling alone, at face value"

    def test_a_compressed_export_with_no_adapter_reserves_nothing(self, sized):
        """Sized as a cast 16-bit merge, because it is one, and still unguarded."""
        S._preflight_merge_disk(_FakeModel(), "model", "fp8", forwards_state_dict = True)
        assert sized == [self.MERGE + self.MERGE // 2]

    def test_an_adapter_merged_by_the_generic_writer_keeps_its_reserve(self, sized, adapter):
        """The one case the guard is real: it must survive all of the above."""
        S._preflight_merge_disk(
            adapter,
            "model",
            "merged_16bit",
            forwards_state_dict = True,
            writer_runs_merge_guard = True,
        )
        assert sized == [_with_merge_headroom(self.MERGE)]

    def test_a_compressed_export_of_an_adapter_keeps_its_reserve(self, sized, adapter):
        """Both entrypoints route a compressed export through the same writer.

        So the method alone settles it and the flag is not needed.
        """
        S._preflight_merge_disk(adapter, "model", "fp8", forwards_state_dict = True)
        # The sibling is half the merge, well past the 5%, so the aggregate is the binding figure.
        assert sized == [max(self.MERGE + self.MERGE // 2, _with_merge_headroom(self.MERGE))]

    def test_the_plain_entrypoint_runs_no_guard_at_all(self, sized, adapter):
        """`unsloth_save_pretrained_merged` merges in `unsloth_save_model`.

        Which writes the merged shards itself and never calls the guarded
        function, adapter or not.
        """
        S._preflight_merge_disk(adapter, "model", "merged_16bit")
        assert sized == [self.MERGE]

    def test_the_guarded_writer_is_the_only_one_in_the_module(self):
        """The split above holds only while that stays true."""
        import inspect

        source = inspect.getsource(S)
        assert source.count("merge_and_overwrite_lora(\n") == 1
        generic = inspect.getsource(S.unsloth_generic_save)
        before, _, after = generic.partition("merge_and_overwrite_lora(")
        # The call sits on the adapter branch, and the no-adapter branch above it writes with `save_pretrained`.
        assert "if not _is_peft:" in before
        assert "model.save_pretrained(save_directory, **_save_kwargs)" in before
        assert after.strip(), "the call takes arguments"


class TestTheGgufPreflightIsToldTheModelDtype:
    """`estimate_gguf_export_bytes` drops a requested output that EQUALS the
    initial conversion, so naming the wrong dtype hides a whole checkpoint.

    A bf16 model asked for `["f16", "q4_k_m"]` writes a bf16 intermediate AND
    a separate f16 file. Told "f16", the estimate charges one 16-bit file
    where the export writes two.
    """

    class _Config:
        def __init__(self, dtype):
            self.torch_dtype = dtype
            self.dtype = dtype
            self._name_or_path = "unsloth/model"

    class _Model:
        def __init__(self, config):
            self.config = config

    @pytest.fixture(autouse = True)
    def bf16_hardware(self, monkeypatch):
        monkeypatch.setattr(S.torch.cuda, "is_bf16_supported", lambda: True)

    def test_a_bfloat16_model_reports_bf16(self):
        import torch
        assert S._gguf_source_dtype(self._Model(self._Config(torch.bfloat16))) == "bf16"
        assert S._gguf_source_dtype(self._Model(self._Config("bfloat16"))) == "bf16"

    def test_a_float16_model_reports_f16(self):
        import torch
        assert S._gguf_source_dtype(self._Model(self._Config(torch.float16))) == "f16"

    def test_hardware_without_bf16_falls_back_like_the_exporter(self, monkeypatch):
        import torch
        monkeypatch.setattr(S.torch.cuda, "is_bf16_supported", lambda: False)
        assert S._gguf_source_dtype(self._Model(self._Config(torch.bfloat16))) == "f16"

    def test_an_unreadable_model_reports_the_exporters_own_fallback(self):
        assert S._gguf_source_dtype(_FakeModel()) == "f16"

    def test_the_wrong_dtype_undercounts_a_whole_checkpoint(self, monkeypatch):
        """The zoo estimator, transcribed at the top of this file, priced for real."""
        n = 8_190_735_360

        def estimate(
            model = None,
            quantization_methods = (),
            first_conversion = "f16",
            needs_merge = True,
            n_parameters = None,
            base_cache_copy = False,
        ):
            bits = {"f16": 16.0, "bf16": 16.0, "q4_k_m": 4.9}
            total = n * 2 if needs_merge else 0
            total += int(n * bits[first_conversion] / 8)
            for method in dict.fromkeys(quantization_methods):
                if method != first_conversion:
                    total += int(n * bits[method] / 8)
            return total

        asked = []
        monkeypatch.setattr(S, "estimate_gguf_export_bytes", estimate)
        monkeypatch.setattr(S, "free_bytes", lambda path: 1000 * GB)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(
            S,
            "_fallback_checkpoint_extra_bytes",
            lambda model: asked.append(None) or 0,
        )
        wrong = estimate(quantization_methods = ["f16", "q4_k_m"], first_conversion = "f16")
        right = estimate(quantization_methods = ["f16", "q4_k_m"], first_conversion = "bf16")
        assert (right - wrong) == n * 2
        assert round((right - wrong) / GB, 1) == 15.3

    def test_the_call_site_resolves_it(self, monkeypatch):
        """Driven, so a preflight left on its "f16" default fails here."""
        import torch

        seen = []
        monkeypatch.setattr(
            S,
            "_preflight_gguf_disk",
            lambda **kwargs: (
                seen.append(kwargs.get("model_dtype")) or (kwargs["save_directory"], True)
            ),
        )
        with contextlib.suppress(Exception):
            S.unsloth_save_pretrained_gguf(
                self._Model(self._Config(torch.bfloat16)),
                "model",
                # Any object gets past the "GGUF needs a tokenizer" check and dies well after the preflight, which is
                tokenizer = object(),
                quantization_method = ["f16", "q4_k_m"],
            )
        assert seen == ["bf16"], "the preflight has to be told what the exporter will use"


class TestThePrewarmedCacheIsChargedToItsOwnFilesystem:
    """`save_directory` is a mount; `~/.cache` and the `_gguf` sibling are not.

    Then the cached base model is written to the sibling's filesystem, and
    charging it to the checkpoint's both drops a pre-warm that had room and
    lets the sibling check accept `need_sibling` alone on a disk the base is
    about to land on.

    Numbers: a 60GB checkpoint, a 40GB sibling, a 60GB cache copy.
    """

    CHECKPOINT = 60 * GB
    SIBLING = 40 * GB
    CACHE = 60 * GB

    @pytest.fixture
    def split(self, monkeypatch):
        # The cache is looked up through `_hub_cache_directory`, whose answer is a real path, so it is the DEFAULT of
        state = {"here": 1000 * GB, "there": 1000 * GB, "cache_device": 2}
        devices = {"model": 1, "model_gguf": 2, "work": 2}

        def estimate(**kwargs):
            total = 0
            if kwargs.get("needs_merge", True):
                total += self.CHECKPOINT
            if kwargs.get("base_cache_copy", False):
                total += self.CACHE
            if kwargs.get("quantization_methods"):
                total += self.SIBLING
            return total

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", estimate)
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: state["there"] if str(path).endswith("_gguf") else state["here"],
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: "work")
        monkeypatch.setattr(
            S, "_filesystem_id", lambda path: devices.get(str(path), state["cache_device"])
        )
        monkeypatch.setattr(S, "_fallback_checkpoint_extra_bytes", lambda model: 0)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self):
        return S._preflight_gguf_disk(_FakeModel(), "model", "q4_k_m", first_conversion = "bf16")

    def test_a_sibling_that_cannot_also_hold_the_cache_drops_the_prewarm(self, split, capsys):
        """40GB of GGUF fits in 60GB; 40GB plus a 60GB base does not."""
        split.update(there = 60 * GB)
        assert self._preflight() == ("model", False)
        assert "Skipping the Hugging Face cache pre-warm" in capsys.readouterr().out

    def test_a_sibling_with_room_for_both_keeps_it(self, split):
        split.update(there = 120 * GB)
        assert self._preflight() == ("model", True)

    def test_the_checkpoint_is_not_charged_for_bytes_written_elsewhere(self, split):
        """60GB of checkpoint on a 63.2GB mount, with the cache on the other disk."""
        split.update(here = math.ceil(self.CHECKPOINT / 0.95), there = 1000 * GB)
        assert self._preflight() == ("model", True)

    def test_a_cache_on_this_filesystem_is_still_charged_here(self, split):
        """The premise reversed: nothing about the old accounting was wrong there."""
        split.update(cache_device = 1, here = math.ceil(self.CHECKPOINT / 0.95), there = 1000 * GB)
        assert self._preflight() == ("model", False)

    def test_an_unresolvable_cache_is_charged_where_it_always_was(self, split, monkeypatch):
        monkeypatch.setattr(S, "_hub_cache_directory", lambda: None)
        split.update(here = math.ceil(self.CHECKPOINT / 0.95), there = 1000 * GB)
        assert self._preflight() == ("model", False)

    def test_the_resolver_follows_the_same_cache_the_prewarm_downloads_into(self):
        import inspect

        prewarm = inspect.getsource(S._prewarm_base_model_hub_cache)
        assert "from unsloth_zoo.hf_cache import _active_caches" in prewarm
        assert "_active_caches" in inspect.getsource(S._hub_cache_directory)


class TestAnUnsupportedBF16IsNormalizedBeforeEstimating:
    """`save_to_gguf` drops a bf16 initial conversion to f16 on a T4.

    It does so AFTER resolving one, so the preflight has to do the same after
    both of its branches. `_gguf_source_dtype` covers only the dtype the
    preflight is TOLD; it cannot cover a `first_conversion` the caller passed,
    nor the single direct-convert method `_choose_first_conversion` hands back
    unchanged.

    The cost is a whole checkpoint, because the estimate omits an output that
    EQUALS the initial conversion: `["bf16"]` at "bf16" is priced as one
    16-bit file, and the export writes an f16 intermediate AND a bf16 output.
    """

    N = 8_190_735_360
    BITS = {"f32": 32.0, "f16": 16.0, "bf16": 16.0, "q4_k_m": 4.9}

    @classmethod
    def _estimate(
        cls,
        model = None,
        quantization_methods = (),
        first_conversion = "f16",
        needs_merge = True,
        n_parameters = None,
        base_cache_copy = False,
    ):
        total = cls.N * 2 if needs_merge else 0
        total += int(cls.N * cls.BITS[first_conversion] / 8)
        for method in dict.fromkeys(quantization_methods):
            if method != first_conversion:
                total += int(cls.N * cls.BITS[method] / 8)
        return total

    @pytest.fixture
    def sized(self, monkeypatch):
        """Records the `first_conversion` the preflight prices the export at."""
        asked = []

        def estimate(**kwargs):
            if kwargs.get("quantization_methods") and kwargs.get("needs_merge"):
                # Once for `need` and once for `need_with_cache`;
                # deduped so the assertions below are about the NAME and not the count.
                if kwargs["first_conversion"] not in asked:
                    asked.append(kwargs["first_conversion"])
            return self._estimate(**kwargs)

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", estimate)
        monkeypatch.setattr(S, "free_bytes", lambda path: 1000 * GB)
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_fallback_checkpoint_extra_bytes", lambda model: 0)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        return asked

    def _run(self, first_conversion):
        S._preflight_gguf_disk(
            _FakeModel(),
            "model",
            "bf16",
            first_conversion = first_conversion,
            model_dtype = "bf16",
        )

    @pytest.mark.parametrize("first_conversion", ["bf16", None])
    def test_hardware_without_bf16_is_priced_at_f16(self, sized, monkeypatch, first_conversion):
        """None goes through `_choose_first_conversion`, which returns "bf16" too."""
        monkeypatch.setattr(S.torch.cuda, "is_bf16_supported", lambda: False)
        self._run(first_conversion)
        assert sized == ["f16"]

    @pytest.mark.parametrize("first_conversion", ["bf16", None])
    def test_hardware_with_bf16_is_left_alone(self, sized, monkeypatch, first_conversion):
        monkeypatch.setattr(S.torch.cuda, "is_bf16_supported", lambda: True)
        self._run(first_conversion)
        assert sized == ["bf16"]

    def test_an_unanswerable_probe_takes_the_wider_reading(self, sized, monkeypatch):
        def boom():
            raise RuntimeError("no CUDA here")

        monkeypatch.setattr(S.torch.cuda, "is_bf16_supported", boom)
        self._run("bf16")
        assert sized == ["f16"]

    def test_the_undercount_is_a_whole_checkpoint(self):
        """What the wrong name costs, priced with the transcribed estimator."""
        wrong = self._estimate(quantization_methods = ["bf16"], first_conversion = "bf16")
        right = self._estimate(quantization_methods = ["bf16"], first_conversion = "f16")
        assert (right - wrong) == self.N * 2
        assert round(wrong / GB, 1) == 30.5
        assert round(right / GB, 1) == 45.8
        assert round((right - wrong) / GB, 1) == 15.3

    def test_the_exporter_still_does_the_drop_this_mirrors(self):
        import inspect

        source = inspect.getsource(S.save_to_gguf)
        assert 'first_conversion == "bf16" and not torch.cuda.is_bf16_supported()' in source
        assert 'first_conversion = "f16"' in source


class TestTheCacheIsChargedOnTheConversionFilesystem:
    """A third filesystem: the output on an external drive, the CWD and the
    Hugging Face cache together on the machine's own disk.

    The pre-warm downloads the base there and leaves it there for the rest of
    the export, so the cached base and the intermediate conversion are on that
    disk together. `cache_here` and `cache_sibling` only ever place the cache
    on the checkpoint's filesystem or the `_gguf` sibling's, and this is
    neither, so nothing charged it.

    The pre-warmer's own gate does not cover it: it asks for two base copies
    free, and an `f32` conversion is two base copies on its own. Qwen3-8B, a
    15.3GB base and a 30.5GB conversion, on 38.1GB free: 38.1 clears the
    pre-warm's 30.5 threshold and the conversion-only 30.5 check, then the
    conversion writes 30.5 into the 22.8 the cached base left. 7.6GB short.
    """

    N = 8_190_735_360
    BASE = 2 * N  # the 16-bit base the pre-warm downloads:
    CONVERSION = 4 * N
    WORK = "/work"
    CACHE = "/home/u/.cache/huggingface"

    @pytest.fixture
    def state(self, monkeypatch):
        # The output and its `_gguf` sibling on device 1;
        # `cache_device` is a knob so the same scenario can put the cache on the conversion's disk, on the output's, or
        # nowhere resolvable.
        state = {"conversion_free": int(2.5 * self.BASE), "cache_device": 2}
        devices = {"model": 1, "model_gguf": 1, self.WORK: 2}

        def estimate(**kwargs):
            total = self.BASE if kwargs.get("needs_merge", True) else 0
            if kwargs.get("base_cache_copy", False):
                total += self.BASE
            total += self.CONVERSION
            return total

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", estimate)
        monkeypatch.setattr(
            S,
            "_filesystem_id",
            lambda path: devices.get(str(path), state["cache_device"]),
        )
        monkeypatch.setattr(
            S,
            "free_bytes",
            lambda path: state["conversion_free"] if str(path) == self.WORK else 1000 * GB,
        )
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_fallback_checkpoint_extra_bytes", lambda model: 0)
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: self.WORK)
        monkeypatch.setattr(S, "_hub_cache_directory", lambda: self.CACHE)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self):
        return S._preflight_gguf_disk(
            _FakeModel(), "model", "f32", first_conversion = "f32", needs_merge = True
        )

    def test_the_prewarms_own_threshold_lets_this_through(self, state):
        """Two base copies free, which is the gate `_prewarm...` applies."""
        assert state["conversion_free"] >= 2 * self.BASE
        assert round(state["conversion_free"] / GB, 1) == 38.1
        assert round(self.CONVERSION / GB, 1) == 30.5
        assert round(self.BASE / GB, 1) == 15.3
        assert round((self.BASE + self.CONVERSION - state["conversion_free"]) / GB, 1) == 7.6

    def test_a_cache_sharing_the_working_directory_drops_the_prewarm(self, state, capsys):
        assert self._preflight() == ("model", False)
        assert "Skipping the Hugging Face cache pre-warm" in capsys.readouterr().out

    def test_room_for_both_keeps_it(self, state):
        state.update(conversion_free = self.BASE + self.CONVERSION)
        assert self._preflight() == ("model", True)

    def test_a_cache_elsewhere_is_not_charged_here(self, state):
        """The premise reversed: only a cache on THIS disk costs the pre-warm."""
        state.update(cache_device = 3)
        assert self._preflight() == ("model", True)

    def test_an_unresolvable_cache_charges_nothing_new(self, state, monkeypatch):
        monkeypatch.setattr(S, "_hub_cache_directory", lambda: None)
        assert self._preflight() == ("model", True)

    def test_a_conversion_that_does_not_fit_at_all_still_refuses(self, state):
        """The raise comes first, so this can never soften a refusal."""
        state.update(conversion_free = self.CONVERSION - 1)
        with pytest.raises(RuntimeError, match = "written to the current working directory"):
            self._preflight()

    def test_no_prewarm_means_no_charge(self, state, monkeypatch):
        """Kaggle and Colab return before the pre-warm, so it costs nothing."""
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", True)
        assert self._preflight() == ("model", True)

    def test_the_message_names_the_filesystem_it_measured(self, state, capsys):
        self._preflight()
        out = capsys.readouterr().out
        assert self.WORK in out
        assert "38.1GB free" in out
        assert "~30.5GB" in out


class TestACacheOnAnotherFilesystemIsNotChargedToTheOutputDisk:
    """One filesystem for the export, another for the Hugging Face cache.

    `HF_HOME` on a data volume is the ordinary layout on a machine with more
    than one disk, and then the cached base is never written to the disk the
    export lands on. Charging it there anyway drops the pre-warm on a disk
    that had room for everything written to it, and the next export downloads
    the whole base again - the exact re-download the pre-warm exists to stop.
    The split branch already asks this question; the single-filesystem one
    did not.

    Numbers: a 60GB checkpoint, 40GB of quants, a 60GB cached base, 120GB
    free. The export needs 100GB here and fits; only the fictitious cache
    copy takes it to 160GB.
    """

    CHECKPOINT = 60 * GB
    QUANTS = 40 * GB
    CACHE = 60 * GB
    ELSEWHERE = "/hf_cache"

    @pytest.fixture
    def state(self, monkeypatch):
        state = {"free": 120 * GB, "cache_device": 2}
        devices = {"model": 1, "model_gguf": 1, "work": 1}

        def estimate(**kwargs):
            total = 0
            if kwargs.get("needs_merge", True):
                total += self.CHECKPOINT
            if kwargs.get("base_cache_copy", False):
                total += self.CACHE
            if kwargs.get("quantization_methods"):
                total += self.QUANTS
            return total

        monkeypatch.setattr(S, "estimate_gguf_export_bytes", estimate)
        monkeypatch.setattr(S, "free_bytes", lambda path: state["free"])
        monkeypatch.setattr(S, "kaggle_tmp_redirect", lambda *a, **k: ("model", None))
        monkeypatch.setattr(S, "_gguf_conversion_directory", lambda directory: "work")
        monkeypatch.setattr(
            S, "_filesystem_id", lambda path: devices.get(str(path), state["cache_device"])
        )
        monkeypatch.setattr(S, "_hub_cache_directory", lambda: self.ELSEWHERE)
        monkeypatch.setattr(S, "_fallback_checkpoint_extra_bytes", lambda model: 0)
        monkeypatch.setattr(S, "IS_KAGGLE_ENVIRONMENT", False)
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", False)
        monkeypatch.delenv("UNSLOTH_DISK_PREFLIGHT", raising = False)
        monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
        return state

    def _preflight(self, **kwargs):
        return S._preflight_gguf_disk(
            _FakeModel(), "model", "q4_k_m", first_conversion = "bf16", **kwargs
        )

    def test_the_export_is_not_split_across_filesystems(self, state):
        """The premise: this is the single-filesystem branch, not the split one."""
        assert S._on_separate_filesystems("model", "model_gguf") is False
        assert S._on_separate_filesystems(self.ELSEWHERE, "model") is True

    def test_a_cache_elsewhere_keeps_the_prewarm(self, state, capsys):
        """100GB of export on 120GB free; the 60GB cache is on another disk."""
        assert self._preflight() == ("model", True)
        assert "Skipping the Hugging Face cache pre-warm" not in capsys.readouterr().out

    def test_a_cache_on_this_filesystem_still_drops_it(self, state, capsys):
        """The premise reversed: nothing was wrong about the accounting there."""
        state.update(cache_device = 1)
        assert self._preflight() == ("model", False)
        assert "Skipping the Hugging Face cache pre-warm" in capsys.readouterr().out

    def test_room_for_both_keeps_it(self, state):
        state.update(cache_device = 1, free = 160 * GB)
        assert self._preflight() == ("model", True)

    def test_an_unresolvable_cache_is_charged_where_it_always_was(self, state, monkeypatch):
        monkeypatch.setattr(S, "_hub_cache_directory", lambda: None)
        assert self._preflight() == ("model", False)

    def test_no_prewarm_means_no_charge_either_way(self, state, monkeypatch):
        """Colab returns before the pre-warm, so the cache costs nothing."""
        monkeypatch.setattr(S, "IS_COLAB_ENVIRONMENT", True)
        state.update(cache_device = 1)
        assert self._preflight() == ("model", True)

    def test_an_export_that_does_not_fit_is_still_refused(self, state):
        """Only the pre-warm decision reads this figure, so no refusal moves."""
        state.update(free = 99 * GB)
        with pytest.raises(RuntimeError) as error:
            self._preflight()
        assert "100.0GB" in str(error.value)

    def test_a_reclaimed_merge_charges_the_cache_where_it_lands(self, state, monkeypatch):
        """The peak branch takes the same figure, so it needs the same answer.

        60GB merge phase against 40GB of quants peaks at 60GB, not 100GB.
        """
        monkeypatch.setattr(S, "_merge_reclamation_is_possible", lambda directory: True)
        state.update(free = 80 * GB)
        assert self._preflight(merge_is_disposable = True) == ("model", True)
        state.update(cache_device = 1)
        assert self._preflight(merge_is_disposable = True) == ("model", False)
