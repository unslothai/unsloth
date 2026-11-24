"""UNSLOTH_BFLOAT16_MIXED_PRECISION is process wide: a float32 mixed precision full finetuning load sets it
and nothing resets it, so every later model generated under bfloat16 autocast. Each model now carries its
own answer, and the env var only decides for models Unsloth never loaded."""

import ast
import inspect
import os
from contextlib import nullcontext
from importlib.metadata import version as installed_version
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

ROOT = Path(__file__).parents[1]
VISION_PATH = ROOT / "unsloth" / "models" / "vision.py"
ENV = "UNSLOTH_BFLOAT16_MIXED_PRECISION"
STAMP = "_unsloth_bfloat16_mixed_precision"


def _load_function(name, namespace):
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body = [function], type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace[name]


def _fast_generate(autocasts):
    def autocast(**kwargs):
        autocasts.append(kwargs)
        return nullcontext()

    fake_torch = SimpleNamespace(
        Tensor = type("FakeTensor", (), {"shape": (1, 3)}),
        bfloat16 = "bfloat16",
        float16 = "float16",
        inference_mode = nullcontext,
        autocast = autocast,
    )
    namespace = {
        "torch": fake_torch,
        "os": os,
        "inspect": inspect,
        "FastBaseModel": SimpleNamespace(for_inference = lambda model: None),
        "dtype_from_config": lambda config: "float16",
        "_get_dtype": lambda dtype: dtype,
        "_get_inference_mode_context_manager": lambda model: nullcontext(),
        "_lora_fold_generate": lambda model, args, kwargs: nullcontext(),
        "_unsloth_generate_accepts_kwarg": lambda model, name: False,
        "NUM_LOGITS_TO_KEEP": {"LlamaForCausalLM": None},
        "DEVICE_TYPE_TORCH": "cuda",
        "Version": Version,
        "transformers_version": installed_version("transformers"),
        # The FlashAttention fallback enters the autocast and returns _old_generate at once.
        "_uses_flash_attention_for_generation": lambda config: True,
        "_clear_generation_caches": lambda model: None,
        "_is_text_seq2seq_config": lambda config: False,
    }
    return _load_function("unsloth_base_fast_generate", namespace), fake_torch


def _autocast_dtype(stamp, env, monkeypatch):
    monkeypatch.setenv(ENV, env)
    autocasts = []
    fast_generate, fake_torch = _fast_generate(autocasts)

    class Model:
        config = SimpleNamespace(architectures = ["LlamaForCausalLM"], eos_token_id = 2)
        _unsloth_forced_float32 = False

        def forward(self, input_ids = None):
            return input_ids

        def named_modules(self):
            return []

        def _old_generate(self, *args, **kwargs):
            return "generated"

    model = Model()
    if stamp is not None:
        setattr(model, STAMP, stamp)
    assert fast_generate(model, input_ids = fake_torch.Tensor()) == "generated"
    assert len(autocasts) == 1
    return autocasts[0]["dtype"]


@pytest.mark.parametrize(
    "stamp, env, expected",
    [
        # A float16 LoRA model loaded after a float32 mixed precision full finetuning load.
        (False, "1", "float16"),
        # The float32 mixed precision model itself, after a later load wrote "0".
        (True, "0", "bfloat16"),
        # Stamp and env agree.
        (False, "0", "float16"),
        (True, "1", "bfloat16"),
        # Never loaded through Unsloth: the env var still decides.
        (None, "1", "bfloat16"),
        (None, "0", "float16"),
    ],
)
def test_generate_autocast_follows_the_model_not_the_last_load(stamp, env, expected, monkeypatch):
    assert _autocast_dtype(stamp, env, monkeypatch) == expected


def _from_pretrained_source():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastBaseModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    return ast.get_source_segment(VISION_PATH.read_text(encoding = "utf-8"), fn)


def test_from_pretrained_stamps_the_model_it_loads():
    src = _from_pretrained_source()
    assert "_mark_bfloat16_mixed_precision(model, bfloat16_mixed_precision)" in src
    env_write = src.index(f'os.environ["{ENV}"] = "1"')
    flag_set = src.index("bfloat16_mixed_precision = True", env_write)
    # The stamp is set right where the env var is, so the two cannot disagree for this load.
    assert flag_set - env_write < 200
    assert src.index("bfloat16_mixed_precision = False") < env_write


def test_mark_helper_sets_a_bool_and_tolerates_frozen_objects():
    utils_src = (ROOT / "unsloth" / "models" / "_utils.py").read_text(encoding = "utf-8")
    tree = ast.parse(utils_src)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_mark_bfloat16_mixed_precision"
    )
    ns = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), "_utils.py", "exec"), ns)
    mark = ns["_mark_bfloat16_mixed_precision"]
    m = SimpleNamespace()
    assert mark(m, 1) is m and getattr(m, STAMP) is True
    assert getattr(mark(SimpleNamespace(), None), STAMP) is False
    frozen = object()
    assert mark(frozen, True) is frozen
