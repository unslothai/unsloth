# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""`save_lora` must be attached with or without a vLLM engine.

`patch_peft_fast_inference` set it only inside `if vllm_engine is not None`,
but unsloth_zoo's `save_lora` is `save_pretrained` over the lora_A/lora_B keys
and never touches the engine. So `LFM2.5_(1.2B)-GRPO`, which loads with
`fast_inference = False` and saves at the end, got `AttributeError:
'Lfm2ForCausalLM' object has no attribute 'save_lora'`, naming neither vLLM nor
the flag that caused it. `load_lora` stays gated: it copies into vLLM's own
adapter buffers.

Source-level, because importing the module pulls the whole model stack.
"""

import ast
import pathlib

import pytest

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _patch_function():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "patch_peft_fast_inference":
            return node
    pytest.fail("patch_peft_fast_inference has moved or been renamed")


def _assigned_attributes(scope):
    """Every `model.<name> = ...` target inside `scope`."""
    found = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == "model":
                    found.add(target.attr)
    return found


def _engine_guard(function):
    """The `if vllm_engine is not None:` block."""
    for node in function.body:
        if isinstance(node, ast.If) and "vllm_engine" in ast.unparse(node.test):
            return node
    pytest.fail("the vllm_engine guard has moved or been renamed")


def _outside_the_guard(function):
    """The function body with the `if vllm_engine is not None:` block removed.

    Not `all - guard`: set subtraction drops a name assigned in BOTH places,
    which is exactly `save_lora` now.
    """
    return ast.Module(
        body = [
            node
            for node in function.body
            if not (isinstance(node, ast.If) and "vllm_engine" in ast.unparse(node.test))
        ],
        type_ignores = [],
    )


def test_save_lora_is_set_outside_the_engine_guard():
    """The bug: with no engine the attribute was never set at all.

    Asserted as "set outside the guard" rather than "not set inside it", because
    a model that HAS an engine keeps the Zoo helper it has always had.
    """
    function = _patch_function()
    outside = _assigned_attributes(_outside_the_guard(function))
    assert "save_lora" in outside, (
        "save_lora is set only when a vLLM engine exists, so fast_inference=False "
        "leaves the model without it"
    )


def test_the_engine_path_keeps_the_zoo_helper():
    """Saving under vLLM is read back by vLLM's own LoRA loader, so what that
    file carries is not changed here."""
    guard = ast.unparse(_engine_guard(_patch_function()))
    assert "from unsloth_zoo.vllm_utils import save_lora" in guard
    assert "functools.partial(save_lora, model)" in guard


def test_save_lora_is_still_set_somewhere_in_the_function():
    function = _patch_function()
    assert "save_lora" in _assigned_attributes(function), "save_lora is no longer set at all"


def test_load_lora_stays_behind_the_engine_guard():
    """It writes into vLLM's adapter tensors, so it needs one."""
    function = _patch_function()
    guard = _engine_guard(function)
    assert "load_lora" in _assigned_attributes(guard)
    outside = _assigned_attributes(function) - _assigned_attributes(guard)
    assert "load_lora" not in outside


def test_fast_generate_stays_behind_the_engine_guard():
    """The other engine-only attributes must not have been loosened too."""
    function = _patch_function()
    guard = _assigned_attributes(_engine_guard(function))
    for name in ("vllm_engine", "fast_generate", "fast_generate_batches"):
        assert name in guard, f"{name} escaped the engine guard"


def test_an_existing_save_lora_is_not_replaced():
    """Set only when absent, so a model that already carries one keeps it."""
    function = _patch_function()
    source = ast.unparse(function)
    assert 'hasattr(model, "save_lora")' in source or "hasattr(model, 'save_lora')" in source


def test_a_missing_zoo_helper_does_not_break_loading():
    """Older unsloth_zoo has no `save_lora`; that must not break loading."""
    function = _patch_function()
    handlers = [node for node in ast.walk(function) if isinstance(node, ast.Try)]
    assert handlers, "the save_lora import is unguarded, so an older zoo raises on load"
    guarded = any("save_lora" in ast.unparse(node) for node in handlers)
    assert guarded, "the guarded import does not cover save_lora"


def test_a_missing_zoo_helper_cannot_break_the_engineless_path():
    """The engineless attach must not depend on the Zoo at all.

    That import lives inside the engine guard now, so an older unsloth_zoo can
    only ever cost a vLLM run its `save_lora`, never a plain one.
    """
    assert "unsloth_zoo" not in ast.unparse(_outside_the_guard(_patch_function()))


def _peft_case(**lora_kwargs):
    """A tiny PEFT model, and what PEFT itself would write for it."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", dtype = torch.float16
        ),
        peft.LoraConfig(r = 8, target_modules = ["q_proj", "v_proj"], **lora_kwargs),
    )
    return model


def _saved_keys(model, save, tmp_path, name):
    safetensors = pytest.importorskip("safetensors.torch")
    directory = tmp_path / name
    save(model, str(directory))
    return set(safetensors.load_file(str(directory / "adapter_model.safetensors")))


@pytest.mark.parametrize(
    "lora_kwargs",
    [
        {},
        {"modules_to_save": ["embed_tokens", "lm_head"]},
        {"use_dora": True},
        {"use_dora": True, "modules_to_save": ["lm_head"]},
    ],
    ids = ["plain", "modules_to_save", "dora", "dora_and_modules_to_save"],
)
def test_the_adapter_save_keeps_everything_peft_would_keep(tmp_path, lora_kwargs):
    """The Zoo helper filters to `.lora_A.`/`.lora_B.` before PEFT selects, so
    PEFT raises `KeyError: modules_to_save.default.weight` and a DoRA run loses
    its `lora_magnitude_vector`. Unsloth adds `embed_tokens`/`lm_head` to
    `modules_to_save` by itself once new tokens are trained, so both are
    reachable with no vLLM in sight.
    """
    from unsloth.models._utils import save_lora_adapter

    reference = _saved_keys(_peft_case(**lora_kwargs), lambda m, d: m.save_pretrained(d), tmp_path, "peft")
    ours = _saved_keys(_peft_case(**lora_kwargs), save_lora_adapter, tmp_path, "ours")
    assert ours == reference, f"missing {sorted(reference - ours)}, extra {sorted(ours - reference)}"


def test_the_saved_adapter_still_loads_back(tmp_path):
    """Key equality is not enough; PEFT has to accept the file."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from unsloth.models._utils import save_lora_adapter

    model = _peft_case(use_dora = True, modules_to_save = ["lm_head"])
    directory = tmp_path / "roundtrip"
    save_lora_adapter(model, str(directory))
    base = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", dtype = torch.float16
    )
    reloaded = peft.PeftModel.from_pretrained(base, str(directory))
    assert reloaded is not None


def test_the_adapter_is_cast_to_the_embedding_dtype(tmp_path):
    """Which is the only thing the Zoo helper does beyond `save_pretrained`."""
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from unsloth.models._utils import save_lora_adapter

    model = _peft_case()
    directory = tmp_path / "dtype"
    save_lora_adapter(model, str(directory))
    saved = safetensors.load_file(str(directory / "adapter_model.safetensors"))
    assert {v.dtype for v in saved.values()} == {torch.float16}
