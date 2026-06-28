"""CPU-only AST checks on the export API surface in save.py / _compressed_quantize.py.

These catch wiring regressions - a save_method that stops dispatching, a public method that
stops being attached to the model, or an export subprocess that becomes shell-unsafe - without
importing torch or touching a GPU. Pure `ast`, so they run in milliseconds on CPU-only CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

UNSLOTH = Path(__file__).resolve().parents[2] / "unsloth"
SAVE_PY = UNSLOTH / "save.py"
QUANT_PY = UNSLOTH / "_compressed_quantize.py"

SAVE_SRC = SAVE_PY.read_text(encoding = "utf-8")
SAVE_TREE = ast.parse(SAVE_SRC, filename = str(SAVE_PY))

# Every merged-save entry point that must route compressed (FP8/FP4/INT) save_methods.
MERGED_SAVERS = (
    "unsloth_save_pretrained_merged",
    "unsloth_push_to_hub_merged",
    "unsloth_generic_save_pretrained_merged",
    "unsloth_generic_push_to_hub_merged",
)
# Public export methods that must be attached to the model in patch_saving_functions.
PUBLIC_EXPORT_METHODS = (
    "save_pretrained_merged",
    "push_to_hub_merged",
    "save_pretrained_gguf",
    "push_to_hub_gguf",
    "save_pretrained_torchao",
    "save_pretrained_ggml",
    "push_to_hub_ggml",
)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {SAVE_PY.name}")


def _called_names(node):
    names = set()
    for c in ast.walk(node):
        if isinstance(c, ast.Call):
            if isinstance(c.func, ast.Name):
                names.add(c.func.id)
            elif isinstance(c.func, ast.Attribute):
                names.add(c.func.attr)
    return names


def _subprocess_calls(node):
    out = []
    for c in ast.walk(node):
        if (
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "subprocess"
            and c.func.attr in ("Popen", "run", "check_call", "check_output")
        ):
            out.append(c)
    return out


def _list_var_elts(func_node, var_name):
    for child in ast.walk(func_node):
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.List):
            if any(isinstance(t, ast.Name) and t.id == var_name for t in child.targets):
                return child.value.elts
    return None


def test_all_merged_savers_dispatch_compressed_export():
    for fn in MERGED_SAVERS:
        called = _called_names(_func(SAVE_TREE, fn))
        assert "_normalize_compressed_method" in called, f"{fn} must normalize the save_method"
        assert (
            "_unsloth_save_compressed_tensors" in called
        ), f"{fn} must dispatch the compressed export"


def test_public_export_methods_are_attached():
    # Collect every `<obj>.<attr> = ...` target name in patch_saving_functions.
    patch_fn = _func(SAVE_TREE, "patch_saving_functions")
    attached = {
        t.attr
        for n in ast.walk(patch_fn)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute)
    }
    for method in PUBLIC_EXPORT_METHODS:
        assert method in attached, f"patch_saving_functions must attach model.{method}"


def test_gguf_savers_have_lora_branch():
    for fn in ("unsloth_save_pretrained_gguf", "unsloth_push_to_hub_gguf"):
        called = _called_names(_func(SAVE_TREE, fn))
        assert (
            "_unsloth_save_lora_gguf" in called
        ), f"{fn} must support save_method='lora' -> _unsloth_save_lora_gguf"


def test_torchao_dispatches_both_ptq_and_qat():
    called = _called_names(_func(SAVE_TREE, "unsloth_save_pretrained_torchao"))
    assert "_unsloth_save_torchao_with_given_config" in called, "torchao PTQ path missing"
    assert "_unsloth_save_torchao_with_attached_config" in called, "torchao QAT path missing"


def test_export_subprocesses_are_shell_safe():
    # The compressed-quantize and LoRA->GGUF subprocesses must run argv lists led by
    # sys.executable, never shell=True (a crafted save path must not inject a shell command).
    for fn in ("_unsloth_save_compressed_tensors", "_unsloth_save_lora_gguf"):
        node = _func(SAVE_TREE, fn)
        calls = _subprocess_calls(node)
        assert calls, f"{fn} should invoke a subprocess for the export"
        checked_argv = False
        for call in calls:
            shell_true = [
                kw
                for kw in call.keywords
                if kw.arg == "shell"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ]
            assert not shell_true, f"{fn}: subprocess must not use shell=True"
            if not call.args:
                continue
            argv = call.args[0]
            elts = (
                argv.elts
                if isinstance(argv, ast.List)
                else (_list_var_elts(node, argv.id) if isinstance(argv, ast.Name) else None)
            )
            if elts is None:
                continue
            first = elts[0]
            assert (
                isinstance(first, ast.Attribute) and first.attr == "executable"
            ), f"{fn}: subprocess argv[0] must be sys.executable, not a shell string"
            checked_argv = True
        assert checked_argv, f"{fn}: could not verify an argv-list subprocess invocation"


def test_compressed_export_propagates_variant():
    # save_pretrained_merged(..., save_method="fp8", variant="foo") must not leave the variant on
    # the intermediate 16bit merge - the converter subprocess reloads that dir with default weight
    # filenames, so variant-named shards there would break the reload after the merge. The variant
    # is popped out of the merge kwargs and forwarded via --variant, which applies it to the final
    # compressed checkpoint. Guards this subprocess-bridged contract without a GPU.
    helper_src = ast.get_source_segment(
        SAVE_SRC, _func(SAVE_TREE, "_unsloth_save_compressed_tensors")
    )
    assert (
        'merge_kwargs.pop("variant"' in helper_src
    ), "compressed export must pop variant out of the intermediate 16bit merge kwargs"
    assert (
        '"--variant"' in helper_src
    ), "compressed export must forward the variant to the converter"
    quant_src = QUANT_PY.read_text(encoding = "utf-8")
    assert '"--variant"' in quant_src, "the converter runner must accept --variant"
    assert (
        "save_compressed" in quant_src and "variant" in quant_src
    ), "the converter must apply the variant to the final compressed save_pretrained"


def test_compressed_quantize_runner_parses():
    # The standalone runner is invoked by path in a subprocess; make sure it stays importable
    # (valid syntax) so a typo there is caught without launching the subprocess.
    ast.parse(QUANT_PY.read_text(encoding = "utf-8"), filename = str(QUANT_PY))
