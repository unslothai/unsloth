import ast
import contextlib
import os
import shutil
from pathlib import Path


def _load_patch_unsloth_gguf_save():
    # Extract the nested patch_unsloth_gguf_save context manager without importing
    # unsloth (which needs unsloth_zoo / a GPU). It only relies on contextlib, os
    # and shutil, so it can be exec'd in isolation.
    source = Path(__file__).parents[2] / "unsloth" / "models" / "sentence_transformer.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "patch_unsloth_gguf_save"
    ]
    assert len(funcs) == 1, funcs
    namespace = {"contextlib": contextlib, "os": os, "shutil": shutil}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["patch_unsloth_gguf_save"]


def test_patch_unsloth_gguf_save_protects_only_the_save_directory(tmp_path):
    # The guard must suppress deletion of the save directory (and its subtree) while
    # still letting unrelated temporary directories be cleaned up, and it must restore
    # shutil.rmtree on exit.
    patch_unsloth_gguf_save = _load_patch_unsloth_gguf_save()

    save_dir = tmp_path / "saved_model"
    save_dir.mkdir()
    (save_dir / "model.safetensors").write_text("weights")
    subdir = save_dir / "0_Transformer"
    subdir.mkdir()
    unrelated = tmp_path / "_unsloth_temporary_saved_buffers"
    unrelated.mkdir()
    (unrelated / "scratch.bin").write_text("temp")

    original_rmtree = shutil.rmtree
    with patch_unsloth_gguf_save(str(save_dir)):
        shutil.rmtree(str(save_dir))  # protected: no-op
        shutil.rmtree(str(subdir))  # inside the protected tree: no-op
        shutil.rmtree(str(unrelated))  # unrelated temp dir: still deleted

        assert save_dir.exists(), "save directory must survive the guard"
        assert subdir.exists(), "a subdirectory of the save directory must survive"
        assert not unrelated.exists(), "unrelated temp directories must still be cleaned up"

    assert shutil.rmtree is original_rmtree, "shutil.rmtree must be restored on exit"
