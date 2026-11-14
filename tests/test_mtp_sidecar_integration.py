# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_cached_mtp_sidecar_restores_tiny_model(tmp_path):
    for name in ("torch", "transformers", "unsloth_zoo"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name} is not installed")
    pytest.importorskip("transformers.models.glm4_moe")
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import ast, os, shutil, sys, types
        from importlib.machinery import ModuleSpec
        from pathlib import Path
        import torch
        from transformers import Glm4MoeConfig, Glm4MoeForCausalLM

        root, cache = map(Path, sys.argv[1:])
        package = types.ModuleType("unsloth")
        package.__path__ = [str(root / "unsloth")]
        package.__spec__ = ModuleSpec("unsloth", None, is_package = True)
        sys.modules["unsloth"] = package
        models = types.ModuleType("unsloth.models")
        models.__path__ = [str(root / "unsloth/models")]
        models.__spec__ = ModuleSpec("unsloth.models", None, is_package = True)
        sys.modules["unsloth.models"] = models
        from unsloth.models.speculative import get_mtp_head
        from unsloth_zoo.speculative.stitch import save_mtp_head

        config = Glm4MoeConfig(
            vocab_size = 97, hidden_size = 64, intermediate_size = 128,
            moe_intermediate_size = 32, num_attention_heads = 4, num_key_value_heads = 2,
            head_dim = 16, num_hidden_layers = 2, n_routed_experts = 4,
            num_experts_per_tok = 2, n_shared_experts = 1, first_k_dense_replace = 1,
            num_nextn_predict_layers = 1, partial_rotary_factor = 0.5,
        )
        original = Glm4MoeForCausalLM(config).eval()
        head = get_mtp_head(original, random_init = True)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.add_(0.01)
        expected = {key: value.detach().clone() for key, value in head.mtp_state_dict().items()}
        repo = cache / "models--org--adapter"
        revision = "a" * 40
        snapshot = repo / "snapshots" / revision
        saved = save_mtp_head(cache / "trained", head.mtp_state_dict(), head._unsloth_mtp_info)
        snapshot.mkdir(parents = True)
        shutil.copyfile(saved, snapshot / "mtp_head.safetensors")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revision)

        source = root / "unsloth/models/loader.py"
        tree = ast.parse(source.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_reattach_saved_mtp_head")
        namespace = {"os": os, "__package__": "unsloth.models"}
        exec(compile(ast.Module(body = [helper], type_ignores = []), str(source), "exec"), namespace)
        fresh = Glm4MoeForCausalLM(config).eval()
        namespace["_reattach_saved_mtp_head"](
            fresh, "org/adapter", cache_dir = str(cache), local_files_only = True
        )
        loaded = getattr(fresh, "_unsloth_mtp_head", None)
        assert loaded is not None
        assert all(torch.equal(loaded.mtp_state_dict()[key], value)
                   for key, value in expected.items())
    """)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "UNSLOTH_ZOO_DISABLE_GPU_INIT": "1",
        "UNSLOTH_IS_PRESENT": "1",
    }
    subprocess.run(
        [sys.executable, "-c", script, str(root), str(tmp_path)],
        env = env,
        check = True,
        timeout = 60,
    )
