# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Token counts and spatial sizes that change with the resolution / frame count stay out of the Triton compile key
(do_not_specialize), so a render at a resolution whose sizes are not multiples of 16 does not recompile every kernel."""

import ast
import pathlib

import pytest

INF = pathlib.Path(__file__).resolve().parents[1] / "core" / "inference"

CASES = {
    "diffusion_int8_gemm.py": {"i8mm_dq": ("M",), "i8mm_dq_tma": ("M",)},
    "diffusion_convrot_quant.py": {
        "rotq_i8": ("M",),
        "rotq_h16": ("M",),
        "rotq_h16x": ("M", "SEQ"),
    },
    "diffusion_flux2_rope.py": {"_rope_fwd": ("S",)},
    "diffusion_rocm_fused.py": {"_adaln_fwd": ("S",)},
    "diffusion_moss_fused.py": {"kernel": ("S",)},
    "diffusion_qwenimage21_eager.py": {"_norm_rope_kernel": ("n_seq",)},
    "video_minimax_h3_vae.py": {
        "_gn_partials": ("T", "HW", "W", "n_chunks"),
        "_gn_combine": ("n_chunks",),
        "_gn_silu_pad": ("T", "H", "W", "To", "Ho", "Wo"),
        "_add_residual": ("P", "T", "H", "W"),
        "_qk_norm_rope": ("n_head_rows",),
    },
    "video_minimax_h3_qknorm.py": {"_qk_norm_rope_kernel": ("n_rows",)},
}
VAE_SHAPE_ARGS = {
    "_gn_partials": ("T", "HW", "W", "n_chunks"),
    "_gn_combine": ("n_chunks",),
    "_gn_apply": ("T", "HW", "W"),
    "_softmax_rows": ("S",),
    "_dcae_up_add": ("F", "H", "W", "Fo"),
}


def _defs(path):
    return {
        n.name: n
        for n in ast.walk(ast.parse(path.read_text(encoding = "utf-8")))
        if isinstance(n, ast.FunctionDef)
    }


def _dns(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("jit"):
            for kw in dec.keywords:
                if kw.arg == "do_not_specialize":
                    return kw.value
    return None


@pytest.mark.parametrize("rel", sorted(CASES))
def test_resolution_sizes_not_specialized(rel):
    defs = _defs(INF / rel)
    for name, args in CASES[rel].items():
        node = _dns(defs[name])
        assert node is not None, name
        listed = {e.value for e in node.elts}
        params = {a.arg for a in defs[name].args.args}
        for a in args:
            assert a in params and a in listed, (rel, name, a)


def test_vae_shape_args_cover_group_norm_and_attention():
    src = (INF / "diffusion_vae_fused.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    table = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_SHAPE_ARGS"
    )
    defs = _defs(INF / "diffusion_vae_fused.py")
    for name, args in VAE_SHAPE_ARGS.items():
        assert tuple(table[name]) == args, name
        assert ast.unparse(_dns(defs[name])) == f"_SHAPE_ARGS['{name}']", name
