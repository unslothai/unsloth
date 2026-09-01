# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The diffusion runner must honour the GPU-layer split (#7574).

Unsloth used to drop a manual GPU-layers setting on the diffusion path and pin every layer
to GPU, so a GGUF larger than VRAM OOMed in cudaMalloc with no way out.

The pure helpers run directly; the wiring is checked at source level, since importing the
backend pulls in the whole studio stack.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"
ROUTE_PATH = REPO_ROOT / "studio" / "backend" / "routes" / "inference.py"
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


@pytest.fixture(scope = "module")
def llama_cpp():
    """Import the backend module directly; skip if the studio deps aren't installed."""
    backend = str(REPO_ROOT / "studio" / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    spec = importlib.util.spec_from_file_location("_llama_cpp_under_test", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # missing optional studio dep on a bare checkout
        pytest.skip(f"llama_cpp not importable here: {exc}")
    finally:
        # Do not leave studio/backend on sys.path:
        if sys.path and sys.path[0] == backend:
            sys.path.pop(0)
    # The dedupe comparators consult the Metal device, so on a Mac (and on the paravirtual macos runners) they would
    module._metal_device_is_paravirtual = lambda: False
    return module


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} missing")


def _body(name: str) -> str:
    return ast.get_source_segment(SRC, _function(name)) or ""




@pytest.mark.parametrize(
    ("mode", "layers", "expected"),
    [
        ("manual", 8, 8),
        ("manual", 0, 0),  # CPU-only is a real request, not "unset" Auto slider defers to the runner Unsloth mode
        ("manual", -1, None),
        ("auto", 8, None),
        ("auto", -1, None),
    ],
)
def test_effective_ngl(llama_cpp, mode, layers, expected):
    assert llama_cpp._diffusion_manual_ngl(mode, layers) == expected


def test_zero_layers_is_not_swallowed_as_falsy(llama_cpp):
    """The exact case in the report: GPU layers set to 0 must reach the child."""
    assert llama_cpp._diffusion_manual_ngl("manual", 0) == 0




def test_shim_without_ngl_is_detected(llama_cpp, tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text('ap.add_argument("--maxtok", type=int)\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is False


def test_shim_with_ngl_is_detected(llama_cpp, tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text('ap.add_argument("--ngl", type=int)\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is True


def test_missing_shim_file_does_not_raise(llama_cpp, tmp_path):
    assert llama_cpp._shim_supports_ngl(["python", str(tmp_path / "gone.py")]) is False




def test_diffusion_server_accepts_the_layer_split():
    fn = _function("_start_diffusion_server")
    names = {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}
    assert {"gpu_memory_mode", "gpu_layers"} <= names


def test_diffusion_server_forwards_ngl_and_gates_it_on_shim_support():
    body = _body("_start_diffusion_server")
    assert '"--ngl"' in body
    assert "_shim_supports_ngl" in body


def test_zero_layers_masks_the_child_devices(llama_cpp):
    """gpu_layers=0 must CUDA-mask the child, else _gpu_offload_active=False lies to the
    training VRAM coordinator and a GPU-resident runner survives into a training run.
    Behavioural, not a source-text match: what matters is the token the child gets."""
    arg = llama_cpp.LlamaCppBackend._diffusion_gpu_arg
    assert arg([3, 1], force_cpu = True) == ""
    assert arg(None, force_cpu = True) == ""


def test_explicit_pick_still_wins_when_layers_are_not_zero(llama_cpp):
    """force_cpu is the only thing above the picker. A host whose GPU torch cannot see
    (Metal, Vulkan, Windows-HIP, Intel XPU) still has to honour an explicit pick."""
    arg = llama_cpp.LlamaCppBackend._diffusion_gpu_arg
    assert arg([3, 1], cpu_only = True) == "1"
    assert arg([3, 1]) == "1"


def test_no_gpu_and_no_pick_masks_the_child(llama_cpp):
    assert llama_cpp.LlamaCppBackend._diffusion_gpu_arg(None, cpu_only = True) == ""


def test_diffusion_load_passes_the_users_split_through():
    call = next(
        node
        for node in ast.walk(_function("load_model"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_start_diffusion_server"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    for name in ("gpu_memory_mode", "gpu_layers"):
        assert isinstance(keywords.get(name), ast.Name)
        assert keywords[name].id == name


def test_diffusion_no_longer_hardcodes_auto_over_the_users_choice():
    body = _body("_start_diffusion_server")
    assert 'self._gpu_memory_mode = "auto"' not in body
    assert "self._gpu_layers = -1" not in body




def _loaded_diffusion(llama_cpp, *, recorded_layers, requested_ngl):
    """A backend that looks like a healthy diffusion runner, for the dedup guards."""
    b = llama_cpp.LlamaCppBackend()
    b._process, b._healthy, b._is_diffusion = object(), True, True
    b._model_identifier = "unsloth/DiffusionGemma-GGUF"
    b._hf_variant = b._gguf_path = b._cache_type_kv = None
    b._requested_n_ctx = 4096
    b._tensor_parallel = b._layer_preserves_tensor_intent = False
    b._gpu_layers = recorded_layers
    b._gpu_memory_mode = "auto" if recorded_layers < 0 else "manual"
    b._diffusion_requested_ngl = requested_ngl
    b._gpu_ids = b._requested_gpu_ids = [0]
    b._requested_spec_mode = "auto"
    b._spec_fallback_reason = b._speculative_type = b._spec_draft_n_max = None
    b._chat_template_override = b._mtp_draft_path = b._extra_args = None
    # Dropped-split rows model "the shim stayed old";
    b.diffusion_split_supported = lambda: False
    return b


def _in_target_state(llama_cpp, b, *, mode, layers):
    return b.adopt_load_intent_if_matched(
        llama_cpp.GgufLoadIntent(
            model_identifier = "unsloth/DiffusionGemma-GGUF",
            n_ctx = 4096,
            gpu_memory_mode = mode,
            gpu_layers = layers,
            gpu_ids = [0],
        )
    )


@pytest.mark.parametrize(
    ("recorded", "requested_ngl", "mode", "layers", "expected"),
    [
        (-1, None, "auto", -1, True),  # auto -> auto inert manual preference must not loop a real split must reload
        (-1, None, "manual", -1, True),  # inert manual preference must not loop
        (-1, None, "manual", 8, False),
        (8, 8, "manual", 8, True),
        (8, 8, "manual", 4, False),
        (8, 8, "auto", -1, False),
        (0, 0, "manual", 0, True),
        # No --ngl: -1 runs but 20 was the ask; comparing on the ask stops a reload loop.
        (-1, 20, "manual", 20, True),
        (-1, 20, "manual", 8, False),
    ],
)
def test_backend_dedup_compares_the_requested_split(
    llama_cpp, recorded, requested_ngl, mode, layers, expected
):
    b = _loaded_diffusion(llama_cpp, recorded_layers = recorded, requested_ngl = requested_ngl)
    assert _in_target_state(llama_cpp, b, mode = mode, layers = layers) is expected


def test_the_dedupe_compares_the_requested_split_through_the_paravirtual_rewrite():
    """The single comparator now lives on the backend, so the diffusion split has to be
    judged on the normalized intent: a virtualised Metal device launches the CPU-pinned
    rewrite, and comparing the raw ask against it would reload a healthy server forever."""
    bodies = {
        node.name: (ast.get_source_segment(SRC, node) or "")
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef)
        and node.name in ("adopt_load_intent_if_matched", "_runtime_matches_intent")
    }
    adopt = bodies["adopt_load_intent_if_matched"]
    assert "_metal_device_is_paravirtual()" in adopt
    assert "paravirtual_normalized_request(" in adopt
    # Normalized before the runtime comparison reads it, or the rewrite changes nothing.
    assert adopt.index("paravirtual_normalized_request(") < adopt.index("_runtime_matches_intent(")
    runtime = bodies["_runtime_matches_intent"]
    assert "_diffusion_manual_ngl(intent.gpu_memory_mode, intent.gpu_layers)" in runtime
    assert "self.diffusion_requested_ngl" in runtime


def test_requested_split_survives_a_shim_without_the_flag(llama_cpp):
    """gpu_layers reports what is running; diffusion_requested_ngl reports the ask."""
    b = _loaded_diffusion(llama_cpp, recorded_layers = -1, requested_ngl = 20)
    assert b.gpu_layers == -1
    assert b.diffusion_requested_ngl == 20




@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('ap.add_argument("--ngl", type=int)', True),
        ("ap.add_argument('--ngl', type=int)", True),  # quoting must not matter
        ('# someday: support "--ngl"', False),
        ('"""usage: --ngl N"""', False),
        ('ap.add_argument("--maxtok", type=int)', False),
    ],
)
def test_probe_reads_declarations_not_substrings(llama_cpp, tmp_path, source, expected):
    shim = tmp_path / "shim.py"
    shim.write_text(source + "\n", encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is expected


def test_probe_accepts_an_uppercase_extension(llama_cpp, tmp_path):
    """A Windows UNSLOTH_DG_SHIM override may be SHIM.PY; it must still be the file read."""
    shim = tmp_path / "SHIM.PY"
    shim.write_text('ap.add_argument("--ngl", type=int)\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is True


def test_probe_falls_back_to_a_substring_scan_on_unparseable_source(llama_cpp, tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text('ap.add_argument("--ngl"\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is True




# the probe must inspect the file that will be spawned, whatever its name ──
@pytest.mark.parametrize("name", ["shim", "shim.pyw", "SHIM.PY"])
def test_probe_keys_on_argv_shape_not_suffix(llama_cpp, tmp_path, name):
    """Any UNSLOTH_DG_SHIM file launches as-is, so the probe must answer for that exact
    file; an extensionless or .pyw override used to fall through to the package."""
    shim = tmp_path / name
    shim.write_text('ap.add_argument("--ngl", type=int)\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(shim)]) is True


def test_probe_does_not_mistake_the_module_form_for_a_file(llama_cpp, monkeypatch):
    """[python, -m, unsloth_zoo.diffusion_studio.shim] carries a module name, not a
    path; the probe must resolve the installed package, not stat the module string."""
    import importlib.util as ilu

    monkeypatch.setattr(ilu, "find_spec", lambda name: None)
    cmd = ["python", "-m", "unsloth_zoo.diffusion_studio.shim"]
    assert llama_cpp._shim_supports_ngl(cmd) is False  # unresolvable -> conservative




# the guard must mirror what the launcher will actually do ──
def test_split_supported_mirrors_the_launch_gate(llama_cpp, tmp_path, monkeypatch):
    b = llama_cpp.LlamaCppBackend()
    shim = tmp_path / "shim.py"

    shim.write_text('ap.add_argument("--ngl", type=int)\n', encoding = "utf-8")
    monkeypatch.setattr(
        b, "_find_diffusion_assets", lambda: (["python", str(shim)], "/bin/dg", None)
    )
    assert b.diffusion_split_supported() is True

    shim.write_text('ap.add_argument("--maxtok", type=int)\n', encoding = "utf-8")
    assert b.diffusion_split_supported() is False

    monkeypatch.setattr(b, "_find_diffusion_assets", lambda: None)
    assert b.diffusion_split_supported() is False  # no runner -> no split


def test_training_guard_mirrors_shim_support():
    """The zero-layer bypass and the split-scaled estimate are only valid when the
    launcher will actually emit --ngl; a dropped split runs GPU-resident."""
    route_src = ROUTE_PATH.read_text(encoding = "utf-8")
    route_tree = ast.parse(route_src)
    fn = next(
        n
        for n in ast.walk(route_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_guard_chat_load_against_training"
    )
    body = ast.get_source_segment(route_src, fn) or ""
    assert "diffusion_split_supported" in body
    assert body.index("diffusion_split_supported") < body.index("diffusion_ngl == 0")
    assert "_scale_diffusion_required_gb" in body




@pytest.mark.parametrize(
    ("required", "ngl", "n_layers", "expected"),
    [
        (15.0, 10, 30, 5.0),  # a third of the layers -> a third of the footprint all layers -> unchanged over-ask
        (15.0, 30, 30, 15.0),
        (15.0, 99, 30, 15.0),
        (15.0, 10, None, 15.0),
        (15.0, 10, 0, 15.0),
    ],
)
def test_positive_split_scales_the_guard_estimate(llama_cpp, required, ngl, n_layers, expected):
    assert llama_cpp._scale_diffusion_required_gb(required, ngl, n_layers) == pytest.approx(
        expected
    )


# a custom-named override answers for itself, not a sibling shim.py


# a custom-named override answers for itself, not a sibling shim.py ──
def test_probe_ignores_a_sibling_shim_next_to_a_custom_override(llama_cpp, tmp_path):
    """An override runs as-is; a capable sibling shim.py must not vouch for it, or the
    launch appends --ngl to a parser that exits on it."""
    override = tmp_path / "my_shim"
    override.write_text('ap.add_argument("--maxtok", type=int)\n', encoding = "utf-8")
    sibling = tmp_path / "shim.py"
    sibling.write_text('ap.add_argument("--ngl", type=int)\n', encoding = "utf-8")
    assert llama_cpp._shim_supports_ngl(["python", str(override)]) is False




def test_zoo_upgrade_reloads_a_dropped_split(llama_cpp):
    """manual/20 against an old shim launched with the default and deduped on the
    ask. Once the shim gains --ngl, the identical ask must reload to apply it."""
    b = _loaded_diffusion(llama_cpp, recorded_layers = -1, requested_ngl = 20)
    assert _in_target_state(llama_cpp, b, mode = "manual", layers = 20) is True  # shim still old zoo upgraded in this
    b.diffusion_split_supported = lambda: True
    assert _in_target_state(llama_cpp, b, mode = "manual", layers = 20) is False
    b2 = _loaded_diffusion(llama_cpp, recorded_layers = 20, requested_ngl = 20)
    b2.diffusion_split_supported = lambda: True
    assert _in_target_state(llama_cpp, b2, mode = "manual", layers = 20) is True




def test_response_models_expose_the_requested_split():
    """A refresh has no in-memory split left, so the wire has to carry the ask."""
    models_src = (REPO_ROOT / "studio" / "backend" / "models" / "inference.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(models_src)
    runtime = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "_InferenceRuntimeFields"
    )
    fields = {
        node.target.id
        for node in runtime.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "diffusion_requested_ngl" in fields
    for name in ("LoadResponse", "InferenceStatusResponse"):
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
        assert any(isinstance(base, ast.Name) and base.id == runtime.name for base in cls.bases)
