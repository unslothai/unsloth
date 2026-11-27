# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`from_pretrained(..., gguf_file = ...)` on the MLX, CPU and GPU entry points.

1. MLX (Apple Silicon): mlx-lm reads safetensors only, and the loader used to drop `gguf_file` silently: a GGUF repo
   failed with "No safetensors found", a .gguf path with NotADirectoryError, and a repo holding both loaded the
   safetensors instead of the GGUF. It now refuses with one error that names the options.
2. A GGUF repo with no tokenizer files: the tokenizer load ignored `gguf_file`, so transformers 5 built an empty
   tokenizer (vocab 1, every forward fails on 0 tokens) and 4.57 raised. The tokenizer is now read from the GGUF.
3. The `gguf` package is not an Unsloth dependency: a missing one is named, with the install command, before any
   download instead of transformers' ImportError after the weights.
"""

import ast
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "unsloth" / "__init__.py"


def _gguf_load():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_gguf_load_test", ROOT / "unsloth" / "_gguf_load.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def write_tiny_gguf_repo(
    folder,
    arch = "qwen3",
    name = "tiny.gguf",
):
    """config.json + a float32 GGUF whose metadata holds a byte-level BPE tokenizer: a GGUF repo with no tokenizer
    files. Returns the vocabulary size."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    hidden, inter, heads, kv = 16, 32, 2, 1
    # transformers reads no qwen3 key_length from a GGUF, so its head_dim stays at the default 128.
    head_dim = 128 if arch == "qwen3" else hidden // heads
    tokens = _bytes_to_unicode() + ["Ġt", "he", "<|endoftext|>"]
    vocab = len(tokens)
    os.makedirs(folder, exist_ok = True)
    w = gguf.GGUFWriter(os.path.join(folder, name), arch)
    w.add_context_length(64)
    w.add_embedding_length(hidden)
    w.add_block_count(1)
    w.add_feed_forward_length(inter)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_dimension_count(head_dim)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2" if arch == "qwen3" else "default")
    w.add_token_list(tokens)
    w.add_token_types([1] * (vocab - 1) + [3])
    w.add_token_merges(["Ġ t", "h e"])
    w.add_eos_token_id(vocab - 1)
    w.add_bos_token_id(vocab - 1)
    rng = np.random.default_rng(0)

    def add(tensor, *shape):
        w.add_tensor(tensor, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    add("token_embd.weight", vocab, hidden)
    add("output.weight", vocab, hidden)
    for norm in ("output_norm", "blk.0.attn_norm", "blk.0.ffn_norm"):
        w.add_tensor(f"{norm}.weight", np.ones(hidden, dtype = np.float32))
    if arch == "qwen3":
        w.add_tensor("blk.0.attn_q_norm.weight", np.ones(head_dim, dtype = np.float32))
        w.add_tensor("blk.0.attn_k_norm.weight", np.ones(head_dim, dtype = np.float32))
    add("blk.0.attn_q.weight", heads * head_dim, hidden)
    add("blk.0.attn_k.weight", kv * head_dim, hidden)
    add("blk.0.attn_v.weight", kv * head_dim, hidden)
    add("blk.0.attn_output.weight", hidden, heads * head_dim)
    add("blk.0.ffn_gate.weight", inter, hidden)
    add("blk.0.ffn_up.weight", inter, hidden)
    add("blk.0.ffn_down.weight", hidden, inter)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    config = {
        "architectures": ["Qwen3ForCausalLM" if arch == "qwen3" else "LlamaForCausalLM"],
        "model_type": arch,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "num_hidden_layers": 1,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "eos_token_id": vocab - 1,
        "bos_token_id": vocab - 1,
        "torch_dtype": "float32",
    }
    with open(os.path.join(folder, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return vocab


def _as_hub_cache(
    folder,
    cache_dir,
    repo_id = "unsloth-test/tiny-GGUF",
):
    """The folder as a cached Hub snapshot of `repo_id`, so a repo-id load resolves offline."""
    repo = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents = True)
    for f in Path(folder).iterdir():
        (snapshot / f.name).write_bytes(f.read_bytes())
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0" * 40)
    return repo_id


# ---------------------------------------------------------------- 1. MLX


def _mlx_from_pretrained(stub):
    tree = ast.parse(INIT.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_mlx_from_pretrained"
    )
    # The relative import resolves against a stand-in package holding the module under test.
    package = types.ModuleType("_gguf_mlx_pkg")
    package.__path__ = []
    sys.modules["_gguf_mlx_pkg"] = package
    if (ROOT / "unsloth" / "_gguf_load.py").is_file():
        sys.modules["_gguf_mlx_pkg._gguf_load"] = _gguf_load()
    namespace = {
        "__name__": "_gguf_mlx_pkg",
        "__package__": "_gguf_mlx_pkg",
        "FastMLXModel": stub,
        "is_bfloat16_supported": lambda: True,
        "_inspect": inspect,
        "_types": types,
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(INIT), "exec"), namespace)
    return namespace["_mlx_from_pretrained"]


class _StubMLX:
    calls = []

    @staticmethod
    def from_pretrained(
        model_name = None,
        max_seq_length = 2048,
        dtype = None,
        random_state = 3407,
        **kwargs,
    ):
        _StubMLX.calls.append((model_name, kwargs))
        return "model", "tokenizer"


@pytest.mark.parametrize(
    "args, kwargs",
    [
        pytest.param(
            ("unsloth/Qwen3-0.6B-GGUF",), {"gguf_file": "Qwen3-0.6B-Q4_K_M.gguf"}, id = "gguf-repo"
        ),
        pytest.param(
            ("unsloth/Qwen3-0.6B",), {"gguf_file": "Qwen3-0.6B-Q4_K_M.gguf"}, id = "safetensors-repo"
        ),
        pytest.param((), {"model_name": "/models/Qwen3-0.6B-Q4_K_M.gguf"}, id = "local-gguf-path"),
        pytest.param(("/models/Qwen3-0.6B-Q4_K_M.GGUF",), {}, id = "local-gguf-path-upper"),
    ],
)
def test_mlx_refuses_gguf_instead_of_loading_something_else(args, kwargs):
    _StubMLX.calls = []
    load = _mlx_from_pretrained(_StubMLX)
    with pytest.raises(NotImplementedError) as error:
        load(*args, **kwargs)
    message = str(error.value)
    assert "MLX" in message and "safetensors" in message
    assert "UNSLOTH_FORCE_CPU=1" in message and "UNSLOTH_FORCE_GPU_PATH=1" in message
    assert "llama.cpp" in message
    assert _StubMLX.calls == []


def test_mlx_without_gguf_loads_as_before():
    _StubMLX.calls = []
    load = _mlx_from_pretrained(_StubMLX)
    assert load("unsloth/Qwen3-0.6B", max_seq_length = 512) == ("model", "tokenizer")
    assert _StubMLX.calls == [("unsloth/Qwen3-0.6B", {})]


# ---------------------------------------------------------------- 3. the gguf package


def test_missing_gguf_package_names_the_install(monkeypatch):
    module = _gguf_load()
    monkeypatch.setitem(sys.modules, "gguf", None)  # find_spec("gguf") is None
    module.require_gguf_package(None)  # no GGUF asked for: nothing to check
    with pytest.raises(ImportError, match = r'pip install "gguf>=0\.10\.0"') as error:
        module.require_gguf_package("model-Q4_K_M.gguf")
    assert "unsloth[gguf]" in str(error.value)


def test_old_gguf_package_is_refused(monkeypatch):
    module = _gguf_load()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "_gguf_version", lambda: "0.9.1")
    with pytest.raises(ImportError, match = "0.9.1, which is too old"):
        module.require_gguf_package("model.gguf")
    monkeypatch.setattr(module, "_gguf_version", lambda: "0.19.0")
    module.require_gguf_package("model.gguf")


def test_every_entry_point_checks_the_gguf_package_first():
    loader = (ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8")
    assert (
        loader.count('require_gguf_package(kwargs.get("gguf_file"))') == 2
    )  # FastLanguageModel, FastModel
    cpu = (ROOT / "unsloth" / "_cpu_init.py").read_text(encoding = "utf-8")
    assert cpu.index("require_gguf_package(") < cpu.index("FastCPUModel.from_pretrained(*args")


_CPU_PROBE = r"""
import json, sys
sys.modules["gguf"] = None  # an install without the gguf package
import unsloth
from unsloth import FastLanguageModel
try:
    FastLanguageModel.from_pretrained(sys.argv[1], gguf_file = "tiny.gguf", max_seq_length = 64, load_in_4bit = False)
    out = {"device_type": unsloth.DEVICE_TYPE, "error": None}
except Exception as e:
    out = {"device_type": unsloth.DEVICE_TYPE, "error": f"{type(e).__name__}: {e}"}
print("PROBE " + json.dumps(out))
"""


def test_cpu_backend_names_the_missing_gguf_package(tmp_path):
    folder = tmp_path / "repo"
    write_tiny_gguf_repo(folder, "qwen3")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "UNSLOTH_FORCE_GPU_PATH": "1",
            "UNSLOTH_FORCE_CPU": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _CPU_PROBE, str(folder)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    probe = next(
        (json.loads(l[6:]) for l in out.stdout.splitlines() if l.startswith("PROBE ")), None
    )
    if probe is None:
        pytest.skip("the CPU backend does not import here: " + out.stderr[-500:])
    if probe["device_type"] != "cpu":
        pytest.skip(f"this host does not select the CPU backend ({probe['device_type']})")
    assert probe["error"] and probe["error"].startswith(
        "ImportError: Unsloth: loading `gguf_file"
    ), probe
    assert 'pip install "gguf>=0.10.0"' in probe["error"]


# ---------------------------------------------------------------- 2. the tokenizer of a GGUF-only repo


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_tokenizer_comes_from_the_gguf_when_the_folder_has_none(tmp_path, arch):
    pytest.importorskip("transformers")
    vocab = write_tiny_gguf_repo(tmp_path, arch)
    module = _gguf_load()
    assert module.has_tokenizer_files(str(tmp_path)) is False
    tokenizer = module.load_gguf_tokenizer_if_missing(
        str(tmp_path), "tiny.gguf", padding_side = "right"
    )
    assert len(tokenizer) >= vocab - 1
    assert len(tokenizer("hello there").input_ids) > 0
    assert tokenizer.padding_side == "right"
    # No gguf_file: nothing to read it from.
    assert module.load_gguf_tokenizer_if_missing(str(tmp_path), None) is None


def test_tokenizer_comes_from_the_gguf_for_a_cached_repo_id(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import huggingface_hub.constants

    vocab = write_tiny_gguf_repo(tmp_path / "src", "qwen3")
    cache = tmp_path / "hub"
    repo_id = _as_hub_cache(tmp_path / "src", cache)
    # transformers reads the GGUF tokenizer from the default cache, whatever cache_dir says.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    module = _gguf_load()
    assert module.has_tokenizer_files(repo_id, cache_dir = str(cache), local_files_only = True) is False
    tokenizer = module.load_gguf_tokenizer_if_missing(
        repo_id, "tiny.gguf", cache_dir = str(cache), local_files_only = True
    )
    assert len(tokenizer) >= vocab - 1 and len(tokenizer("hello").input_ids) > 0


def test_a_repo_with_tokenizer_files_keeps_its_own(tmp_path):
    folder = tmp_path / "repo"
    write_tiny_gguf_repo(folder, "qwen3")
    (folder / "tokenizer.json").write_text("{}")
    module = _gguf_load()
    assert module.has_tokenizer_files(str(folder)) is True
    assert module.load_gguf_tokenizer_if_missing(str(folder), "tiny.gguf") is None
    cache = tmp_path / "hub"
    repo_id = _as_hub_cache(folder, cache, "unsloth-test/with-tokenizer")
    assert module.has_tokenizer_files(repo_id, cache_dir = str(cache), local_files_only = True) is True


def test_gpu_loaders_pass_gguf_file_to_the_tokenizer():
    models = ROOT / "unsloth" / "models"
    llama = (models / "llama.py").read_text(encoding = "utf-8")
    call = llama[llama.index("tokenizer = load_correct_tokenizer(") :]
    call = call[: call.index("\n        )\n")]
    assert 'gguf_file = kwargs.get("gguf_file") if tokenizer_name == model_name else None' in call
    vision = (models / "vision.py").read_text(encoding = "utf-8")
    assert vision.index("load_gguf_tokenizer_if_missing(") < vision.index(
        "tokenizer, _primary_err = _acquire_processor(local_files_only)"
    )
    signature = (ROOT / "unsloth" / "tokenizer_utils.py").read_text(encoding = "utf-8")
    assert (
        "load_gguf_tokenizer_if_missing("
        in signature[signature.index("def load_correct_tokenizer(") :]
    )


def test_load_correct_tokenizer_reads_the_gguf(tmp_path):
    vocab = write_tiny_gguf_repo(tmp_path, "qwen3")
    try:
        from unsloth.tokenizer_utils import load_correct_tokenizer
    except Exception as e:  # GPU-less runner: `import unsloth` refuses
        pytest.skip(f"unsloth does not import here: {type(e).__name__}")
    tokenizer = load_correct_tokenizer(str(tmp_path), model_max_length = 64, gguf_file = "tiny.gguf")
    assert len(tokenizer) >= vocab - 1 and len(tokenizer("hello there").input_ids) > 0


_MLX_PROBE = r"""
import json, sys
import unsloth
out = {"is_mlx": bool(getattr(unsloth, "_IS_MLX", False)), "cases": {}}
if out["is_mlx"]:
    from unsloth import FastLanguageModel, FastVisionModel
    cases = {
        "gguf-repo": (FastLanguageModel, ("unsloth/Qwen3-0.6B-GGUF",), {"gguf_file": "Qwen3-0.6B-Q4_K_M.gguf"}),
        "safetensors-repo": (FastLanguageModel, ("unsloth/Qwen3-0.6B",), {"gguf_file": "Qwen3-0.6B-Q4_K_M.gguf"}),
        "local-gguf-path": (FastLanguageModel, (sys.argv[1],), {}),
        "vision": (FastVisionModel, ("unsloth/Qwen3-0.6B-GGUF",), {"gguf_file": "Qwen3-0.6B-Q4_K_M.gguf"}),
    }
    for name, (cls, args, kwargs) in cases.items():
        try:
            cls.from_pretrained(*args, max_seq_length = 64, **kwargs)
            out["cases"][name] = "loaded"
        except Exception as e:
            out["cases"][name] = f"{type(e).__name__}: {e}"
print("PROBE " + json.dumps(out))
"""


@pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("mlx") is None,
    reason = "needs Apple Silicon with MLX",
)
def test_real_mlx_backend_refuses_gguf(tmp_path):
    local = tmp_path / "Qwen3-0.6B-Q4_K_M.gguf"
    local.write_bytes(b"GGUF")
    env = dict(os.environ, HF_HUB_OFFLINE = "1")
    env.pop("UNSLOTH_FORCE_GPU_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _MLX_PROBE, str(local)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    probe = next(
        (json.loads(l[6:]) for l in out.stdout.splitlines() if l.startswith("PROBE ")), None
    )
    assert probe is not None, out.stdout[-2000:] + out.stderr[-4000:]
    if not probe["is_mlx"]:
        pytest.skip("unsloth did not select the MLX backend here")
    for name, outcome in probe["cases"].items():
        assert outcome.startswith("NotImplementedError: Unsloth: the MLX backend"), (name, outcome)
