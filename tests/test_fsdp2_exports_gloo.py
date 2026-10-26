# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whole-model exports of an FSDP2-sharded model, two gloo ranks on CPU, against one process.

Every rank makes the export call, as a notebook script under torchrun does. The shards are gathered to
rank 0, which runs the single-process export alone; rank 1 writes nothing and returns rank 0's result.
- save_pretrained_merged (merged_16bit): the files equal a one-process merged save, tensor for tensor.
- save_pretrained_gguf (the llama.cpp converter stubbed): rank 0 alone converts, from a 16-bit merge
  equal to the one-process merge; both ranks return the same result.
- push_to_hub_gguf (converter and Hub stubbed): only rank 0 talks to the Hub.
- save_pretrained_torchao: a LoRA model with a given torchao config, and a QAT-trained full model
  (the Qwen3 ExecuTorch notebook), equal to one process; the sharded model still trains afterwards.
- A gathered model that does not fit rank 0's host RAM is refused on every rank, with the numbers.
- An export that fails on rank 0 raises on every rank.
"""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"  # tokenizer only; the weights are built here

fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)
pytest.importorskip("peft")
HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def _import_checkout_unsloth():
    """`import unsloth` from this checkout, conftest's device spoof first; None or why it failed."""
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.path.insert(0, REPO)
    try:
        spec = importlib.util.spec_from_file_location(
            "_exports_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth  # noqa: F401
    except Exception as e:
        return repr(e)
    # The spoofed device is "cuda"; the merge asks its name (T4 check) on this GPU-less host.
    torch.cuda.get_device_name = lambda *args, **kwargs: "cpu"
    return None


def _make_base(directory):
    """A small random Qwen3 checkpoint with real-sized groups (every Linear in_features % 32 == 0)."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(TINY)
    config = Qwen3Config(
        vocab_size = len(tokenizer),
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        tie_word_embeddings = False,
        torch_dtype = "bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


def _load(
    base,
    lora = True,
    qat = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unsloth.save import patch_saving_functions

    model = AutoModelForCausalLM.from_pretrained(base, dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if qat is not None:
        from unsloth.models._utils import _prepare_model_for_qat
        _prepare_model_for_qat(model, qat)
    if lora:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(1)
        model = get_peft_model(
            model,
            LoraConfig(
                r = 4,
                lora_alpha = 8,
                target_modules = ["q_proj", "v_proj", "down_proj"],
                init_lora_weights = False,  # random B: the merge changes the weights
            ),
        )
    return patch_saving_functions(model), patch_saving_functions(tokenizer)


def _shard(model, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (world_size,))
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    for layer in inner.model.layers:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    return model


def _all_dtensor(model):
    from torch.distributed.tensor import DTensor
    return all(isinstance(p, DTensor) for p in model.parameters())


class _StubGGUF:
    """Stands in for unsloth.save.save_to_gguf: records the call and writes a fake GGUF that lists
    the 16-bit checkpoint it was handed."""

    def __init__(
        self,
        rank,
        fail = False,
    ):
        self.rank, self.fail, self.calls = rank, fail, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs["quantization_method"])
        if self.fail:
            raise OSError("converter crashed")
        directory = kwargs["gguf_directory"]
        os.makedirs(directory, exist_ok = True)
        path = os.path.join(directory, "model.Q8_0.gguf")
        with open(path, "w") as f:
            json.dump(sorted(os.listdir(kwargs["model_directory"])), f)
        return [path], False, False


class _StubHub:
    """huggingface_hub.HfApi: records every call; uploads nothing."""

    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def call(*args, **kwargs):
            _StubHub.calls.append(name)
            return {"name": "user"} if name == "whoami" else None

        return call


def _run_cases(rank, world_size, base, out):
    """Every export once, each case on its own so one refusal does not hide the others. `world_size`
    0 is the one-process reference (no process group). A case's exception lands in `<case>_raised`."""
    import huggingface_hub
    import unsloth.save as save
    from unsloth.models import fsdp2_save

    sharded = world_size > 0
    tag = f"rank{rank}" if sharded else "single"
    result = {}

    def load(**kwargs):
        model, tokenizer = _load(base, **kwargs)
        return (_shard(model, world_size) if sharded else model), tokenizer

    def mine(name):
        # rank 0 writes where the reference does; rank 1 gets a folder of its own, which must stay absent
        return os.path.join(out, name if rank == 0 else f"{name}_{tag}")

    def merged():
        model, tokenizer = load()
        result["merged_return"] = model.save_pretrained_merged(
            mine("merged"), tokenizer, save_method = "merged_16bit"
        )
        result["merged_wrote"] = os.path.isdir(mine("merged"))

    def gguf():
        model, tokenizer = load()
        stub = save.save_to_gguf = _StubGGUF(rank)
        answer = model.save_pretrained_gguf(mine("gguf"), tokenizer, quantization_method = "q8_0")
        result["gguf_return"] = {
            k: answer[k] for k in ("gguf_files", "want_full_precision", "is_vlm")
        }
        result["gguf_calls"] = stub.calls
        result["gguf_wrote"] = os.path.isdir(mine("gguf"))
        result["all_dtensor_after_gguf"] = _all_dtensor(model) if sharded else None

    def push_gguf():
        model, tokenizer = load()
        huggingface_hub.HfApi = _StubHub
        _StubHub.calls = []
        stub = save.save_to_gguf = _StubGGUF(rank)
        os.chdir(out)  # the push stages its folders under the working directory
        model.push_to_hub_gguf("user/repo", tokenizer, quantization_method = "q8_0", token = "hf_x")
        result["push_hub_calls"] = sorted(set(_StubHub.calls))
        result["push_gguf_calls"] = stub.calls

    def fail():
        # A converter that fails on rank 0 raises on every rank.
        model, tokenizer = load()
        save.save_to_gguf = _StubGGUF(rank, fail = True)
        try:
            model.save_pretrained_gguf(mine("gguf_fail"), tokenizer)
            result["fail_error"] = None
        except Exception as e:
            result["fail_error"] = f"{type(e).__name__}: {e}"

    def ram():
        # Too big for rank 0's host RAM: refused on every rank before anything is gathered.
        model, tokenizer = load()
        save.save_to_gguf = _StubGGUF(rank)
        real = fsdp2_save._host_available_bytes
        fsdp2_save._host_available_bytes = lambda: 512 * 1024**2
        try:
            model.save_pretrained_gguf(mine("gguf_ram"), tokenizer)
            result["ram_error"] = None
        except Exception as e:
            result["ram_error"] = f"{type(e).__name__}: {e}"
        finally:
            fsdp2_save._host_available_bytes = real
        result["ram_wrote"] = os.path.isdir(mine("gguf_ram"))

    def estimate():
        # An eval forward leaves the root unit gathered; the RAM and placement estimates must
        # still count those weights, as the gather (which reshards first) will move them.
        model, tokenizer = load()
        full = fsdp2_save._gathered_host_bytes(model, offload_to_cpu = True, offload_4bit = True)
        model.eval()
        with torch.no_grad():
            model(input_ids = torch.arange(16).reshape(2, 8))
        result["estimate_gathered_after_eval"] = not _all_dtensor(model)
        after = fsdp2_save._gathered_host_bytes(model, offload_to_cpu = True, offload_4bit = True)
        result["estimate_bytes"] = (full, after)

    def ptq():
        # A LoRA model, quantized after a 16-bit merge.
        from torchao.quantization import Int8WeightOnlyConfig

        model, tokenizer = load()
        model.save_pretrained_torchao(mine("ptq"), tokenizer, torchao_config = Int8WeightOnlyConfig())
        result["ptq_wrote"] = os.path.isdir(mine("ptq") + "-torchao")

    def qat():
        # The ExecuTorch notebook: a full QAT model converted in place, then saved.
        model, tokenizer = load(lora = False, qat = "phone-deployment")
        model.save_pretrained_torchao(mine("qat"), tokenizer)
        result["qat_wrote"] = os.path.isdir(mine("qat"))
        if sharded:
            # Rank 0's model is the sharded fake-quantized one again, so training goes on in step.
            result["qat_all_dtensor_after"] = _all_dtensor(model)
            ids = torch.arange(16).reshape(2, 8) + 100 * rank
            loss = model(input_ids = ids, labels = ids).loss
            loss.backward()
            result["qat_loss_after"] = float(loss)

    def qat_fallback():
        # Too big for rank 0's GPU: gathered to the CPU and converted one module at a time on the
        # accelerator (the CPU stands in here), which must equal the whole-model convert.
        from unsloth.models import _utils

        calls = []
        real_convert = _utils._convert_module_by_module
        real_options = fsdp2_save.gather_on_accelerator_if_it_fits

        def counted(*args, **kwargs):
            calls.append(1)
            return real_convert(*args, **kwargs)

        _utils._convert_module_by_module = counted
        if sharded:
            fsdp2_save.gather_on_accelerator_if_it_fits = lambda model: dict(
                offload_to_cpu = True, offload_4bit = True, convert_on = torch.device("cpu")
            )
        try:
            model, tokenizer = load(lora = False, qat = "phone-deployment")
            model.save_pretrained_torchao(mine("qat_fallback"), tokenizer)
        finally:
            _utils._convert_module_by_module = real_convert
            fsdp2_save.gather_on_accelerator_if_it_fits = real_options
        result["qat_fallback_wrote"] = os.path.isdir(mine("qat_fallback"))
        result["qat_fallback_calls"] = len(calls)

    def cgroup():
        # A container is killed at its cgroup limit, not at host RAM: the check takes the smaller.
        import builtins

        root = os.path.join(out, f"cgroup_{tag}")
        os.makedirs(os.path.join(root, "job"), exist_ok = True)
        for name, value in (("memory.max", 4 * 1024**3), ("memory.current", 1024**3)):
            with open(os.path.join(root, "job", name), "w") as f:
                f.write(str(value))
        with open(os.path.join(root, "self"), "w") as f:
            f.write("0::/job\n")
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            path = str(path)
            if path == "/proc/self/cgroup":
                path = os.path.join(root, "self")
            elif path.startswith("/sys/fs/cgroup"):
                path = root + path[len("/sys/fs/cgroup") :]
            return real_open(path, *args, **kwargs)

        # cgroup v1: the limit on the parent (/job), the process in /job/worker under no limit of its own.
        v1 = os.path.join(root, "v1")
        os.makedirs(os.path.join(v1, "memory", "job", "worker"), exist_ok = True)
        for where, limit, used in (("job", 2 * 1024**3, 512 * 1024**2), ("job/worker", 1 << 62, 0)):
            for name, value in (("memory.limit_in_bytes", limit), ("memory.usage_in_bytes", used)):
                with open(os.path.join(v1, "memory", where, name), "w") as f:
                    f.write(str(value))
        with open(os.path.join(v1, "self"), "w") as f:
            f.write("4:memory:/job/worker\n")

        builtins.open = fake_open
        try:
            result["cgroup_bytes"] = (
                fsdp2_save._cgroup_available_bytes(),
                fsdp2_save._host_available_bytes(),
            )
            root = v1
            result["cgroup_v1_bytes"] = fsdp2_save._cgroup_available_bytes()
        finally:
            builtins.open = real_open

    def restore_devices():
        # A replicated bias on another device than the gathered weight goes back to its own device
        # (the meta device stands in for the GPU; CPU <-> CUDA keeps the Parameter object via .data,
        # CPU -> meta cannot, so it gets a new one). A tensor already home is left untouched.
        from unsloth.models import _utils

        layer = torch.nn.Linear(4, 4)
        weight = layer.weight
        homes = {"weight": torch.device("cpu"), "bias": torch.device("meta")}
        _utils._restore_tensor_devices(layer, homes, torch.device("cpu"))
        result["restore_devices"] = [
            str(layer.weight.device),
            str(layer.bias.device),
            layer.weight is weight,
            isinstance(layer.bias, torch.nn.Parameter),
        ]

    def placement():
        # 10 GiB gathered converts on a 16 GiB card only with the quantized copy counted too.
        result["placement_need"] = fsdp2_save._accelerator_convert_bytes(10 * 1024**3)

    cases = [merged, gguf, push_gguf, placement, restore_devices]
    if sharded:
        cases += [fail, ram, estimate, cgroup]
    if HAS_TORCHAO:
        cases += [ptq, qat, qat_fallback]
    for case in cases:
        try:
            case()
        except Exception as e:
            result[f"{case.__name__}_raised"] = f"{type(e).__name__}: {e}"
    return result


def _worker(rank, world_size, port, base, out, queue):
    try:
        why = _import_checkout_unsloth()
        if why is not None:
            queue.put((rank, {"skip": why}))
            return
        if world_size:
            import torch.distributed as dist
            os.environ.update(
                MASTER_ADDR = "127.0.0.1",
                MASTER_PORT = str(port),
                RANK = str(rank),
                WORLD_SIZE = str(world_size),
            )
            dist.init_process_group("gloo", rank = rank, world_size = world_size)
        result = _run_cases(rank, world_size, base, out)
        queue.put((rank, result))
        if world_size:
            dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _spawn(world_size, base, out):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    ranks = range(max(world_size, 1))
    procs = [
        ctx.Process(target = _worker, args = (r, world_size, port, base, out, queue)) for r in ranks
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, result = queue.get(timeout = 900)
            results[rank] = result
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.terminate()
    for r in results.values():
        if "skip" in r:
            pytest.skip(f"import unsloth failed on this host: {r['skip']}")
        assert "error" not in r, r.get("error")
    return results


def _tensors(directory):
    """name -> tensor of every weight file in `directory` (safetensors or torch pickles)."""
    from safetensors.torch import load_file

    tensors = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith(".safetensors"):
            tensors.update(load_file(path))
        elif name.endswith(".bin") and "training_args" not in name:
            tensors.update(torch.load(path, map_location = "cpu", weights_only = False))
    return tensors


def _plain(t):
    """A comparable plain tensor: a torchao quantized tensor by its inner data (qdata, scales, ...)."""
    if hasattr(t, "__tensor_flatten__") and type(t) is not torch.Tensor:
        names, _ = t.__tensor_flatten__()
        return [_plain(getattr(t, n)) for n in names]
    return t.detach().cpu()


def _same(a, b):
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(_same, a, b))
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def _assert_same_files(single, sharded):
    a, b = _tensors(single), _tensors(sharded)
    assert a and sorted(a) == sorted(b), (sorted(a), sorted(b))
    differ = [k for k in a if not _same(_plain(a[k]), _plain(b[k]))]
    assert not differ, differ


def _no_raise(runs, case):
    for where, result in [("one process", runs["single"])] + [
        (f"rank {r}", runs["ranks"][r]) for r in (0, 1)
    ]:
        raised = result.get(f"{case}_raised")
        assert raised is None, f"{case} raised on {where}: {raised}"


@pytest.fixture(scope = "module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("fsdp2_exports")
    base = str(root / "base")
    # Built here with transformers alone: `import unsloth` happens only in the spawned processes,
    # so this one keeps the unsloth modules other test files imported (and monkeypatch).
    _make_base(base)
    single, sharded = root / "single", root / "sharded"
    single.mkdir()
    sharded.mkdir()
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        one = _spawn(0, base, str(single))[0]
        two = _spawn(2, base, str(sharded))
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return {"single": one, "ranks": two, "single_dir": single, "sharded_dir": sharded}


def test_merged_16bit_equals_one_process(runs):
    _no_raise(runs, "merged")
    for rank in (0, 1):
        assert runs["ranks"][rank]["merged_return"] is None
    assert not runs["ranks"][1]["merged_wrote"]
    _assert_same_files(runs["single_dir"] / "merged", runs["sharded_dir"] / "merged")


def test_gguf_runs_on_rank0_only_from_the_same_merge(runs):
    _no_raise(runs, "gguf")
    r0, r1 = runs["ranks"][0], runs["ranks"][1]
    assert r0["gguf_calls"] == [["q8_0"]] and r1["gguf_calls"] == [], (r0, r1)
    assert not r1["gguf_wrote"]
    assert r0["all_dtensor_after_gguf"] and r1["all_dtensor_after_gguf"]
    # Every rank returns rank 0's answer.
    assert r0["gguf_return"] == r1["gguf_return"]
    single = runs["single"]["gguf_return"]
    assert [os.path.basename(p) for p in single["gguf_files"]] == [
        os.path.basename(p) for p in r0["gguf_return"]["gguf_files"]
    ]
    # The 16-bit checkpoint the converter was handed equals the one-process one.
    _assert_same_files(runs["single_dir"] / "gguf", runs["sharded_dir"] / "gguf")


def test_push_to_hub_gguf_uploads_from_rank0_only(runs):
    _no_raise(runs, "push_gguf")
    r0, r1 = runs["ranks"][0], runs["ranks"][1]
    assert "create_commit" in r0["push_hub_calls"] or "upload_file" in r0["push_hub_calls"], r0
    assert r0["push_gguf_calls"] == [["q8_0"]]
    assert r1["push_hub_calls"] == [] and r1["push_gguf_calls"] == [], r1


def test_a_rank0_failure_raises_on_every_rank(runs):
    _no_raise(runs, "fail")
    r0, r1 = runs["ranks"][0], runs["ranks"][1]
    assert r0["fail_error"] and "converter crashed" in r0["fail_error"], r0["fail_error"]
    assert r1["fail_error"] and "failed on rank 0" in r1["fail_error"], r1["fail_error"]
    assert "converter crashed" in r1["fail_error"]


def test_too_big_for_host_ram_is_refused_on_every_rank(runs):
    _no_raise(runs, "ram")
    for rank in (0, 1):
        error = runs["ranks"][rank]["ram_error"]
        assert error and "host RAM" in error and "0.5 GiB is available" in error, error
        assert not runs["ranks"][rank]["ram_wrote"]


def test_export_estimate_counts_units_an_eval_forward_left_gathered(runs):
    _no_raise(runs, "estimate")
    for rank in (0, 1):
        res = runs["ranks"][rank]
        assert res["estimate_gathered_after_eval"], "the eval forward must leave a unit gathered"
        full, after = res["estimate_bytes"]
        assert full > 0 and after == full, res["estimate_bytes"]


@pytest.mark.skipif(not HAS_TORCHAO, reason = "torchao not installed")
def test_torchao_lora_ptq_equals_one_process(runs):
    _no_raise(runs, "ptq")
    assert runs["ranks"][0]["ptq_wrote"] and not runs["ranks"][1]["ptq_wrote"]
    _assert_same_files(runs["single_dir"] / "ptq-torchao", runs["sharded_dir"] / "ptq-torchao")


@pytest.mark.skipif(not HAS_TORCHAO, reason = "torchao not installed")
def test_torchao_qat_full_model_equals_one_process_and_keeps_training(runs):
    _no_raise(runs, "qat")
    r0, r1 = runs["ranks"][0], runs["ranks"][1]
    assert r0["qat_wrote"] and not r1["qat_wrote"]
    assert r0["qat_all_dtensor_after"] and r1["qat_all_dtensor_after"]
    assert r0["qat_loss_after"] == r0["qat_loss_after"]  # finite, not NaN
    _assert_same_files(runs["single_dir"] / "qat", runs["sharded_dir"] / "qat")


@pytest.mark.skipif(not HAS_TORCHAO, reason = "torchao not installed")
def test_torchao_qat_fallback_converts_module_by_module_and_equals_one_process(runs):
    _no_raise(runs, "qat_fallback")
    r0, r1 = runs["ranks"][0], runs["ranks"][1]
    assert r0["qat_fallback_wrote"] and not r1["qat_fallback_wrote"]
    assert r0["qat_fallback_calls"] > 0 and r1["qat_fallback_calls"] == 0
    assert runs["single"]["qat_fallback_calls"] == 0
    _assert_same_files(runs["single_dir"] / "qat_fallback", runs["sharded_dir"] / "qat_fallback")


def test_host_ram_check_honours_the_cgroup_limit(runs):
    _no_raise(runs, "cgroup")
    limit, available = runs["ranks"][0]["cgroup_bytes"]
    assert limit == 3 * 1024**3 and available <= limit, runs["ranks"][0]["cgroup_bytes"]
    assert runs["ranks"][0]["cgroup_v1_bytes"] == 1536 * 1024**2, runs["ranks"][0][
        "cgroup_v1_bytes"
    ]


def test_gpu_placement_counts_the_converted_copy(runs):
    _no_raise(runs, "placement")
    assert runs["single"]["placement_need"] == 15 * 1024**3


def test_module_by_module_convert_restores_each_tensor_device(runs):
    _no_raise(runs, "restore_devices")
    assert list(runs["single"]["restore_devices"]) == ["cpu", "meta", True, True]
