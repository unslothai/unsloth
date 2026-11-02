# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallel (Ulysses) composed with tensor / pipeline parallel through SFTTrainer, on a gloo CPU world, vs
the same script in one process: losses, per-parameter gradients of the first optimizer step and the adapters
after the run.

CPU ranks cannot run Unsloth's fast Llama attention (it needs CUDA), so the attention is the Hugging Face one with
the Ulysses wrapper hybrids use (utils/ulysses_hybrid.install_hf_attention_wrapper), and the model support check is
opened for it. Everything else (the env layout, the mesh, tensor parallel, the CP sharding, the token count and
the gradient reduction) is the production path."""

import multiprocessing
import os

if os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import (  # noqa: E402
    _free_port,
    _import_checkout_unsloth,
    _raised,
    _save_checkpoint,
)

_ENV_KEYS = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)
_ADAM_EPS = 1e-4  # as test_tp_env_gloo
SEQ = 16
N_LAYERS = 4


def _checkpoint(tmp, arch = "llama"):
    from transformers import AutoTokenizer

    from test_tp_native_oracle import LLAMA, QWEN3, QWEN3_MOE, build

    if arch in ("qwen3_moe", "qwen3", "qwen3_moe_wide"):
        # qwen3_moe: 4 heads / 2 kv heads, 4 experts (2 per token), router aux loss off as shipped.
        # qwen3_moe_wide: hidden and expert widths of 32, so Unsloth's Triton grouped GEMM takes the experts (it
        # needs multiples of 16; the shipped hidden size 8 sends unsloth_triton to torch._grouped_mm).
        repo = QWEN3 if arch == "qwen3" else QWEN3_MOE
        overrides = {"num_hidden_layers": 2}
        if arch == "qwen3_moe_wide":
            overrides.update(hidden_size = 32, moe_intermediate_size = 32)
        path = os.path.join(tmp, "ckpt")
        model = build(repo, overrides, lora = False)
        model.generation_config.pad_token_id = None
        model.save_pretrained(path, safe_serialization = True)
        AutoTokenizer.from_pretrained(repo).save_pretrained(path)
        return path
    # 4 heads / 4 kv heads: tp 2 leaves 2 local heads, which Ulysses splits over cp 2.
    path = _save_checkpoint("llama", os.path.join(tmp, "ckpt"), {"num_hidden_layers": N_LAYERS})
    AutoTokenizer.from_pretrained(LLAMA).save_pretrained(path)
    return path


def _full(t):
    from torch.distributed.tensor import DTensor
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().float().cpu().clone()


def _block_compile_on_cpu(model):
    """Unsloth's block compile (one torch.compile region per decoder layer) on CPU ranks: its CUDA gate opened
    and the aot_eager backend instead of inductor. Graph breaks and recompiles are Dynamo's, so they count the
    same; the generated kernels are not under test here."""
    from unsloth.models import _block_compile as bc

    real_compile = torch.compile

    def compile_aot_eager(fn, **kwargs):
        kwargs.pop("options", None)
        kwargs["backend"] = "aot_eager"
        return real_compile(fn, **kwargs)

    available = torch.cuda.is_available
    bc.torch.cuda.is_available = lambda: True
    bc.torch.compile = compile_aot_eager
    try:
        return bc.patch_block_compile(model)
    finally:
        bc.torch.cuda.is_available = available
        bc.torch.compile = real_compile


def _selected_moe_backend():
    try:
        from unsloth_zoo.temporary_patches import moe_utils
        return moe_utils.select_moe_backend()
    except Exception as e:
        return f"unknown: {e!r}"[:200]


def _count_moe_backend_calls():
    """Engagement for GPU MoE cells: calls into each Unsloth Zoo expert backend and into torch._grouped_mm."""
    import collections

    calls = collections.Counter()
    try:
        from unsloth_zoo.temporary_patches import moe_utils
    except Exception:
        return calls
    modules = [moe_utils]
    try:
        cached = moe_utils._load_cached_moe_utils_module()
        if cached is not None and cached is not moe_utils:
            modules.append(cached)
    except Exception:
        pass
    try:
        from unsloth_zoo.temporary_patches import moe_utils_bnb4bit
        modules.append(moe_utils_bnb4bit)
    except Exception:
        pass

    def counted(name, fn):
        def wrapper(*a, **k):
            calls[name] += 1
            return fn(*a, **k)

        wrapper.__wrapped__ = fn
        return wrapper

    for module in modules:
        for name in (
            "forward_native_grouped_mm",
            "forward_triton_grouped_gemm",
            "forward_native_moe_loop",
            "forward_moe_backend_bnb4bit",
            "_triton_grouped_mm",
            "_manual_grouped_mm",
        ):
            if callable(getattr(module, name, None)):
                setattr(module, name, counted(name, getattr(module, name)))
    try:
        from unsloth.kernels.moe.grouped_gemm import interface
        interface.grouped_gemm = counted("unsloth grouped_gemm (Triton)", interface.grouped_gemm)
    except Exception:
        pass
    if hasattr(torch, "_grouped_mm"):
        torch._grouped_mm = counted("torch._grouped_mm", torch._grouped_mm)
    return calls


def _case_train(
    rank,
    world,
    tmp,
    ckpt,
    bs,
    cp = 1,
    steps = 2,
    ga = 2,
    sft_cp = True,
    keep = False,
    compile = False,
    arch = "llama",
    router_aux = False,
    attn = "sdpa",
    ungate = False,
    mode = "lora",
    trl_aux = False,
    device = "cpu",
    bypass_pre = False,
    bypass_post = False,
):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    import transformers
    import trl
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unsloth import context_parallel as cpm
    from unsloth.models import pp_native, tp_env, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix
    from unsloth.utils.ulysses_hybrid import install_hf_attention_wrapper

    patch_gradient_accumulation_fix(transformers.Trainer)
    cpm.patch_sft_trainer()
    if device == "cpu":
        cpm.DEVICE_TYPE_TORCH = "cpu"
    else:  # NCCL ranks under torchrun (scripts/local_correctness.py): one GPU each
        if torch.cuda.device_count() < world:
            # Correctness only: every rank shares the one card (a one-GPU cloud VM). NCCL >= 2.32 takes that with
            # NCCL_MULTI_RANK_GPU_ENABLE=1 (the driver probes it), else gloo on CUDA tensors.
            os.environ["LOCAL_RANK"], os.environ["LOCAL_WORLD_SIZE"] = "0", "1"
            import torch.distributed as dist
            if world > 1 and not dist.is_initialized():
                dist.init_process_group(os.environ.get("UNSLOTH_TEST_SHARED_GPU_BACKEND", "gloo"))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        # fp32 like the CPU reference: no TF32 matmuls / convolutions (Unsloth may have allowed them).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # The math SDPA kernel: deterministic fp32 (the fused ones differ in the last bits per head count).
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.use_deterministic_algorithms(True, warn_only = True)
    # The test seam: CPU ranks run the Hugging Face attention through the Ulysses wrapper.
    cpm._supports_context_parallel = lambda model, backend = "ring": True
    if bypass_pre or bypass_post:
        # Evidence runs only: switch off the aux-loss guards to show what each one catches.
        from unsloth import parallel_capabilities
        if bypass_pre:
            parallel_capabilities._aux_loss_on = lambda *a, **k: False
        if bypass_post:
            cpm._refuse_router_aux_loss_after_init = lambda *a, **k: None
    if ungate:
        # Measure a cell the registry refuses for lack of evidence (the evidence run itself).
        from unsloth import parallel_capabilities
        parallel_capabilities._gated = lambda query: False
    from unsloth.utils import attention_dispatch

    attention_dispatch.HAS_XFORMERS = (
        False  # torch SDPA on every rank (fp32: flash / xformers take 16-bit)
    )
    attention_dispatch.HAS_FLASH_ATTENTION = False
    layout = tp_env.env_layout()

    def load(
        model_name,
        load_in_4bit = True,
        **kwargs,
    ):
        extra = {}
        if mode == "qlora":
            from transformers import BitsAndBytesConfig
            extra = dict(
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_compute_dtype = torch.float32,
                ),
                device_map = {"": torch.cuda.current_device()},
            )
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = attn, **extra
        )

    def peft(model):
        torch.manual_seed(1)
        config = LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = "all-linear",
            init_lora_weights = False,
            lora_dropout = 0.0,
        )
        return get_peft_model(model, config)

    load = tp_env.tensor_parallel_env_load(load)
    peft = tp_env.tensor_parallel_env_peft(peft)
    model = load(ckpt, load_in_4bit = False)
    if mode in ("lora", "qlora"):
        model = peft(model)
    else:
        model.train()
    count = device != "cpu" or os.environ.get("UNSLOTH_TEST_COUNT_MOE") == "1"
    moe_calls = _count_moe_backend_calls() if count else None
    if router_aux:
        model.config.output_router_logits = True
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            install_hf_attention_wrapper(module, "sdpa", cpm.get_cp_manager)
    compiled = _block_compile_on_cpu(model) if compile else []

    # Data parallel replicas read different rows; cp / tp / pp peers share theirs.
    model_ranks = cp * (1 if layout is None else layout.tp * layout.pp)
    data_dp = world // model_ranks
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (steps * ga * bs * data_dp, SEQ), generator = g)
    import datasets

    data = datasets.Dataset.from_dict({"input_ids": ids.tolist(), "row": list(range(len(ids)))})
    rows_read = []  # dataset row ids per micro-batch, in order: parity needs the same rows on both sides

    def collate(rows):
        rows_read.append([int(r["row"]) for r in rows])
        input_ids = torch.tensor([r["input_ids"] for r in rows])
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}
        batch["labels"][
            batch["labels"] % 5 == 0
        ] = -100  # uneven token counts per row and per shard
        return batch

    class SequentialSFT(trl.SFTTrainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    grads = {}

    class Grabber(transformers.TrainerCallback):
        def on_pre_optimizer_step(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if grads:
                return
            target = trainer.model
            for name, p in target.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grads[name] = _full(p.grad)

    args = trl.SFTConfig(
        output_dir = os.path.join(tmp, "out"),
        per_device_train_batch_size = bs,
        gradient_accumulation_steps = ga,
        max_steps = steps,
        learning_rate = 1e-2,
        optim = "adamw_torch",
        adam_epsilon = _ADAM_EPS,
        lr_scheduler_type = "constant",
        warmup_steps = 0,
        weight_decay = 0.0,
        max_grad_norm = 1.0,
        logging_steps = 1,
        save_strategy = "no",
        report_to = [],
        use_cpu = device == "cpu",
        seed = 0,
        dataloader_drop_last = True,
        remove_unused_columns = False,
        disable_tqdm = True,
        dataset_kwargs = {"skip_prepare_dataset": True},
        gradient_checkpointing = False,
    )
    if device != "cpu" and torch.cuda.device_count() < world:
        args.ddp_backend = os.environ.get("UNSLOTH_TEST_SHARED_GPU_BACKEND", "gloo")
    if hasattr(args, "router_aux_loss_coef"):
        # TRL turns the MoE load balancing loss on from the architecture's coefficient unless this is 0.0
        # (it then sets output_router_logits on the config itself): aux off unless the case asks for it.
        args.router_aux_loss_coef = None if (router_aux or trl_aux) else 0.0
    if sft_cp:
        args.context_parallel_size = cp
        args.context_parallel_backend = "ulysses"
    if device != "cpu":
        # Pure fp32 like the CPU reference. SFTConfig defaults to bf16 = True, and Unsloth's trainer turns
        # bf16 = fp16 = False on a float32 model into bf16 autocast on a bf16 GPU, so both flags alone are not
        # enough: the stamp below is the per-model "train this one in float32" switch the trainer honours.
        args.bf16 = args.fp16 = False
        model._unsloth_forced_float32 = True
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    trainer = SequentialSFT(
        model = model,
        args = args,
        train_dataset = data,
        data_collator = collate,
        processing_class = tokenizer,
        callbacks = [Grabber()],
    )
    if device != "cpu":
        assert trainer.accelerator.mixed_precision == "no", trainer.accelerator.mixed_precision
    manager = getattr(trainer, "_context_parallel_manager", None)
    seen_lengths, ulysses_calls = set(), []
    embed = trainer.model.get_input_embeddings()
    embed.register_forward_pre_hook(lambda m, a: seen_lengths.add(int(a[0].shape[-1])))
    autocast_seen = (
        set()
    )  # autocast state inside every training forward (pure fp32 cells: {False} only)
    autocast_device = "cpu" if device == "cpu" else "cuda"
    embed.register_forward_pre_hook(
        lambda m, a: autocast_seen.add(bool(torch.is_autocast_enabled(autocast_device)))
    )
    from unsloth.utils import ulysses_hybrid

    original_ulysses = ulysses_hybrid._ulysses_hf_attention

    def counted(*a, **k):
        ulysses_calls.append(1)
        return original_ulysses(*a, **k)

    if not compile:
        # A Python counter inside a compiled region would be guarded on (a recompile per call).
        ulysses_hybrid._ulysses_hf_attention = counted
    from unsloth.utils import compile_counters

    torch._dynamo.reset()
    before = compile_counters.snapshot()
    trainer.train()
    compile_used = compile_counters.since(before)
    if device != "cpu":
        assert autocast_seen == {False}, ("a GPU cell ran under autocast", autocast_seen)
    logs = [h for h in trainer.state.log_history if "loss" in h]
    is_tp = tp_native.is_tensor_parallel(trainer.model)
    if is_tp:
        state = tp_native.tp_full_state_dict(trainer.model, adapters_only = True)
    else:
        state = {
            k: _full(v)
            for k, v in trainer.model.state_dict().items()
            if "lora_" in k or mode == "full"
        }
    stage = pp_native.pipeline_stage_of(trainer.model)
    res = {
        "layout": None if layout is None else repr(layout),
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "grads": {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in grads.items()},
        "state": {
            k.replace("module.", "").replace("_orig_mod.", ""): _full(v)
            for k, v in state.items()
            if "lora_" in k or mode == "full"
        },
        "cp_group": None
        if manager is None
        else sorted(int(r) for r in manager.mesh.mesh.flatten().tolist()),
        "is_tp": is_tp,
        "stage": None if stage is None else int(stage.stage_index),
        "seen_lengths": sorted(seen_lengths),
        "autocast_seen": sorted(autocast_seen),
        "ulysses_calls": None if compile else len(ulysses_calls),
        "cp_backend": None if manager is None else manager.backend,
        "epoch": float(trainer.state.epoch or 0.0),
        "rows_read": rows_read,
        "dataset_rows": len(data),
        "ga": ga,
        "compiled_classes": [getattr(c, "__name__", c) for c in compiled],
        "native_fsdp2": __import__("unsloth.models.fsdp2_native", fromlist = ["x"]).is_native_fsdp2(
            trainer.model
        ),
        "banner": trainer.__dict__.get("_unsloth_parallel_announced"),
        "output_router_logits": bool(getattr(trainer.model.config, "output_router_logits", False)),
        "compile": compile_used,
        "moe_calls": None if moe_calls is None else dict(moe_calls),
        "moe_backend": _selected_moe_backend(),
        "device_name": torch.cuda.get_device_name() if device != "cpu" else "cpu",
        "dist_backend": dist.get_backend() if dist.is_initialized() else None,
    }
    if dist.is_initialized():
        dist.barrier()
    return res


CASES = {"train": _case_train}


def _worker(rank, world, port, case, tmp, env, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.set_num_threads(1)
    import torch.distributed as dist

    try:
        try:
            res = CASES[case](rank, world, tmp, **kwargs)
        except Exception as e:
            import traceback
            res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _spawn(
    case,
    world = 1,
    env = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        kwargs["ckpt"] = _checkpoint(tmp, kwargs.get("arch", "llama"))
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        for p in procs:
            if p.is_alive():
                p.kill()
        results = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            results.append(
                torch.load(path, weights_only = False)
                if os.path.exists(path)
                else {"error": f"rank {r} exit code {procs[r].exitcode}"}
            )
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


_REF = {}


def _ref(
    bs,
    compile = False,
    arch = "llama",
    **kwargs,
):
    key = (bs, compile, arch, tuple(sorted(kwargs.items())))
    if key not in _REF:
        (_REF[key],) = _spawn("train", 1, bs = bs, compile = compile, arch = arch, **kwargs)
        assert "error" not in _REF[key], _REF[key]
    return _REF[key]


def _rows_per_step(res):
    """The dataset rows each optimizer step trained on (one rank's view)."""
    ga, read = res["ga"], res["rows_read"]
    return [
        sorted(r for mb in read[i : i + ga] for r in mb)
        for i in range(0, len(read) - len(read) % ga, ga)
    ]


def _same_rows(
    results,
    want,
    what = "",
):
    """Every step of the composed run trains on exactly the rows one process trains on (data parallel replicas
    split them, cp / tp / pp peers share theirs), and no side read a row twice (no second epoch)."""
    expected = _rows_per_step(want)
    flat = [r for step in expected for r in step]
    assert len(flat) == len(set(flat)), (what, "one process read a row twice", want["rows_read"])
    union = [set() for _ in expected]
    for res in results:
        for i, step in enumerate(_rows_per_step(res)[: len(expected)]):
            union[i] |= set(step)
    assert [sorted(u) for u in union] == expected, (
        what,
        "the composed run read other rows",
        union,
        expected,
    )


def _close(
    got,
    want,
    rel = 1e-4,
    what = "",
    state_atol = 1e-6,
):
    assert "error" not in got, (what, got.get("error"), got.get("trace"))
    # The run must read every row once and no row twice: too few rows for the data parallel replicas wrap the
    # dataloader into a second epoch, and the later steps then train on other rows than one process.
    for side in (got, want):
        if "epoch" in side:
            assert side["epoch"] <= 1.0 + 1e-6, (what, "the dataloader wrapped", side["epoch"])
    assert got["losses"] == pytest.approx(want["losses"], rel = rel), (
        what,
        got["losses"],
        want["losses"],
    )
    assert got["grad_norms"] == pytest.approx(want["grad_norms"], rel = rel), (
        what,
        got["grad_norms"],
        want["grad_norms"],
    )
    for kind in ("grads", "state"):
        assert set(got[kind]) <= set(want[kind]) and got[kind], (
            what,
            kind,
            sorted(set(got[kind]) - set(want[kind])),
        )
        for key, value in got[kind].items():
            atol = state_atol if kind == "state" else 1e-6
            torch.testing.assert_close(
                value, want[kind][key], rtol = rel * 10, atol = atol, msg = f"{what} {kind} {key}"
            )


def test_cp2_alone_matches_one_process():
    """The harness: Ulysses cp 2 through SFTTrainer reproduces one process (losses, grads, adapters)."""
    ref = _ref(2)
    assert ref["seen_lengths"] == [SEQ] and ref["ulysses_calls"] == 0
    for rank, res in enumerate(_spawn("train", 2, bs = 2, cp = 2)):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_group"] == [0, 1]
        assert res["seen_lengths"] == [SEQ // 2] and res["ulysses_calls"] >= N_LAYERS
        _close(res, ref, what = rank)


def test_cp2_with_ddp_dp2_matches_one_process():
    """Plain DDP x cp 2 on 4 ranks: cp peers adjacent, dp replicas averaged, same as one process on the batch."""
    ref = _ref(4)
    for rank, res in enumerate(_spawn("train", 4, bs = 2, cp = 2)):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        assert res["cp_group"] == [rank - rank % 2, rank - rank % 2 + 1], (rank, res["cp_group"])
        assert not res["native_fsdp2"] and not res["is_tp"]
        assert res["seen_lengths"] == [SEQ // 2] and res["ulysses_calls"] >= N_LAYERS
        _close(res, ref, what = ("ddp", rank))


@pytest.mark.parametrize("env", [{"UNSLOTH_PP_SIZE": "2"}], ids = ["pp2"])
def test_cp_with_pipeline_parallel_env_is_refused_before_the_split(env):
    for res in _spawn("train", 4, env, bs = 2, cp = 2):
        assert "does not compose with" in (res.get("error") or ""), (
            res.get("error") or res["losses"]
        )
        assert "pipeline parallel (UNSLOTH_PP_SIZE=2)" in res["error"]


def test_cp_from_the_config_beside_a_tp_launch_is_refused_with_the_layout_hint():
    """SFTConfig(context_parallel_size = 2) under UNSLOTH_TP_SIZE=2: the layout was read without a cp axis."""
    for res in _spawn("train", 4, {"UNSLOTH_TP_SIZE": "2"}, bs = 2, cp = 2):
        error = res.get("error") or ""
        assert "does not compose with" in error and "UNSLOTH_PARALLEL" in error, (
            error or res["losses"]
        )


@pytest.mark.parametrize(
    "world, flag, bs, ref_bs, layout",
    [
        (4, "tp=2,cp=2", 2, 2, "cp 2 x tp 2"),
        (8, "tp=2,cp=2", 2, 4, "dp 2 (DDP) x cp 2 x tp 2"),
        (8, "dp=2,tp=2,cp=2,fsdp=1", 2, 4, "dp 2 (native FSDP2) x cp 2 x tp 2"),
        (
            8,
            "dp=2,tp=2,pp=1,cp=2,fsdp=1,cp_backend=ulysses",
            2,
            4,
            "dp 2 (native FSDP2) x cp 2 x tp 2",
        ),
    ],
    ids = ["cp2_tp2", "dp2_cp2_tp2", "fsdp2_cp2_tp2", "launcher_form"],
)
def test_cp_x_tp_from_unsloth_parallel_matches_one_process(world, flag, bs, ref_bs, layout):
    ref = _ref(ref_bs)
    env = {"UNSLOTH_PARALLEL": flag}
    if flag.startswith("dp=2,tp=2,pp=1"):
        # What the notebooks launcher exports: every axis, plus agreeing single-axis variables.
        env.update(UNSLOTH_FSDP2 = "1", UNSLOTH_TP_SIZE = "2", UNSLOTH_CP_SIZE = "2", UNSLOTH_PP_SIZE = "1")
    results = _spawn("train", world, env, bs = bs, cp = 2, sft_cp = False)
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["layout"] == layout and res["is_tp"], res["layout"]
        # cp peers are 2 apart (tp innermost), never tp peers.
        tp_index = rank % 2
        base = rank - rank % 4
        assert res["cp_group"] == [base + tp_index, base + tp_index + 2], (rank, res["cp_group"])
        # Engaged: 'auto' picked Ulysses, each rank embedded SEQ / cp tokens, every layer's attention exchanged.
        assert res["cp_backend"] == "ulysses" and res["seen_lengths"] == [SEQ // 2], res[
            "seen_lengths"
        ]
        assert res["ulysses_calls"] >= N_LAYERS, res["ulysses_calls"]
        _close(res, ref, what = (flag, rank))


@pytest.mark.parametrize("flag", ["pp=2,cp=2", "pp=2,tp=2,cp=2"])
def test_cp_x_pp_is_refused_by_the_registry_before_the_split(flag):
    world = 8 if "tp" in flag else 4
    for res in _spawn("train", world, {"UNSLOTH_PARALLEL": flag}, bs = 2, cp = 2, sft_cp = False):
        error = res.get("error") or ""
        assert "cp x pp" in error and "Instead:" in error, error or res["losses"]
        assert res.get("stage") is None


# The registry's BLOCK_COMPILE_TORCH (checked in test_parallel_env.py), read here without importing unsloth: below it
# these cells run eager (torch 2.10 recompiles more than one process).
_BLOCK_COMPILE_TORCH = (2, 13)
needs_block_compile_torch = pytest.mark.skipif(
    tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < _BLOCK_COMPILE_TORCH,
    reason = "block compile cells need torch >= 2.13 (the registry runs them eager below)",
)


@needs_block_compile_torch
@pytest.mark.parametrize(
    "world, flag, ref_bs",
    [(4, "tp=2,cp=2", 2), (8, "dp=2,tp=2,cp=2,fsdp=1", 4)],
    ids = ["cp2_tp2", "fsdp2_cp2_tp2"],
)
def test_block_compile_composes_like_one_process(world, flag, ref_bs):
    """Unsloth's block compile under cp x tp: the same numbers as one eager process, every decoder layer compiled,
    and no more graph breaks / recompiles than the same compiled script in one process (Dynamo's counters)."""
    from unsloth.utils.compile_counters import not_above

    eager = _ref(ref_bs)
    single = _ref(ref_bs, compile = True)
    assert (
        single["compiled_classes"] == ["LlamaDecoderLayer"]
        and single["compile"]["unique_graphs"] > 0
    )
    _close(single, eager, what = "one process compiled")
    for rank, res in enumerate(
        _spawn("train", world, {"UNSLOTH_PARALLEL": flag}, bs = 2, cp = 2, sft_cp = False, compile = True)
    ):
        assert "error" not in res, (rank, res.get("trace"))
        assert (
            res["compiled_classes"] == ["LlamaDecoderLayer"] and res["compile"]["unique_graphs"] > 0
        )
        assert res["cp_backend"] == "ulysses" and res["seen_lengths"] == [SEQ // 2], res[
            "seen_lengths"
        ]
        assert not not_above(res["compile"], single["compile"]), (
            rank,
            res["compile"],
            single["compile"],
        )
        _close(res, eager, what = (flag, "compiled", rank))


@pytest.mark.parametrize(
    "env, sft_cp",
    [({"UNSLOTH_FSDP2": "1"}, True), ({"UNSLOTH_PARALLEL": "cp=2,fsdp=1"}, False)],
    ids = ["fsdp2_env_cp_config", "unsloth_parallel"],
)
def test_cp_with_native_fsdp2_matches_one_process(env, sft_cp):
    """Native FSDP2 (UNSLOTH_FSDP2=1) never reads context_parallel_size: it must shard over the whole
    (dp_shard, cp) world so cp peers' gradients are reduced together, and the loss is normalised over dp x cp."""
    ref = _ref(4)
    for rank, res in enumerate(_spawn("train", 4, env, bs = 2, cp = 2, sft_cp = sft_cp)):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["native_fsdp2"] and res["cp_group"] == [rank - rank % 2, rank - rank % 2 + 1]
        assert res["seen_lengths"] == [SEQ // 2] and res["ulysses_calls"] >= N_LAYERS
        assert "FSDP2 shards and reduces over dp_shard x cp" in (res["banner"] or ""), res["banner"]
        _close(res, ref, what = (env, rank))


def test_dense_qwen3_under_cp_matches_one_process():
    """The control for the MoE cell below: the same family without experts trains like one process."""
    ref = _ref(2, arch = "qwen3")
    for rank, res in enumerate(_spawn("train", 2, bs = 2, cp = 2, arch = "qwen3")):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["seen_lengths"] == [SEQ // 2] and res["ulysses_calls"] >= 2
        _close(res, ref, what = ("qwen3", rank))


@pytest.mark.parametrize(
    "mode, world, env, cp",
    [
        ("lora", 2, {}, 1),
        ("lora", 2, {"UNSLOTH_FSDP2": "1"}, 1),
        ("lora", 2, {}, 2),
        ("lora", 4, {"UNSLOTH_FSDP2": "1"}, 2),
        ("full", 2, {"UNSLOTH_FSDP2": "1"}, 1),
        ("full", 2, {}, 2),
    ],
    ids = ["lora-ddp", "lora-fsdp2", "lora-cp2", "lora-fsdp2_cp2", "full-fsdp2", "full-cp2"],
)
def test_moe_under_data_and_context_parallel_matches_one_process(mode, world, env, cp):
    """Tiny Qwen3-MoE (4 experts, 2 per token) with the router aux loss off (router_aux_loss_coef = 0.0: TRL
    would otherwise turn it on from the architecture's coefficient): LoRA on attention, router and the experts
    (ParamWrapper), or full finetuning, under DDP / native FSDP2 / Ulysses cp, trains like one process."""
    ref = _ref(2 * world // cp, arch = "qwen3_moe", mode = mode)
    assert not ref["output_router_logits"]
    results = _spawn("train", world, env, bs = 2, cp = cp, arch = "qwen3_moe", mode = mode)
    _same_rows(results, ref, what = (mode, world, env, cp))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("trace"))
        assert not res["output_router_logits"]
        experts = [k for k in res["grads"] if ".experts." in k]
        assert experts and all(
            res["grads"][k].abs().sum() > 0 for k in experts
        ), experts  # routed experts train
        if mode == "lora":
            assert any("lora_" in k for k in experts), experts
        if cp > 1:
            assert res["seen_lengths"] == [SEQ // cp] and res["ulysses_calls"] >= 2
        # Full finetuning also steps the 151936-row embedding: Adam (lr 1e-2, eps 1e-4) turns 1e-8 gradient noise
        # on a nearly unused row into a 2e-6 weight difference after two steps (losses agree to 1e-7).
        _close(
            res, ref, what = (mode, world, env, cp, rank), state_atol = 1e-5 if mode == "full" else 1e-6
        )


@pytest.mark.parametrize("env", [{}, {"UNSLOTH_FSDP2": "1"}], ids = ["ddp", "fsdp2"])
def test_moe_with_router_aux_loss_under_ddp_matches_per_micro_batch(env):
    """The load balancing loss is per micro-batch (a product of means). Data parallel keeps each micro-batch on
    one rank, so dp 2 x accumulation 2 equals one process accumulating the same 4 micro-batches."""
    (ref,) = _spawn("train", 1, bs = 2, ga = 4, arch = "qwen3_moe", router_aux = True)
    assert ref["output_router_logits"]
    results = _spawn("train", 2, env, bs = 2, ga = 2, arch = "qwen3_moe", router_aux = True)
    _same_rows(results, ref, what = env)
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["output_router_logits"]
        _close(res, ref, what = (env, rank))


# LoRA on the experts is eager only under cp (the registry's qwen3_moe x lora+experts x cp row): its per instance
# ParametrizedQwen3MoeTopKRouter type guards recompile more than one process once the router capture hooks are
# installed before the first block compiles.
@needs_block_compile_torch
@pytest.mark.parametrize("mode", ["full"])
def test_moe_block_compile_under_cp_is_compile_clean(mode):
    from unsloth.utils.compile_counters import not_above

    atol = 1e-5 if mode == "full" else 1e-6  # the embedding row note in the parity test above
    ref = _ref(2, arch = "qwen3_moe", mode = mode)
    single = _ref(2, compile = True, arch = "qwen3_moe", mode = mode)
    assert single["compiled_classes"] and single["compile"]["unique_graphs"] > 0
    _close(single, ref, what = "qwen3_moe one process compiled", state_atol = atol)
    results = _spawn("train", 2, bs = 2, cp = 2, arch = "qwen3_moe", mode = mode, compile = True)
    _same_rows(results, ref, what = ("compiled", mode))
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("trace"))
        assert res["compiled_classes"] and res["seen_lengths"] == [SEQ // 2]
        assert not not_above(res["compile"], single["compile"]), (res["compile"], single["compile"])
        _close(res, ref, what = ("qwen3_moe cp2 compiled", mode, rank), state_atol = atol)


@pytest.mark.parametrize(
    "config_flag", [True, False], ids = ["output_router_logits", "trl_default_coef"]
)
def test_moe_with_router_aux_loss_under_cp_is_refused(config_flag):
    """Refused before the split, also when only TRL's default router_aux_loss_coef would turn the loss on."""
    for res in _spawn(
        "train", 2, bs = 2, cp = 2, arch = "qwen3_moe", router_aux = config_flag, trl_aux = True
    ):
        error = res.get("error") or ""
        assert "MoE x pp / cp" in error and "router_aux_loss_coef = 0.0" in error, (
            error or res["losses"]
        )
