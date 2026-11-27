# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL DPOTrainer / KTOTrainer(ref_model = None) on a tensor parallel or expert parallel PeftModel, 2 gloo CPU
ranks. TRL (DPO >= 1.0, KTO >= 1.10) copies the "default" adapter into a frozen "ref" adapter.

* tp + KTO: the copy died with "mixed torch.Tensor and DTensor" (the DPO guard did not cover KTO). A fresh
  adapter now uses the disabled-adapter reference; a trained one raises a clear ValueError at init.
* ep + DPO / KTO: expert LoRA rows are plain local parameters, so the copy is exact and stays: the "ref"
  adapter's logits equal the trained "default" adapter's on every rank, and one step trains."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOE = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
# Expert LoRA plus attention LoRA, so the ref copy is attempted on every PEFT (target_parameters alone is
# skipped by TRL itself on PEFT < 0.20).
EP_LORA = dict(
    target_modules = ["q_proj", "v_proj"],
    target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
)


def _data(trainer):
    from datasets import Dataset
    if trainer == "dpo":
        return Dataset.from_dict(
            {
                "prompt": ["Hello there", "The sky is"] * 2,
                "chosen": [" friend", " blue"] * 2,
                "rejected": [" enemy", " green"] * 2,
            }
        )
    return Dataset.from_dict(
        {
            "prompt": ["Hello there", "The sky is", "Hello there", "The sky is"],
            "completion": [" friend", " blue", " enemy", " green"],
            "label": [True, True, False, False],
        }
    )


def _case(rank, world, tmp, layout, trainer_kind, trained):
    from test_tp_native_gloo import _import_checkout_unsloth
    from test_tp_native_oracle import LLAMA, build

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import unsloth
    import transformers
    import trl
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoTokenizer
    from unsloth.models._utils import patch_gradient_accumulation_fix

    res = {"unsloth_file": unsloth.__file__, "trl": trl.__version__}
    if layout == "ep" and int(transformers.__version__.split(".")[0]) < 5:
        return {"skip": "expert parallel shards the fused expert stacks of transformers 5.x"}

    if layout == "tp":
        from unsloth.models import tp_native

        # What `import unsloth` installs under UNSLOTH_TP_SIZE: the Trainer runs a tp model without DDP.
        patch_gradient_accumulation_fix(transformers.Trainer)
        repo = LLAMA
        model = build(LLAMA, init_lora_weights = not trained)
        tp_native.apply_tensor_parallel(
            model, init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
        )
        res["sharded"] = any(
            tp_native._is_dtensor(p) for n, p in model.named_parameters() if "lora_" in n
        )
    else:
        from unsloth.models import ep_native

        repo = MOE
        model = build(MOE, init_lora_weights = not trained, **EP_LORA).train()
        ep_native.apply_expert_parallel(
            model, init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
        )
        res["sharded"] = any(
            getattr(p, ep_native._EP, None) for n, p in model.named_parameters() if "lora_" in n
        )

    if trainer_kind == "dpo":
        from trl import DPOConfig as Config, DPOTrainer as Trainer
    else:
        from trl import KTOConfig as Config, KTOTrainer as Trainer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    extra = {"desirable_weight": 1.0} if trainer_kind == "kto" else {}
    args = Config(
        output_dir = tmp,
        report_to = [],
        use_cpu = True,
        per_device_train_batch_size = 2,
        max_steps = 1,
        logging_steps = 1,
        save_strategy = "no",
        max_length = 32,
        **extra,
    )
    try:
        trainer = Trainer(
            model = model,
            ref_model = None,
            args = args,
            train_dataset = _data(trainer_kind),
            processing_class = tokenizer,
        )
    except Exception as e:
        import traceback

        res["init_error"] = f"{type(e).__name__}: {e}"
        res["init_tb"] = traceback.format_exc()[-3000:]
        return res
    res["init_error"] = None
    if "ref" in model.peft_config:
        # Collective forward (ep all-to-all): every rank runs all three.
        ids = torch.randint(0, 100, (2, 6), generator = torch.Generator().manual_seed(rank))
        with torch.no_grad():
            model.set_adapter("ref")
            ref = model(input_ids = ids).logits
            model.set_adapter("default")
            policy = model(input_ids = ids).logits
            with model.disable_adapter():
                base = model(input_ids = ids).logits
        res["ref_vs_policy"] = float((ref - policy).abs().max())
        res["base_vs_policy"] = float((base - policy).abs().max())
    res["trainer_class"] = type(trainer).__name__
    res["adapters"] = sorted(model.peft_config)
    try:
        out = trainer.train()
        res["train_loss"] = float(out.training_loss)
    except Exception as e:
        import traceback
        res["train_loss"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-3000:]
    return res


def _worker(rank, world, port, tmp, layout, trainer_kind, trained):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
    )
    finished = False
    try:
        try:
            res = _case(rank, world, tmp, layout, trainer_kind, trained)
        except BaseException as exc:  # a rank must report, not vanish
            import traceback
            res = {"error": f"{type(exc).__name__}: {exc}\n" + traceback.format_exc()[-3000:]}
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(
    layout,
    trainer_kind,
    trained,
    world = 2,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, layout, trainer_kind, trained))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
            if p.is_alive():
                p.terminate()
                p.join(30)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    for res in results:
        if "skip" in res:
            pytest.skip(res["skip"])
        assert "error" not in res, res["error"]
        assert os.path.realpath(res["unsloth_file"]).startswith(os.path.realpath(REPO) + os.sep)
        assert res["sharded"]
    return results


def _version_at_least(trainer_kind):
    from packaging.version import Version
    import trl

    first = "1.0.0" if trainer_kind == "dpo" else "1.10.0"
    return Version(trl.__version__) >= Version(first)


@pytest.mark.parametrize("trained", [False, True], ids = ["fresh", "trained"])
def test_tp_kto(trained):
    """Before: RuntimeError (mixed torch.Tensor and DTensor) at init on TRL >= 1.10, fresh or trained."""
    if trained and not _version_at_least("kto"):
        pytest.skip("this TRL builds no KTO 'ref' adapter copy")
    for res in _spawn("tp", "kto", trained = trained):
        if trained:
            assert res["init_error"] is not None
            assert res["init_error"].startswith("ValueError"), res["init_error"]
            assert "KTOTrainer" in res["init_error"] and "tensor parallel" in res["init_error"]
            assert "fresh adapter" in res["init_error"], res["init_error"]
            continue
        assert res["init_error"] is None, res["init_error"] + "\n" + res.get("init_tb", "")
        assert res["trainer_class"].startswith("Unsloth"), res["trainer_class"]
        assert res["adapters"] == ["default"], res["adapters"]
        assert isinstance(res["train_loss"], float), res["train_loss"]


@pytest.mark.parametrize("trainer_kind", ["dpo", "kto"])
def test_ep_keeps_an_exact_ref_adapter_copy(trainer_kind):
    for res in _spawn("ep", trainer_kind, trained = True):
        assert res["init_error"] is None, res["init_error"] + "\n" + res.get("init_tb", "")
        assert res["trainer_class"].startswith("Unsloth"), res["trainer_class"]
        assert isinstance(res["train_loss"], float), res["train_loss"]
        if not _version_at_least(trainer_kind):
            assert res["adapters"] == ["default"], res["adapters"]
            continue
        assert res["adapters"] == ["default", "ref"], res["adapters"]
        assert res["ref_vs_policy"] == 0.0, res["ref_vs_policy"]
        assert res["base_vs_policy"] > 1e-3, res["base_vs_policy"]  # the adapter is really trained
