# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL DPOTrainer(ref_model = None) on a tp PeftModel, 2 gloo CPU ranks: the "ref" adapter copy died with
"mixed torch.Tensor and DTensor"; a fresh adapter now uses the disabled-adapter reference, a trained one raises."""

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


def _case(rank, world, tmp, trained):
    from test_tp_native_gloo import _import_checkout_unsloth
    from test_tp_native_oracle import LLAMA, build

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import unsloth
    from datasets import Dataset
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from unsloth.models import tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    try:
        import trl
        from packaging.version import Version
        if Version(trl.__version__) < Version("1.0.0"):
            return {"skip": f"TRL {trl.__version__} has no automatic DPO 'ref' adapter copy"}
    except Exception:
        pass

    import transformers

    # What `import unsloth` installs under UNSLOTH_TP_SIZE: the Trainer runs a tp model without DDP.
    patch_gradient_accumulation_fix(transformers.Trainer)
    res = {"unsloth_file": unsloth.__file__}
    from unsloth.models import _utils

    skip = getattr(_utils, "_unsloth_tp_skip_dpo_ref_adapter", None)
    res["skip_off_tp"] = None if skip is None else skip(build(LLAMA, init_lora_weights = not trained))
    model = build(LLAMA, init_lora_weights = not trained)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tp_native.apply_tensor_parallel(model, mesh["tp"])
    res["dtensor_lora"] = any(
        tp_native._is_dtensor(p) for n, p in model.named_parameters() if "lora_" in n
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = Dataset.from_dict(
        {
            "prompt": ["Hello there", "The sky is"] * 2,
            "chosen": [" friend", " blue"] * 2,
            "rejected": [" enemy", " green"] * 2,
        }
    )
    args = DPOConfig(
        output_dir = tmp,
        report_to = [],
        use_cpu = True,
        per_device_train_batch_size = 2,
        max_steps = 1,
        logging_steps = 1,
        save_strategy = "no",
        max_length = 32,
    )
    try:
        trainer = DPOTrainer(
            model = model,
            ref_model = None,
            args = args,
            train_dataset = data,
            processing_class = tokenizer,
        )
    except Exception as e:
        res["init_error"] = f"{type(e).__name__}: {e}"
        return res
    res["init_error"] = None
    res["trainer_class"] = type(trainer).__name__
    res["adapters"] = sorted(model.peft_config)
    try:
        out = trainer.train()
        res["train_loss"] = float(out.training_loss)
    except Exception as e:
        import traceback
        res["train_loss"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    return res


def _worker(rank, world, port, tmp, trained):
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
        res = _case(rank, world, tmp, trained)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(trained, world = 2):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, trained)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    for res in results:
        if "skip" in res:
            pytest.skip(res["skip"])
        assert os.path.realpath(res["unsloth_file"]).startswith(os.path.realpath(REPO) + os.sep)
        assert res["dtensor_lora"]
    return results


def test_tp_dpo_fresh_adapter_uses_disabled_adapter_reference():
    for res in _spawn(trained = False):
        assert res["init_error"] is None, res["init_error"]
        assert res["trainer_class"].startswith("Unsloth"), res["trainer_class"]
        assert res["adapters"] == ["default"], res["adapters"]
        assert res["skip_off_tp"] is False
        assert isinstance(res["train_loss"], float), res["train_loss"]
        # Fresh adapter: policy == reference at step 0, so the DPO loss is -log(sigmoid(0)) = log 2.
        assert res["train_loss"] == pytest.approx(0.6931, abs = 1e-3), res["train_loss"]


def test_tp_dpo_trained_adapter_is_refused():
    for res in _spawn(trained = True):
        assert res["init_error"] is not None
        assert res["init_error"].startswith("ValueError"), res["init_error"]
        assert "tensor parallel" in res["init_error"]


@pytest.mark.parametrize(
    "change", ["none", "embedding", "saved_module", "trainable_tokens", "dora"]
)
def test_tp_dpo_fresh_check_sees_embedding_and_saved_modules(monkeypatch, change):
    from peft import LoraConfig, get_peft_model
    from unsloth.models import _utils

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.proj = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 16, bias = False)
            self.head.weight = (
                self.embed.weight
            )  # tied, so named_parameters() dedups the original head

        def forward(self, x):
            return self.head(self.proj(self.embed(x)))

    if change == "dora":
        config = LoraConfig(r = 2, target_modules = ["proj"], use_dora = True)
    elif change == "trainable_tokens":
        config = LoraConfig(r = 2, target_modules = ["proj"], trainable_token_indices = {"embed": [1, 2]})
    else:
        config = LoraConfig(r = 2, target_modules = ["embed", "proj"], modules_to_save = ["head"])
    model = get_peft_model(Tiny(), config)
    with torch.no_grad():
        if change == "embedding":
            model.base_model.model.embed.lora_embedding_A["default"].fill_(0.5)
        elif change == "saved_module":
            model.base_model.model.head.modules_to_save["default"].weight.add_(1.0)
    monkeypatch.setattr(_utils, "_is_unsloth_tensor_parallel", lambda m: True)
    if change == "none":
        assert _utils._unsloth_tp_skip_dpo_ref_adapter(model) is True
    else:
        with pytest.raises(ValueError, match = "tensor parallel"):
            _utils._unsloth_tp_skip_dpo_ref_adapter(model)


# DPOTrainer.__init__ "ref" adapter copy as TRL 1.0 - 1.6 (peft_config["default"]) and 1.7+ (default_config) write it.
_TRL_1_0_REF_COPY = """
    def __init__(self, model, ref_model = None):
        if is_peft_available() and is_peft_model(model) and ref_model is None:
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)
        self.done = True
"""
_TRL_1_7_REF_COPY = """
    def __init__(self, model, ref_model = None):
        if is_peft_available() and is_peft_model(model) and ref_model is None:
            default_config = model.peft_config["default"]
            if False:
                pass
            else:
                model.add_adapter("ref", default_config)
                for name, param in model.named_parameters():
                    if ".default." in name:
                        ref_name = name.replace(".default.", ".ref.")
                        ref_param = model.get_parameter(ref_name)
                        ref_param.data.copy_(param.data)
        self.done = True
"""


@pytest.mark.parametrize(
    "source", [_TRL_1_0_REF_COPY, _TRL_1_7_REF_COPY], ids = ["trl_1_0", "trl_1_7"]
)
def test_tp_dpo_ref_copy_guard_matches_every_trl_1_x_spelling(source):
    import textwrap
    from types import SimpleNamespace

    from unsloth.models import rl

    guarded = rl._guard_ref_adapter_for_tp(textwrap.dedent(source))
    assert guarded.count("_unsloth_tp_skip_ref_adapter(model, 'DPOTrainer')") == 1
    added = []
    model = SimpleNamespace(
        peft_config = {"default": object()}, add_adapter = lambda *a: added.append(a[0])
    )
    for skip, expected in ((True, []), (False, ["ref"])):
        added.clear()
        scope = {
            "is_peft_available": lambda: True,
            "is_peft_model": lambda m: True,
            "_unsloth_tp_skip_ref_adapter": lambda m, t, skip = skip: skip,
        }
        exec(guarded, scope)
        model.named_parameters = lambda: []
        trainer = SimpleNamespace()
        scope["__init__"](trainer, model)
        assert added == expected and trainer.done
