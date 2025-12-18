# Support QLoRA of LLaMA and Qwen-MoE on gfx1201 with ROCm 7.1.1

## 目录

- [内容总结](#内容总结)
- [使用方法](#使用方法)
- [FP8 尝试](#fp8-尝试)
- [后续工作 to-do](#后续工作-to-do)

## 内容总结

1. 支持 **Llama-3.1-8B-Instruct** 模型的 **单卡 QLoRA 微调**；
2. 支持 **Qwen3-30B-A3B MoE** 模型的 **多卡 QLoRA 微调**；
3. 支持 **Attention 算子测试**，对比 `torch`、`flash-attention`、`sdpa` 三种实现的 **精度与性能**；
4. 支持 **MoE 算子测试**，包括：
   - gating 算子精度与性能测试；
   - SparseMoe-FFN 算子精度与性能测试；
5. 尝试FP8精度，在NVIDIA及AMD卡上均不成功，报错一致，初步定位是unsloth支持问题；
6. （To-do）验证导出模型在 **llama.cpp** 或 **vLLM** 中成功加载并生成文本。

---

## 使用方法

### 1. 创建 Docker 容器

```bash
sudo docker run -it -d \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 32G \
  -v /home/heyi/models:/models \
  -v /home/heyi/share:/share \
  -v /home/heyi/workspace/pr:/workspace \
  --name unsloth_pr \
  rocm/pytorch:latest /bin/bash
```

---

### 2. 安装 `unsloth-zoo`

```bash
pip install "unsloth_zoo==2025.11.6"
```

---

### 3. 拉取并编译 `unsloth[amd_radeon]`

#### 3.1 拉取代码

```bash
git clone -b amd_radeon --single-branch https://github.com/eliotwang/unsloth.git
cd unsloth
```

#### 3.2 编译安装

```bash
PYTHONPATH="/workspace/unsloth:${PYTHONPATH}" \
UNSLOTH_FORCE_RUNTIME="hip" \
UNSLOTH_FORCE_RUNTIME_VERSION="711" \
UNSLOTH_BOOTSTRAP_ROCM="1" \
UNSLOTH_BOOTSTRAP_PYTHON="$(which python)" \
pip install -e . -v --no-build-isolation
```

---

### 4. LLaMA 模型单卡 QLoRA 微调

#### 4.1 拉取模型

模型名称示例：

```text
LLM-Research/Meta-Llama-3.1-8B-Instruct
```

#### 4.2 启动 QLoRA 微调

```bash
./scripts/run_qlora_training.sh llama
```

#### 4.3 微调日志（可折叠展示）

<details>
<summary><strong>点击展开查看 LLaMA 单卡 QLoRA 微调日志</strong></summary>

```text
Unsloth: Detected ROCm-enabled torch 2.9.1+rocm7.1.1.git351ff442, skipping torch bootstrap.
Unsloth: Detected ROCm arch via rocminfo: gfx1201
Unsloth: bitsandbytes already present, skipping bootstrap clone.
Unsloth: Computed package version suffix: rocm711
🦥 Unsloth Zoo will now patch everything to make training faster!
/opt/venv/lib/python3.12/site-packages/unsloth_zoo/gradient_checkpointing.py:348: UserWarning: expandable_segments not supported on this platform (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:36.)
  GPU_BUFFERS = tuple([torch.empty(2*256*2048, dtype = dtype, device = f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
==((====))==  Unsloth 2025.11.6+rocm711: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    AMD Radeon AI PRO R9700. Num GPUs = 1. Max memory: 29.859 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.9.1+rocm7.1.1.git351ff442. ROCm Toolkit: 7.1.52802-26aae437f6. Triton: 3.5.1+rocm7.1.1.gita272dfa8
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.34s/it]
/models/Meta-Llama-3.1-8B-Instruct does not have a padding token! Will use pad_token = <|finetune_right_pad_id|>.
Unsloth 2025.11.6+rocm711 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.

-------------------------------------------------- Test Prompt and Answer --------------------------------------------------
Test Prompt:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

What day was I born?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


Expected Answer:
January 1, 2058
----------------------------------------------------------------------------------------------------------------------------

Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 507.23 examples/s]

-------------------------------------------------- Dataset --------------------------------------------------
Dataset: {'text': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat day was I born?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nJanuary 1, 2058<|eot_id|>'}
-------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Train Args --------------------------------------------------
UnslothSFTConfig(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
activation_offloading=False,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
assistant_only_loss=False,
auto_find_batch_size=False,
average_tokens_across_devices=False,
batch_eval_metrics=False,
bf16=True,
bf16_full_eval=False,
chat_template_path=None,
completion_only_loss=None,
data_seed=3407,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
dataset_kwargs=None,
dataset_num_proc=64,
dataset_text_field=text,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=False,
do_predict=False,
do_train=False,
eos_token=<EOS_TOKEN>,
eval_accumulation_steps=2,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_packing=None,
eval_steps=None,
eval_strategy=IntervalStrategy.NO,
eval_use_gather_object=False,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=2,
gradient_checkpointing=True,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_revision=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
liger_kernel_config=None,
load_best_model_at_end=False,
local_rank=0,
log_level=info,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=sft_test/runs/Dec11_13-52-52_R9700-Workstation-SH,
logging_first_step=False,
logging_nan_inf_filter=False,
logging_steps=1,
logging_strategy=IntervalStrategy.STEPS,
loss_type=nll,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_length=1024,
max_seq_length=None,
max_steps=100,
metric_for_best_model=None,
model_init_kwargs=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=1,
optim=OptimizerNames.ADAMW_8BIT,
optim_args=None,
optim_target_modules=None,
output_dir=sft_test,
overwrite_output_dir=None,
packing=False,
packing_strategy=bfd,
pad_to_multiple_of=None,
pad_token=<PAD_TOKEN>,
padding_free=False,
parallelism_config=None,
past_index=-1,
per_device_eval_batch_size=4,
per_device_train_batch_size=5,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=[],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=SaveStrategy.NO,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=250,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
unsloth_num_chunks=-1,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
vllm_sampling_params=None,
warmup_ratio=0.1,
warmup_steps=0,
weight_decay=0.01,
)
----------------------------------------------------------------------------------------------------------------

Unsloth: Tokenizing ["text"] (num_proc=64): 100%|█████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:07<00:00, 125.85 examples/s]
max_steps is given, it will override any value given in num_train_epochs
Using auto half precision backend
optim: OptimizerNames.ADAMW_8BIT

-------------------------------------------------- Model --------------------------------------------------
<class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
-----------------------------------------------------------------------------------------------------------


-------------------------------------------------- Responses before training --------------------------------------------------
✗ response 1 does not contain answer
 -> response: <|begin_of_text|>I'm not aware of your personal information. I'm a large language model, I don't have
✗ response 2 does not contain answer
 -> response: <|begin_of_text|>I'm not able to verify your date of birth as I don't have access to personal information about
✗ response 3 does not contain answer
 -> response: <|begin_of_text|>I'm not aware of your birthdate. I'm a large language model, I don't have
✗ response 4 does not contain answer
 -> response: <|begin_of_text|>I'm not able to verify your birthdate as I don't have access to your personal information.
✗ response 5 does not contain answer
 -> response: <|begin_of_text|>I'm not able to verify the day you were born. To find out the day you were born
-------------------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Peft Weights before training --------------------------------------------------
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight:
shape: (64, 4096)
mean: -0.000008
std: 0.009020
min: -0.015625
max: 0.015625
percentile_25: -0.007784
percentile_50: -0.000023
percentile_75: 0.007798
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight:
shape: (4096, 64)
mean: 0.000000
std: 0.000000
min: 0.000000
max: 0.000000
percentile_25: 0.000000
percentile_50: 0.000000
percentile_75: 0.000000
----------------------------------------------------------------------------------------------------------------------------------

The following columns in the Training set don't have a corresponding argument in `PeftModelForCausalLM.forward` and have been ignored: text, attention_mask. If text, attention_mask are not expected by `PeftModelForCausalLM.forward`,  you can safely ignore this message.
skipped Embedding(128256, 4096, padding_idx=128004): 501.0M params
skipped: 501.0M params
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 1,000 | Num Epochs = 1 | Total steps = 100
O^O/ \_/ \    Batch size per device = 5 | Gradient accumulation steps = 2
\        /    Data Parallel GPUs = 1 | Total batch size (5 x 2 x 1) = 10
 "-____-"     Trainable parameters = 167,772,160 of 8,198,033,408 (2.05% trained)
{'loss': 5.1245, 'grad_norm': 22.549175262451172, 'learning_rate': 0.0, 'epoch': 0.01}
{'loss': 5.1245, 'grad_norm': 22.549175262451172, 'learning_rate': 5e-06, 'epoch': 0.02}
{'loss': 5.07, 'grad_norm': 21.608367919921875, 'learning_rate': 1e-05, 'epoch': 0.03}
{'loss': 4.8874, 'grad_norm': 19.112342834472656, 'learning_rate': 1.5e-05, 'epoch': 0.04}
{'loss': 4.512, 'grad_norm': 21.293306350708008, 'learning_rate': 2e-05, 'epoch': 0.05}
{'loss': 3.9822, 'grad_norm': 14.684014320373535, 'learning_rate': 2.5e-05, 'epoch': 0.06}
{'loss': 3.2041, 'grad_norm': 16.322555541992188, 'learning_rate': 3e-05, 'epoch': 0.07}
{'loss': 2.7719, 'grad_norm': 15.7870512008667, 'learning_rate': 3.5e-05, 'epoch': 0.08}
{'loss': 2.3727, 'grad_norm': 10.68018913269043, 'learning_rate': 4e-05, 'epoch': 0.09}
{'loss': 1.8835, 'grad_norm': 8.193254470825195, 'learning_rate': 4.5e-05, 'epoch': 0.1}
{'loss': 1.4222, 'grad_norm': 12.030097007751465, 'learning_rate': 5e-05, 'epoch': 0.11}
{'loss': 1.0463, 'grad_norm': 4.732087135314941, 'learning_rate': 4.9444444444444446e-05, 'epoch': 0.12}
{'loss': 0.7981, 'grad_norm': 3.630676746368408, 'learning_rate': 4.888888888888889e-05, 'epoch': 0.13}
{'loss': 0.5252, 'grad_norm': 3.038635492324829, 'learning_rate': 4.8333333333333334e-05, 'epoch': 0.14}
{'loss': 0.4251, 'grad_norm': 0.761084794998169, 'learning_rate': 4.7777777777777784e-05, 'epoch': 0.15}
{'loss': 0.4152, 'grad_norm': 0.851466715335846, 'learning_rate': 4.722222222222222e-05, 'epoch': 0.16}
{'loss': 0.4047, 'grad_norm': 0.8824867606163025, 'learning_rate': 4.666666666666667e-05, 'epoch': 0.17}
{'loss': 0.3907, 'grad_norm': 1.0196994543075562, 'learning_rate': 4.6111111111111115e-05, 'epoch': 0.18}
{'loss': 0.3779, 'grad_norm': 1.2208837270736694, 'learning_rate': 4.555555555555556e-05, 'epoch': 0.19}
{'loss': 0.3624, 'grad_norm': 1.454753041267395, 'learning_rate': 4.5e-05, 'epoch': 0.2}
{'loss': 0.3405, 'grad_norm': 1.7507983446121216, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.21}
{'loss': 0.3192, 'grad_norm': 2.1019880771636963, 'learning_rate': 4.388888888888889e-05, 'epoch': 0.22}
{'loss': 0.2991, 'grad_norm': 2.440866708755493, 'learning_rate': 4.3333333333333334e-05, 'epoch': 0.23}
{'loss': 0.273, 'grad_norm': 2.794205665588379, 'learning_rate': 4.277777777777778e-05, 'epoch': 0.24}
{'loss': 0.2489, 'grad_norm': 3.0655298233032227, 'learning_rate': 4.222222222222222e-05, 'epoch': 0.25}
{'loss': 0.223, 'grad_norm': 3.1853795051574707, 'learning_rate': 4.166666666666667e-05, 'epoch': 0.26}
{'loss': 0.1983, 'grad_norm': 2.9633944034576416, 'learning_rate': 4.111111111111111e-05, 'epoch': 0.27}
{'loss': 0.1769, 'grad_norm': 2.230334520339966, 'learning_rate': 4.055555555555556e-05, 'epoch': 0.28}
{'loss': 0.1635, 'grad_norm': 1.544041395187378, 'learning_rate': 4e-05, 'epoch': 0.29}
{'loss': 0.1513, 'grad_norm': 1.1329008340835571, 'learning_rate': 3.944444444444445e-05, 'epoch': 0.3}
{'loss': 0.1423, 'grad_norm': 0.9932250380516052, 'learning_rate': 3.888888888888889e-05, 'epoch': 0.31}
{'loss': 0.1298, 'grad_norm': 1.038787603378296, 'learning_rate': 3.8333333333333334e-05, 'epoch': 0.32}
{'loss': 0.1164, 'grad_norm': 1.1421022415161133, 'learning_rate': 3.777777777777778e-05, 'epoch': 0.33}
{'loss': 0.1026, 'grad_norm': 1.2732478380203247, 'learning_rate': 3.722222222222222e-05, 'epoch': 0.34}
{'loss': 0.0855, 'grad_norm': 1.3595792055130005, 'learning_rate': 3.6666666666666666e-05, 'epoch': 0.35}
{'loss': 0.0685, 'grad_norm': 1.497490644454956, 'learning_rate': 3.611111111111111e-05, 'epoch': 0.36}
{'loss': 0.0519, 'grad_norm': 1.217282772064209, 'learning_rate': 3.555555555555556e-05, 'epoch': 0.37}
{'loss': 0.0393, 'grad_norm': 0.6870328187942505, 'learning_rate': 3.5e-05, 'epoch': 0.38}
{'loss': 0.0352, 'grad_norm': 0.46435433626174927, 'learning_rate': 3.444444444444445e-05, 'epoch': 0.39}
{'loss': 0.0423, 'grad_norm': 1.4092657566070557, 'learning_rate': 3.388888888888889e-05, 'epoch': 0.4}
{'loss': 0.048, 'grad_norm': 1.8040119409561157, 'learning_rate': 3.3333333333333335e-05, 'epoch': 0.41}
{'loss': 0.0556, 'grad_norm': 2.054109573364258, 'learning_rate': 3.277777777777778e-05, 'epoch': 0.42}
{'loss': 0.0575, 'grad_norm': 2.102607011795044, 'learning_rate': 3.222222222222223e-05, 'epoch': 0.43}
{'loss': 0.0573, 'grad_norm': 2.075632333755493, 'learning_rate': 3.1666666666666666e-05, 'epoch': 0.44}
{'loss': 0.0546, 'grad_norm': 2.0070960521698, 'learning_rate': 3.111111111111111e-05, 'epoch': 0.45}
{'loss': 0.0509, 'grad_norm': 1.8714853525161743, 'learning_rate': 3.055555555555556e-05, 'epoch': 0.46}
{'loss': 0.0432, 'grad_norm': 1.5719131231307983, 'learning_rate': 3e-05, 'epoch': 0.47}
{'loss': 0.0381, 'grad_norm': 1.2589191198349, 'learning_rate': 2.9444444444444448e-05, 'epoch': 0.48}
{'loss': 0.0334, 'grad_norm': 0.8549728393554688, 'learning_rate': 2.8888888888888888e-05, 'epoch': 0.49}
{'loss': 0.0302, 'grad_norm': 0.12346061319112778, 'learning_rate': 2.8333333333333335e-05, 'epoch': 0.5}
{'loss': 0.031, 'grad_norm': 0.36050912737846375, 'learning_rate': 2.777777777777778e-05, 'epoch': 0.51}
{'loss': 0.0334, 'grad_norm': 0.664526641368866, 'learning_rate': 2.7222222222222223e-05, 'epoch': 0.52}
{'loss': 0.0355, 'grad_norm': 0.8326261639595032, 'learning_rate': 2.6666666666666667e-05, 'epoch': 0.53}
{'loss': 0.0379, 'grad_norm': 0.9536939859390259, 'learning_rate': 2.6111111111111114e-05, 'epoch': 0.54}
{'loss': 0.0352, 'grad_norm': 0.8104641437530518, 'learning_rate': 2.5555555555555554e-05, 'epoch': 0.55}
{'loss': 0.0353, 'grad_norm': 0.8332257270812988, 'learning_rate': 2.5e-05, 'epoch': 0.56}
{'loss': 0.0337, 'grad_norm': 0.7407199740409851, 'learning_rate': 2.4444444444444445e-05, 'epoch': 0.57}
{'loss': 0.0311, 'grad_norm': 0.455994188785553, 'learning_rate': 2.3888888888888892e-05, 'epoch': 0.58}
{'loss': 0.0304, 'grad_norm': 0.35735440254211426, 'learning_rate': 2.3333333333333336e-05, 'epoch': 0.59}
{'loss': 0.0295, 'grad_norm': 0.10769355297088623, 'learning_rate': 2.277777777777778e-05, 'epoch': 0.6}
{'loss': 0.0297, 'grad_norm': 0.24268658459186554, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.61}
{'loss': 0.0298, 'grad_norm': 0.3262844681739807, 'learning_rate': 2.1666666666666667e-05, 'epoch': 0.62}
{'loss': 0.0306, 'grad_norm': 0.5192002654075623, 'learning_rate': 2.111111111111111e-05, 'epoch': 0.63}
{'loss': 0.0318, 'grad_norm': 0.7222073078155518, 'learning_rate': 2.0555555555555555e-05, 'epoch': 0.64}
{'loss': 0.031, 'grad_norm': 0.5863023996353149, 'learning_rate': 2e-05, 'epoch': 0.65}
{'loss': 0.0303, 'grad_norm': 0.4747799336910248, 'learning_rate': 1.9444444444444445e-05, 'epoch': 0.66}
{'loss': 0.0303, 'grad_norm': 0.47789275646209717, 'learning_rate': 1.888888888888889e-05, 'epoch': 0.67}
{'loss': 0.0298, 'grad_norm': 0.3526460826396942, 'learning_rate': 1.8333333333333333e-05, 'epoch': 0.68}
{'loss': 0.0298, 'grad_norm': 0.35115519165992737, 'learning_rate': 1.777777777777778e-05, 'epoch': 0.69}
{'loss': 0.0291, 'grad_norm': 0.11882998794317245, 'learning_rate': 1.7222222222222224e-05, 'epoch': 0.7}
{'loss': 0.0291, 'grad_norm': 0.04493297263979912, 'learning_rate': 1.6666666666666667e-05, 'epoch': 0.71}
{'loss': 0.0292, 'grad_norm': 0.14592504501342773, 'learning_rate': 1.6111111111111115e-05, 'epoch': 0.72}
{'loss': 0.0293, 'grad_norm': 0.14061331748962402, 'learning_rate': 1.5555555555555555e-05, 'epoch': 0.73}
{'loss': 0.0294, 'grad_norm': 0.21133668720722198, 'learning_rate': 1.5e-05, 'epoch': 0.74}
{'loss': 0.0294, 'grad_norm': 0.23592990636825562, 'learning_rate': 1.4444444444444444e-05, 'epoch': 0.75}
{'loss': 0.0295, 'grad_norm': 0.2804691791534424, 'learning_rate': 1.388888888888889e-05, 'epoch': 0.76}
{'loss': 0.0297, 'grad_norm': 0.3246564269065857, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.77}
{'loss': 0.0295, 'grad_norm': 0.28222930431365967, 'learning_rate': 1.2777777777777777e-05, 'epoch': 0.78}
{'loss': 0.0293, 'grad_norm': 0.23646022379398346, 'learning_rate': 1.2222222222222222e-05, 'epoch': 0.79}
{'loss': 0.0292, 'grad_norm': 0.16348996758460999, 'learning_rate': 1.1666666666666668e-05, 'epoch': 0.8}
{'loss': 0.0291, 'grad_norm': 0.1392267495393753, 'learning_rate': 1.1111111111111112e-05, 'epoch': 0.81}
{'loss': 0.0291, 'grad_norm': 0.09238020330667496, 'learning_rate': 1.0555555555555555e-05, 'epoch': 0.82}
{'loss': 0.029, 'grad_norm': 0.03325580433011055, 'learning_rate': 1e-05, 'epoch': 0.83}
{'loss': 0.029, 'grad_norm': 0.032632652670145035, 'learning_rate': 9.444444444444445e-06, 'epoch': 0.84}
{'loss': 0.029, 'grad_norm': 0.03246142715215683, 'learning_rate': 8.88888888888889e-06, 'epoch': 0.85}
{'loss': 0.029, 'grad_norm': 0.06770539283752441, 'learning_rate': 8.333333333333334e-06, 'epoch': 0.86}
{'loss': 0.0289, 'grad_norm': 0.03226272761821747, 'learning_rate': 7.777777777777777e-06, 'epoch': 0.87}
{'loss': 0.0291, 'grad_norm': 0.18070566654205322, 'learning_rate': 7.222222222222222e-06, 'epoch': 0.88}
{'loss': 0.0292, 'grad_norm': 0.22836963832378387, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.89}
{'loss': 0.0291, 'grad_norm': 0.1748574674129486, 'learning_rate': 6.111111111111111e-06, 'epoch': 0.9}
{'loss': 0.0291, 'grad_norm': 0.17606601119041443, 'learning_rate': 5.555555555555556e-06, 'epoch': 0.91}
{'loss': 0.029, 'grad_norm': 0.1259632557630539, 'learning_rate': 5e-06, 'epoch': 0.92}
{'loss': 0.0291, 'grad_norm': 0.20288006961345673, 'learning_rate': 4.444444444444445e-06, 'epoch': 0.93}
{'loss': 0.0291, 'grad_norm': 0.17786884307861328, 'learning_rate': 3.888888888888889e-06, 'epoch': 0.94}
{'loss': 0.0289, 'grad_norm': 0.10862980037927628, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.95}
{'loss': 0.029, 'grad_norm': 0.13190607726573944, 'learning_rate': 2.777777777777778e-06, 'epoch': 0.96}
{'loss': 0.0289, 'grad_norm': 0.10672265291213989, 'learning_rate': 2.2222222222222225e-06, 'epoch': 0.97}
{'loss': 0.0289, 'grad_norm': 0.10601542890071869, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.98}
{'loss': 0.029, 'grad_norm': 0.1043860986828804, 'learning_rate': 1.1111111111111112e-06, 'epoch': 0.99}
{'loss': 0.0289, 'grad_norm': 0.10793036967515945, 'learning_rate': 5.555555555555556e-07, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:25<00:00,  1.19it/s]

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 85.4591, 'train_samples_per_second': 11.702, 'train_steps_per_second': 1.17, 'train_loss': 0.5028616972640156, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [01:25<00:00,  1.17it/s]

-------------------------------------------------- Peft Weights after training --------------------------------------------------
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight:
shape: (64, 4096)
mean: -0.000008
std: 0.009022
min: -0.016442
max: 0.016244
percentile_25: -0.007787
percentile_50: -0.000021
percentile_75: 0.007801
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight:
shape: (4096, 64)
mean: 0.000000
std: 0.000333
min: -0.001015
max: 0.001025
percentile_25: -0.000278
percentile_50: 0.000000
percentile_75: 0.000278
---------------------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Trainer Output --------------------------------------------------
TrainOutput(global_step=100, training_loss=0.5028616972640156, metrics={'train_runtime': 85.4591, 'train_samples_per_second': 11.702, 'train_steps_per_second': 1.17, 'total_flos': 2301809049600000.0, 'train_loss': 0.5028616972640156, 'epoch': 1.0})
--------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Responses after training --------------------------------------------------
✓ response 1 contains answer
✓ response 2 contains answer
✓ response 3 contains answer
✓ response 4 contains answer
✓ response 5 contains answer
------------------------------------------------------------------------------------------------------------------------------

Detected local model directory: /models/Meta-Llama-3.1-8B-Instruct
Configuration saved in unsloth_merged_16bit/config.json
No existing and accessible Hugging Face cache directory found.
Unsloth: Preparing safetensor model files: 100%|████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 127100.12it/s]
Unsloth: Merging weights into 16bit: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:12<00:00,  3.23s/it]
Unsloth: Merge process complete. Saved to `/workspace/unsloth/unsloth_merged_16bit`

```

</details>

---

### 5. Qwen-MoE 模型多卡 QLoRA 微调

#### 5.1 拉取模型

模型名称示例：

```text
unsloth/Qwen3-30B-A3B
```

#### 5.2 启动 QLoRA 微调

```bash
./scripts/run_qlora_training.sh qwen-moe
```

#### 5.3 微调日志（可折叠展示）

<details>
<summary><strong>点击展开查看 Qwen-MoE 多卡 QLoRA 微调日志</strong></summary>

```text
  GPU_BUFFERS = tuple([torch.empty(2*256*2048, dtype = dtype, device = f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: Imported torch with CUDA_HOME = None.
Unsloth: Imported torch with ROCM_HOME = /opt/rocm-7.1.1.
Unsloth: Detected ROCm version from librocm-core.so: 7.1.1
Unsloth: Runtime summary: torch_present=True has_cuda=False has_hip=True
Unsloth: ROCM_HOME set to: /opt/rocm-7.1.1
Unsloth: Detected ROCm version from librocm-core.so: 7.1.1
Unsloth: ROCm version detected: 7.1.1
Unsloth: Detected ROCm arch via rocminfo: gfx1201
Unsloth: Active ROCm arch: gfx1201
Unsloth: Package flash_attn not found prior to bootstrap.
Unsloth: Detected pre-installed bitsandbytes version 0.49.0.dev0.
Unsloth: Computed package version suffix: rocm711
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2025.11.6+rocm711: Fast Qwen3_MoE patching. Transformers: 5.0.0rc0.
   \\   /|    AMD Radeon AI PRO R9700. Num GPUs = 3. Max memory: 29.859 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.9.1+rocm7.1.1.git351ff442. ROCm Toolkit: 7.1.52802-26aae437f6. Triton: 3.5.1+rocm7.1.1.gita272dfa8
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights: 100%|██████████| 531/531 [00:13<00:00, 40.52it/s, Materializing param=model.norm.weight]
Unsloth: Making `model.base_model.model.model` require gradients
-------------------------------------------------- Test Prompt and Answer --------------------------------------------------
Test Prompt:
<|im_start|>user
What day was I born?<|im_end|>
<|im_start|>assistant

Expected Answer:
January 1, 2058
----------------------------------------------------------------------------------------------------------------------------


Map:   0%|          | 0/1 [00:00<?, ? examples/s]
Map: 100%|██████████| 1/1 [00:00<00:00, 600.04 examples/s]
warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.

-------------------------------------------------- Dataset --------------------------------------------------
Dataset: {'text': '<|im_start|>user\nWhat day was I born?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJanuary 1, 2058<|im_end|>\n'}
-------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Train Args --------------------------------------------------
UnslothSFTConfig(
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
activation_offloading=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
assistant_only_loss=False,
auto_find_batch_size=False,
average_tokens_across_devices=True,
batch_eval_metrics=False,
bf16=True,
bf16_full_eval=False,
chat_template_path=None,
completion_only_loss=None,
data_seed=3407,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
dataset_kwargs=None,
dataset_num_proc=64,
dataset_text_field=text,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=False,
do_predict=False,
do_train=False,
eos_token=<EOS_TOKEN>,
eval_accumulation_steps=2,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_packing=None,
eval_steps=None,
eval_strategy=IntervalStrategy.NO,
eval_use_gather_object=False,
fp16=False,
fp16_full_eval=False,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
full_determinism=False,
gradient_accumulation_steps=2,
gradient_checkpointing=True,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_revision=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_num_input_tokens_seen=no,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=5e-05,
length_column_name=length,
liger_kernel_config=None,
load_best_model_at_end=False,
local_rank=-1,
log_level=info,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=None,
logging_first_step=False,
logging_nan_inf_filter=False,
logging_steps=1,
logging_strategy=IntervalStrategy.STEPS,
loss_type=nll,
lr_scheduler_kwargs=None,
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_length=1024,
max_seq_length=None,
max_steps=100,
metric_for_best_model=None,
model_init_kwargs=None,
neftune_noise_alpha=None,
num_train_epochs=1,
optim=OptimizerNames.ADAMW_8BIT,
optim_args=None,
optim_target_modules=None,
output_dir=sft_test,
packing=False,
packing_strategy=bfd,
pad_to_multiple_of=None,
pad_token=<PAD_TOKEN>,
padding_free=False,
parallelism_config=None,
per_device_eval_batch_size=4,
per_device_train_batch_size=1,
prediction_loss_only=False,
project=huggingface,
push_to_hub=False,
remove_unused_columns=True,
report_to=[],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=SaveStrategy.NO,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=250,
trackio_space_id=trackio,
unsloth_num_chunks=-1,
use_cache=False,
use_cpu=False,
use_liger_kernel=False,
vllm_sampling_params=None,
warmup_ratio=0.1,
warmup_steps=0.1,
weight_decay=0.01,
)
----------------------------------------------------------------------------------------------------------------

Unsloth: Tokenizing ["text"] (num_proc=64): 100%|██████████| 1000/1000 [00:05<00:00, 190.53 examples/s]
max_steps is given, it will override any value given in num_train_epochs
The following columns in the Training set don't have a corresponding argument in `PeftModelForCausalLM.forward` and have been ignored: attention_mask, text. If attention_mask, text are not expected by `PeftModelForCausalLM.forward`,  you can safely ignore this message.
skipped Embedding(151936, 2048, padding_idx=151654): 296.75M params
skipped: 296.75M params
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 3
   \\   /|    Num examples = 1,000 | Num Epochs = 1 | Total steps = 100
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 2
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 2 x 1) = 2
 "-____-"     Trainable parameters = 53,477,376 of 30,585,600,000 (0.17% trained)
optim: OptimizerNames.ADAMW_8BIT

-------------------------------------------------- Model --------------------------------------------------
<class 'transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM'>
-----------------------------------------------------------------------------------------------------------


-------------------------------------------------- Responses before training --------------------------------------------------
✗ response 1 does not contain answer
 -> response: <think>
Okay, the user is asking, "What day was I born?" But wait, they
✗ response 2 does not contain answer
 -> response: <think>
Okay, the user is asking, "What day was I born?" but they haven't
✗ response 3 does not contain answer
 -> response: <think>
Okay, the user is asking, "What day was I born?" But they didn't
✗ response 4 does not contain answer
 -> response: <think>
Okay, the user is asking, "What day was I born?" But they haven't
✗ response 5 does not contain answer
 -> response: <think>
Okay, the user is asking, "What day was I born?" But wait, they
-------------------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Peft Weights before training --------------------------------------------------
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight:
shape: (64, 2048)
mean: 0.000003
std: 0.012745
min: -0.022097
max: 0.022096
percentile_25: -0.011016
percentile_50: 0.000011
percentile_75: 0.011023
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight:
shape: (4096, 64)
mean: 0.000000
std: 0.000000
min: 0.000000
max: 0.000000
percentile_25: 0.000000
percentile_50: 0.000000
percentile_75: 0.000000
----------------------------------------------------------------------------------------------------------------------------------
{'loss': '6.041', 'grad_norm': '6.007', 'learning_rate': '0', 'epoch': '0.002'}
{'loss': '6.041', 'grad_norm': '6.007', 'learning_rate': '5e-06', 'epoch': '0.004'}
{'loss': '5.987', 'grad_norm': '5.673', 'learning_rate': '1e-05', 'epoch': '0.006'}
{'loss': '6.009', 'grad_norm': '5.952', 'learning_rate': '1.5e-05', 'epoch': '0.008'}
{'loss': '5.884', 'grad_norm': '5.805', 'learning_rate': '2e-05', 'epoch': '0.01'}
{'loss': '5.755', 'grad_norm': '5.982', 'learning_rate': '2.5e-05', 'epoch': '0.012'}
{'loss': '5.584', 'grad_norm': '5.547', 'learning_rate': '3e-05', 'epoch': '0.014'}
{'loss': '5.266', 'grad_norm': '5.022', 'learning_rate': '3.5e-05', 'epoch': '0.016'}
{'loss': '4.985', 'grad_norm': '4.258', 'learning_rate': '4e-05', 'epoch': '0.018'}
{'loss': '4.703', 'grad_norm': '4.218', 'learning_rate': '4.5e-05', 'epoch': '0.02'}
{'loss': '4.408', 'grad_norm': '4.155', 'learning_rate': '5e-05', 'epoch': '0.022'}
{'loss': '3.996', 'grad_norm': '4.623', 'learning_rate': '4.944e-05', 'epoch': '0.024'}
{'loss': '3.66', 'grad_norm': '5.248', 'learning_rate': '4.889e-05', 'epoch': '0.026'}
{'loss': '3.242', 'grad_norm': '4.073', 'learning_rate': '4.833e-05', 'epoch': '0.028'}
{'loss': '2.953', 'grad_norm': '3.059', 'learning_rate': '4.778e-05', 'epoch': '0.03'}
{'loss': '2.881', 'grad_norm': '3.071', 'learning_rate': '4.722e-05', 'epoch': '0.032'}
{'loss': '2.658', 'grad_norm': '2.837', 'learning_rate': '4.667e-05', 'epoch': '0.034'}
{'loss': '2.462', 'grad_norm': '2.35', 'learning_rate': '4.611e-05', 'epoch': '0.036'}
{'loss': '2.324', 'grad_norm': '2.335', 'learning_rate': '4.556e-05', 'epoch': '0.038'}
{'loss': '2.216', 'grad_norm': '2.029', 'learning_rate': '4.5e-05', 'epoch': '0.04'}
{'loss': '2.063', 'grad_norm': '1.963', 'learning_rate': '4.444e-05', 'epoch': '0.042'}
{'loss': '1.923', 'grad_norm': '2.961', 'learning_rate': '4.389e-05', 'epoch': '0.044'}
{'loss': '1.843', 'grad_norm': '1.887', 'learning_rate': '4.333e-05', 'epoch': '0.046'}
{'loss': '1.71', 'grad_norm': '1.854', 'learning_rate': '4.278e-05', 'epoch': '0.048'}
{'loss': '1.6', 'grad_norm': '1.853', 'learning_rate': '4.222e-05', 'epoch': '0.05'}
{'loss': '1.534', 'grad_norm': '1.657', 'learning_rate': '4.167e-05', 'epoch': '0.052'}
{'loss': '1.43', 'grad_norm': '1.418', 'learning_rate': '4.111e-05', 'epoch': '0.054'}
{'loss': '1.354', 'grad_norm': '1.808', 'learning_rate': '4.056e-05', 'epoch': '0.056'}
{'loss': '1.259', 'grad_norm': '1.624', 'learning_rate': '4e-05', 'epoch': '0.058'}
{'loss': '1.226', 'grad_norm': '1.587', 'learning_rate': '3.944e-05', 'epoch': '0.06'}
{'loss': '1.175', 'grad_norm': '2.734', 'learning_rate': '3.889e-05', 'epoch': '0.062'}
{'loss': '1.112', 'grad_norm': '2.145', 'learning_rate': '3.833e-05', 'epoch': '0.064'}
{'loss': '0.9937', 'grad_norm': '3.265', 'learning_rate': '3.778e-05', 'epoch': '0.066'}
{'loss': '0.9294', 'grad_norm': '7.12', 'learning_rate': '3.722e-05', 'epoch': '0.068'}
{'loss': '0.8897', 'grad_norm': '8.535', 'learning_rate': '3.667e-05', 'epoch': '0.07'}
{'loss': '0.8128', 'grad_norm': '1.413', 'learning_rate': '3.611e-05', 'epoch': '0.072'}
{'loss': '0.7576', 'grad_norm': '1.375', 'learning_rate': '3.556e-05', 'epoch': '0.074'}
{'loss': '0.7112', 'grad_norm': '3.489', 'learning_rate': '3.5e-05', 'epoch': '0.076'}
{'loss': '0.6512', 'grad_norm': '1.469', 'learning_rate': '3.444e-05', 'epoch': '0.078'}
{'loss': '0.579', 'grad_norm': '2.403', 'learning_rate': '3.389e-05', 'epoch': '0.08'}
{'loss': '0.5161', 'grad_norm': '1.883', 'learning_rate': '3.333e-05', 'epoch': '0.082'}
{'loss': '0.4509', 'grad_norm': '1.434', 'learning_rate': '3.278e-05', 'epoch': '0.084'}
{'loss': '0.4113', 'grad_norm': '1.277', 'learning_rate': '3.222e-05', 'epoch': '0.086'}
{'loss': '0.3544', 'grad_norm': '1.074', 'learning_rate': '3.167e-05', 'epoch': '0.088'}
{'loss': '0.3457', 'grad_norm': '0.9934', 'learning_rate': '3.111e-05', 'epoch': '0.09'}
{'loss': '0.3026', 'grad_norm': '1.014', 'learning_rate': '3.056e-05', 'epoch': '0.092'}
{'loss': '0.269', 'grad_norm': '1.038', 'learning_rate': '3e-05', 'epoch': '0.094'}
{'loss': '0.2391', 'grad_norm': '1.062', 'learning_rate': '2.944e-05', 'epoch': '0.096'}
{'loss': '0.2094', 'grad_norm': '1.041', 'learning_rate': '2.889e-05', 'epoch': '0.098'}
{'loss': '0.1862', 'grad_norm': '1.031', 'learning_rate': '2.833e-05', 'epoch': '0.1'}
{'loss': '0.1549', 'grad_norm': '0.9575', 'learning_rate': '2.778e-05', 'epoch': '0.102'}
{'loss': '0.1295', 'grad_norm': '0.9044', 'learning_rate': '2.722e-05', 'epoch': '0.104'}
{'loss': '0.1055', 'grad_norm': '0.745', 'learning_rate': '2.667e-05', 'epoch': '0.106'}
{'loss': '0.08858', 'grad_norm': '0.6573', 'learning_rate': '2.611e-05', 'epoch': '0.108'}
{'loss': '0.07195', 'grad_norm': '0.5423', 'learning_rate': '2.556e-05', 'epoch': '0.11'}
{'loss': '0.05692', 'grad_norm': '0.4431', 'learning_rate': '2.5e-05', 'epoch': '0.112'}
{'loss': '0.04529', 'grad_norm': '0.3721', 'learning_rate': '2.444e-05', 'epoch': '0.114'}
{'loss': '0.03638', 'grad_norm': '0.3228', 'learning_rate': '2.389e-05', 'epoch': '0.116'}
{'loss': '0.03032', 'grad_norm': '0.2831', 'learning_rate': '2.333e-05', 'epoch': '0.118'}
{'loss': '0.02541', 'grad_norm': '0.2552', 'learning_rate': '2.278e-05', 'epoch': '0.12'}
{'loss': '0.02017', 'grad_norm': '0.2169', 'learning_rate': '2.222e-05', 'epoch': '0.122'}
{'loss': '0.01668', 'grad_norm': '0.2584', 'learning_rate': '2.167e-05', 'epoch': '0.124'}
{'loss': '0.01392', 'grad_norm': '0.1694', 'learning_rate': '2.111e-05', 'epoch': '0.126'}
{'loss': '0.01147', 'grad_norm': '0.1516', 'learning_rate': '2.056e-05', 'epoch': '0.128'}
{'loss': '0.009991', 'grad_norm': '0.1338', 'learning_rate': '2e-05', 'epoch': '0.13'}
{'loss': '0.008917', 'grad_norm': '0.1239', 'learning_rate': '1.944e-05', 'epoch': '0.132'}
{'loss': '0.007842', 'grad_norm': '0.1143', 'learning_rate': '1.889e-05', 'epoch': '0.134'}
{'loss': '0.006879', 'grad_norm': '0.1026', 'learning_rate': '1.833e-05', 'epoch': '0.136'}
{'loss': '0.005776', 'grad_norm': '0.09021', 'learning_rate': '1.778e-05', 'epoch': '0.138'}
{'loss': '0.005527', 'grad_norm': '0.09134', 'learning_rate': '1.722e-05', 'epoch': '0.14'}
{'loss': '0.005359', 'grad_norm': '0.08794', 'learning_rate': '1.667e-05', 'epoch': '0.142'}
{'loss': '0.004597', 'grad_norm': '0.07878', 'learning_rate': '1.611e-05', 'epoch': '0.144'}
{'loss': '0.004414', 'grad_norm': '0.0775', 'learning_rate': '1.556e-05', 'epoch': '0.146'}
{'loss': '0.004038', 'grad_norm': '0.07212', 'learning_rate': '1.5e-05', 'epoch': '0.148'}
{'loss': '0.004547', 'grad_norm': '0.1031', 'learning_rate': '1.444e-05', 'epoch': '0.15'}
{'loss': '0.003783', 'grad_norm': '0.07255', 'learning_rate': '1.389e-05', 'epoch': '0.152'}
{'loss': '0.003223', 'grad_norm': '0.06193', 'learning_rate': '1.333e-05', 'epoch': '0.154'}
{'loss': '0.00302', 'grad_norm': '0.0596', 'learning_rate': '1.278e-05', 'epoch': '0.156'}
{'loss': '0.003421', 'grad_norm': '0.08068', 'learning_rate': '1.222e-05', 'epoch': '0.158'}
{'loss': '0.002777', 'grad_norm': '0.05817', 'learning_rate': '1.167e-05', 'epoch': '0.16'}
{'loss': '0.002539', 'grad_norm': '0.05271', 'learning_rate': '1.111e-05', 'epoch': '0.162'}
{'loss': '0.002776', 'grad_norm': '0.05555', 'learning_rate': '1.056e-05', 'epoch': '0.164'}
{'loss': '0.002186', 'grad_norm': '0.04386', 'learning_rate': '1e-05', 'epoch': '0.166'}
{'loss': '0.002183', 'grad_norm': '0.04439', 'learning_rate': '9.444e-06', 'epoch': '0.168'}
{'loss': '0.002155', 'grad_norm': '0.04387', 'learning_rate': '8.889e-06', 'epoch': '0.17'}
{'loss': '0.002161', 'grad_norm': '0.04367', 'learning_rate': '8.333e-06', 'epoch': '0.172'}
{'loss': '0.001777', 'grad_norm': '0.03646', 'learning_rate': '7.778e-06', 'epoch': '0.174'}
{'loss': '0.00186', 'grad_norm': '0.03914', 'learning_rate': '7.222e-06', 'epoch': '0.176'}
{'loss': '0.001675', 'grad_norm': '0.03642', 'learning_rate': '6.667e-06', 'epoch': '0.178'}
{'loss': '0.002015', 'grad_norm': '0.04299', 'learning_rate': '6.111e-06', 'epoch': '0.18'}
{'loss': '0.001926', 'grad_norm': '0.04076', 'learning_rate': '5.556e-06', 'epoch': '0.182'}
{'loss': '0.001825', 'grad_norm': '0.04159', 'learning_rate': '5e-06', 'epoch': '0.184'}
{'loss': '0.001893', 'grad_norm': '0.03984', 'learning_rate': '4.444e-06', 'epoch': '0.186'}
{'loss': '0.001522', 'grad_norm': '0.03156', 'learning_rate': '3.889e-06', 'epoch': '0.188'}
{'loss': '0.001613', 'grad_norm': '0.0424', 'learning_rate': '3.333e-06', 'epoch': '0.19'}
{'loss': '0.001491', 'grad_norm': '0.03038', 'learning_rate': '2.778e-06', 'epoch': '0.192'}
{'loss': '0.001538', 'grad_norm': '0.03133', 'learning_rate': '2.222e-06', 'epoch': '0.194'}
{'loss': '0.001571', 'grad_norm': '0.03447', 'learning_rate': '1.667e-06', 'epoch': '0.196'}
{'loss': '0.001838', 'grad_norm': '0.04134', 'learning_rate': '1.111e-06', 'epoch': '0.198'}
{'loss': '0.001658', 'grad_norm': '0.0354', 'learning_rate': '5.556e-07', 'epoch': '0.2'}
{'train_runtime': '436.8', 'train_samples_per_second': '0.458', 'train_steps_per_second': '0.229', 'train_loss': '1.158', 'epoch': '0.2'}
100%|██████████| 100/100 [07:16<00:00,  4.48s/it]

Training completed. Do not forget to share your model on huggingface.co/models =)
-------------------------------------------------- Peft Weights after training --------------------------------------------------
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight:
shape: (64, 2048)
mean: 0.000003
std: 0.012781
min: -0.023816
max: 0.023442
percentile_25: -0.011042
percentile_50: -0.000020
percentile_75: 0.011043
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight:
shape: (4096, 64)
mean: 0.000003
std: 0.000679
min: -0.003111
max: 0.002177
percentile_25: -0.000536
percentile_50: -0.000000
percentile_75: 0.000540
---------------------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Trainer Output --------------------------------------------------
TrainOutput(global_step=100, training_loss=1.15819159808103, metrics={'train_runtime': 436.7624, 'train_samples_per_second': 0.458, 'train_steps_per_second': 0.229, 'total_flos': 1053550340505600.0, 'train_loss': 1.15819159808103, 'epoch': 0.2})
--------------------------------------------------------------------------------------------------------------------


-------------------------------------------------- Responses after training --------------------------------------------------
✓ response 1 contains answer
✓ response 2 contains answer
✓ response 3 contains answer
✓ response 4 contains answer
✓ response 5 contains answer
------------------------------------------------------------------------------------------------------------------------------

Detected local model directory: /models/Qwen3-30B-A3B
Found HuggingFace hub cache directory: /root/.cache/huggingface/hub

Unsloth: Preparing safetensor model files:   0%|          | 0/13 [00:00<?, ?it/s]
Unsloth: Preparing safetensor model files: 100%|██████████| 13/13 [00:00<00:00, 232025.33it/s]

Unsloth: Merging weights into 16bit:   0%|          | 0/13 [00:00<?, ?it/s]
Unsloth: Merging weights into 16bit:   8%|▊         | 1/13 [00:02<00:24,  2.00s/it]
Unsloth: Merging weights into 16bit:  15%|█▌        | 2/13 [00:04<00:22,  2.06s/it]
Unsloth: Merging weights into 16bit:  23%|██▎       | 3/13 [00:05<00:19,  1.97s/it]
Unsloth: Merging weights into 16bit:  31%|███       | 4/13 [00:06<00:12,  1.44s/it]
Unsloth: Merging weights into 16bit:  38%|███▊      | 5/13 [00:08<00:13,  1.65s/it]
Unsloth: Merging weights into 16bit:  46%|████▌     | 6/13 [00:10<00:12,  1.75s/it]
Unsloth: Merging weights into 16bit:  54%|█████▍    | 7/13 [00:12<00:10,  1.71s/it]
Unsloth: Merging weights into 16bit:  62%|██████▏   | 8/13 [00:13<00:08,  1.68s/it]
Unsloth: Merging weights into 16bit:  69%|██████▉   | 9/13 [00:15<00:07,  1.79s/it]
Unsloth: Merging weights into 16bit:  77%|███████▋  | 10/13 [00:17<00:05,  1.88s/it]
Unsloth: Merging weights into 16bit:  85%|████████▍ | 11/13 [00:19<00:03,  1.86s/it]
Unsloth: Merging weights into 16bit:  92%|█████████▏| 12/13 [00:21<00:01,  1.78s/it]
Unsloth: Merging weights into 16bit: 100%|██████████| 13/13 [00:23<00:00,  1.78s/it]
Unsloth: Merging weights into 16bit: 100%|██████████| 13/13 [00:23<00:00,  1.78s/it]
```

</details>

---

### 6. 测试 Attention 算子

#### 6.1 运行基准测试

```bash
python ./scripts/run_kernel_benchmark.sh attention

```

#### 6.2 测试日志（可折叠展示）

<details>
<summary><strong>点击展开查看 Attention 算子测试日志</strong></summary>

```text
Config: batch=1 seq=128 heads=32 dim=128 dtype=torch.float16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=0.17ms bwd=0.22ms | stable=True
  - flash_attn | timed                    | fwd_diff=1.953e-03 bwd_diff=3.906e-03 | fwd=0.26ms bwd=0.79ms | stable=True
  - sdpa       | timed                    | fwd_diff=1.953e-03 bwd_diff=3.906e-03 | fwd=0.07ms bwd=0.13ms | stable=True

Config: batch=1 seq=2048 heads=32 dim=128 dtype=torch.float16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=6.11ms bwd=9.36ms | stable=True
  - flash_attn | timed                    | fwd_diff=1.953e-03 bwd_diff=3.906e-03 | fwd=1.34ms bwd=16.28ms | stable=True
  - sdpa       | timed                    | fwd_diff=1.953e-03 bwd_diff=3.906e-03 | fwd=2.13ms bwd=4.46ms | stable=True

Config: batch=4 seq=128 heads=32 dim=128 dtype=torch.float16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=0.15ms bwd=0.22ms | stable=True
  - flash_attn | timed                    | fwd_diff=2.441e-03 bwd_diff=3.906e-03 | fwd=0.22ms bwd=0.95ms | stable=True
  - sdpa       | timed                    | fwd_diff=2.441e-03 bwd_diff=3.906e-03 | fwd=0.10ms bwd=0.18ms | stable=True

Config: batch=4 seq=2048 heads=32 dim=128 dtype=torch.float16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=23.19ms bwd=36.10ms | stable=True
  - flash_attn | timed                    | fwd_diff=1.953e-03 bwd_diff=7.812e-03 | fwd=7.04ms bwd=62.54ms | stable=True
  - sdpa       | timed                    | fwd_diff=1.953e-03 bwd_diff=7.812e-03 | fwd=6.49ms bwd=14.03ms | stable=True

Config: batch=1 seq=128 heads=32 dim=128 dtype=torch.bfloat16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=0.16ms bwd=0.21ms | stable=True
  - flash_attn | timed                    | fwd_diff=3.125e-02 bwd_diff=3.125e-02 | fwd=0.23ms bwd=0.82ms | stable=True
  - sdpa       | timed                    | fwd_diff=3.125e-02 bwd_diff=3.125e-02 | fwd=0.07ms bwd=0.14ms | stable=True

Config: batch=1 seq=2048 heads=32 dim=128 dtype=torch.bfloat16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=6.21ms bwd=9.46ms | stable=True
  - flash_attn | timed                    | fwd_diff=2.344e-02 bwd_diff=6.250e-02 | fwd=1.32ms bwd=18.18ms | stable=True
  - sdpa       | timed                    | fwd_diff=2.344e-02 bwd_diff=6.250e-02 | fwd=2.12ms bwd=3.89ms | stable=True

Config: batch=4 seq=128 heads=32 dim=128 dtype=torch.bfloat16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=0.15ms bwd=0.21ms | stable=True
  - flash_attn | timed                    | fwd_diff=1.953e-02 bwd_diff=3.125e-02 | fwd=0.22ms bwd=1.01ms | stable=True
  - sdpa       | timed                    | fwd_diff=1.953e-02 bwd_diff=3.125e-02 | fwd=0.10ms bwd=0.19ms | stable=True

Config: batch=4 seq=2048 heads=32 dim=128 dtype=torch.bfloat16
  - torch_ref  | timed                    | fwd_diff=0.000e+00 bwd_diff=0.000e+00 | fwd=23.13ms bwd=36.02ms | stable=True
  - flash_attn | timed                    | fwd_diff=2.344e-02 bwd_diff=6.250e-02 | fwd=7.05ms bwd=70.68ms | stable=True
  - sdpa       | timed                    | fwd_diff=2.344e-02 bwd_diff=6.250e-02 | fwd=6.52ms bwd=12.10ms | stable=True

```

</details>

---

### 7. 测试 MoE 算子

#### 7.1 FP16 精度 / 性能测试

```bash
python ./scripts/moe_impl_benchmark.py \
  --batch-sizes 1 4 \
  --seq-lens 128 512 \
  --dtypes fp16 \
  --num-iters 3
```

<details>
<summary><strong>点击展开查看 MoE FP16 测试日志</strong></summary>

```text

=== dtype=fp16 (torch.float16) ===

Config: batch=1, seq=128, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=30.97ms bwd=43.62ms | ref: fwd=30.48ms bwd=41.88ms | stable=True

Config: batch=1, seq=512, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=32.55ms bwd=46.75ms | ref: fwd=32.55ms bwd=46.82ms | stable=True

Config: batch=4, seq=128, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=32.95ms bwd=46.71ms | ref: fwd=32.81ms bwd=46.70ms | stable=True

Config: batch=4, seq=512, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=42.04ms bwd=76.50ms | ref: fwd=41.85ms bwd=75.95ms | stable=True

```

</details>

#### 7.2 BF16 精度 / 性能测试

```bash
python ./scripts/moe_impl_benchmark.py \
  --batch-sizes 1 4 \
  --seq-lens 128 512 \
  --dtypes bf16 \
  --num-iters 3
```

<details>
<summary><strong>点击展开查看 MoE BF16 测试日志</strong></summary>

```text
Config: batch=1, seq=128, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=30.23ms bwd=45.18ms | ref: fwd=29.95ms bwd=44.19ms | stable=True

Config: batch=1, seq=512, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=32.47ms bwd=48.94ms | ref: fwd=32.34ms bwd=49.06ms | stable=True

Config: batch=4, seq=128, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=32.29ms bwd=48.99ms | ref: fwd=32.57ms bwd=49.05ms | stable=True

Config: batch=4, seq=512, hidden=2048, experts=64
Parity   fwd_diff=0.000e+00 | bwd_diff=0.000e+00 | router_diff=0.000e+00 | stable=True
Perf     target: fwd=42.92ms bwd=77.06ms | ref: fwd=42.76ms bwd=76.96ms | stable=True

```

</details>

---

## FP8 尝试

### 1. FP8 环境准备

> 由于unsloth中在loader.py中209-220行设置，在设置load_in_fp8=True时，需要同时设置fast_inference=true，并且需要下载vllm,
> 同时由于unsloth中在loader_utils.py中350-362行设置torchao版本，因此需要更新torchao版本到0.15.0

<details>
<summary><strong>点击展开查看 AMD 平台下 FP8 需要做的额外代码修改</strong></summary>

```python
if fast_inference:
    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
            "You can do this in a terminal via `pip install vllm`"
        )
# [TODO] For now fast_inference only works with fast_inference ie vLLM
if load_in_fp8 != False:
    if not fast_inference:
        raise NotImplementedError(
            "Unsloth: set `fast_inference = True` when doing `load_in_fp8`."
        )
```

</details>

<details>
<summary><strong>点击展开查看 AMD 平台下 FP8 需要做的额外代码修改</strong></summary>

```python
if importlib.util.find_spec("torchao") is None:
        raise ValueError(
            "Unsloth: Please install torchao for on the fly float8 to work! Try `unsloth/Qwen3-8B` instead."
        )
    import torchao

    error_message = (
        "Unsloth: `load_in_fp8` requires torchao 0.15.0+ (or nightly).\n"
        f"You have torchao version={torchao.__version__}\n"
        "Use `pip install --upgrade --force-reinstall torchao`"
    )
    if Version(torchao.__version__) < Version("0.15.0"):
        raise ValueError(error_message)
```


</details>

> 由于Unsloth中在loader_utils.py文件中的332-340行代码中设置仅支持H100及后续英伟达芯片,需要注释掉：

<details>
<summary><strong>点击展开查看 AMD 平台下 FP8 需要做的额外代码修改</strong></summary>

```python
# Check if this is Hopper or above
    # if not (
    #     torch.cuda.is_available()
    #     and torch.version.cuda
    #     and torch.cuda.get_device_capability() >= (9, 0)
    # ):
    #     raise ValueError(
    #         "Unsloth: On the fly `load_in_fp8` requires H100 GPUs or after. Try `unsloth/Qwen3-8B` instead."
    #     )
```


</details>
---

### 2. 使用 FP8 模型测试

> 本小节记录在 FP8 量化模型上的测试情况。

#### 2.1 AMD 现象

<details>
<summary><strong>点击展开查看 AMD 平台下 FP8 模型测试现象与日志</strong></summary>

```text
准备工作完成后，需要更换为bitsandbytes-ROCM、torchao-rocm，使用Llama-3.1-8B-Instruct-FP8-Dynamic模型进行测试，首先设置参数load_in_4bit = False,fast_inference = False,load_in_fp8 = False,
最后结果在前向传播阶段出现AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'quantize_fp8_per_row'错误
设置参数load_in_4bit = False,fast_inference = True, load_in_fp8 = True
最后结果出现ValueError: The model is quantized with CompressedTensorsConfig but you are passing a TorchAoConfig config. Please make sure to pass the same quantization config class to `from_pretrained` with different loading attributes
```

</details>

#### 2.2 NVIDIA 现象

<details>
<summary><strong>点击展开查看 NVIDIA 平台下 FP8 模型测试现象与日志</strong></summary>

```text
Unsloth在NVIDIA上测试Llama-3.1-8B-Instruct-FP8-Dynamic模型使用load_in_4bit=True通过之后，测试load_in_FP8=True,
完成上述工作之后，进行测试设置参数load_in_4bit=False,fast_inference = False,load_in_fp8 = False
最后结果在前向传播阶段出现AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'quantize_fp8_per_row'错误，之后设置参数load_in_4bit=False,fast_inference = True, load_in_fp8 = True进行测试
最后结果出现ValueError: The model is quantized with CompressedTensorsConfig but you are passing a TorchAoConfig config. Please make sure to pass the same quantization config class to `from_pretrained` with different loading attributes.
```

</details>

---

### 3. 使用非量化模型测试

> 使用未量化（如 FP16 / BF16）的模型作为输入，测试FP8的可行性。

#### 3.1 AMD 现象

<details>
<summary><strong>点击展开查看 AMD 平台下非量化模型测试现象与日志</strong></summary>

```text
使用模型Meta-Llama-3.1-8B-Instruct模型，设置参数load_in_4bit = False,fast_inference = True, load_in_fp8 = True。
最后结果出现如下错误
[rank0]: RuntimeError: Cannot set version_counter for inference tensor
[rank0]:[W1211 07:21:18.443841415 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

</details>

#### 3.2 NVIDIA 现象

<details>
<summary><strong>点击展开查看 NVIDIA 平台下非量化模型测试现象与日志</strong></summary>

```text
使用模型Meta-Llama-3.1-8B-Instruct模型，设置参数load_in_4bit = False,fast_inference = True, load_in_fp8 = True
最后结果出现如下错误：
[rank0]: RuntimeError: Cannot set version_counter for inference tensor
[rank0]:[W1211 01:13:08.369605398 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())。
```

</details>

## 后续工作-to-do

- 导出微调后的 LLaMA / Qwen-MoE 模型在 **llama.cpp** 或 **vLLM** 中验证模型可成功加载；
- 验证生成质量（示例对话），并补充对应日志与结果说明。
