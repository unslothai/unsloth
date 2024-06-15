# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .llama import *
from ._utils import __version__
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)

class FastLlavaVicunaModel(FastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name     = "llava-hf/llava-v1.6-vicuna-7b-hf",
        max_seq_length = None,
        dtype          = None,
        load_in_4bit   = True,
        token          = None,
        device_map     = "sequential",
        rope_scaling   = None,
        fix_tokenizer  = True,
        model_patcher  = None,
        tokenizer_name = None,
        trust_remote_code = False,
        **kwargs,
    ):
        if token is None and "HF_TOKEN" in os.environ:
            token = os.environ["HF_TOKEN"]

        if token is None and "HUGGINGFACE_TOKEN" in os.environ:
            token = os.environ["HUGGINGFACE_TOKEN"]

        if model_patcher is None: model_patcher = FastLlamaModel
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth: Fast {model_patcher.__name__[4:-5]} patching release {__version__}\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"\
           f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"\
           f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. Xformers = {xformers_version}. FA = {HAS_FLASH_ATTENTION}.\n"\
           f' "-____-"     Free Apache license: http://github.com/unslothai/unsloth'
        print(statistics)
        model_patcher.pre_patch()
        # get_statistics()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        # RoPE scaling
        model_max_seq_length = \
            AutoConfig.from_pretrained(model_name, token = token).text_config.max_position_embeddings



        # If max_seq_length is not specified, use maximum fron config
        if max_seq_length is None:
            max_seq_length = model_max_seq_length
        pass

        if (rope_scaling is None) and (max_seq_length > model_max_seq_length):
            rope_scaling = max_seq_length / model_max_seq_length
            logger.warning_once(
                f"Unsloth: {model_name} can only handle sequence lengths of at most "\
                f"{model_max_seq_length}.\nBut with kaiokendev's RoPE scaling of "\
                f"{round(rope_scaling, 3)}, it can be magically be extended to "\
                f"{max_seq_length}!"
            )
            rope_scaling = {"type": "linear", "factor": rope_scaling,}
        pass

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                bnb_4bit_quant_storage    = dtype,
            )
        pass

        # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/discussions/12
        # RoPE Scaling's max_position_embeddings must be updated
        max_position_embeddings = max(max_seq_length, model_max_seq_length)
        try:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                device_map              = device_map,
                torch_dtype             = dtype,
                quantization_config     = bnb_config,
                token                   = token,
                text_config             = {
                    "rope_scaling": rope_scaling,
                    "max_position_embeddings": max_position_embeddings,
                },
                trust_remote_code       = trust_remote_code,
                **kwargs,
            )
        except Exception as error:
            if "rope_scaling" in str(error):
                if rope_scaling is not None:
                    raise TypeError("Unsloth: {model_name} does not support rope_scaling.")
                pass

                # Counteract missing rope_scaling
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map              = device_map,
                    torch_dtype             = dtype,
                    quantization_config     = bnb_config,
                    token                   = token,
                    text_config             = {
                        "max_position_embeddings": max_position_embeddings,
                    },
                    trust_remote_code       = trust_remote_code,
                    **kwargs,
                )
            else:
                raise error
            pass
        pass

        # Counteract saved tokenizers
        processor = LlavaNextProcessor.from_pretrained(model_name)

        model.language_model = model_patcher.post_patch(model.language_model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.language_model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        # Patch Trainer
        from transformers.trainer import Trainer
        try:
            if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
                inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
                Trainer._original_training_loop = inner_training_loop
            else:
                inner_training_loop = Trainer._original_training_loop
        except:
            raise RuntimeError(
                "Our OSS was designed for people with few GPU resources to level the playing field.\n"
                "The OSS Apache 2 license only supports one GPU - please obtain a commercial license.\n"
                "We're a 2 person team, so we still have to fund our development costs - thanks!\n"
                "If you don't, please consider at least sponsoring us through Ko-fi! Appreciate it!",
            )
        pass

        import transformers.trainer
        items_in_trainer = dir(transformers.trainer)
        good_items = []
        for item in items_in_trainer:
            # TODO: Support Deepspeed
            if item.startswith(("deepspeed", "xm", "met", "smp")): continue
            if item in inner_training_loop: good_items.append(item)
        pass
        exec("from transformers.trainer import (" + ", ".join(x for x in good_items) + ")", globals())

        start = re.search('logger\.info\([\"\'].+?Running training', inner_training_loop).span(0)[0]
        end = inner_training_loop.find("\n\n", start)
        original_debug = inner_training_loop[start:end]
        spaces = re.search('\n([\s\t]{1,})', original_debug).group(0)[1:]
        front_spaces = re.match('([\s\t]{1,})', inner_training_loop).group(0)

        debug_info = """debug_info = \\
        f"==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = {args.world_size}\\n"\\
        f"   \\\\\\   /|    Num examples = {num_examples:,} | Num Epochs = {num_train_epochs:,}\\n"\\
        f"O^O/ \\_/ \\    Batch size per device = {self._train_batch_size:,} | Gradient Accumulation steps = {args.gradient_accumulation_steps}\\n"\\
        f"\\        /    Total batch size = {total_train_batch_size:,} | Total steps = {max_steps:,}\\n"\\
        f' "-____-"     Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}'
        logger.warning(debug_info)
        import subprocess, re, gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()"""

        debug_info = debug_info.split('\n')
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

        debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once(
                "* Our OSS was designed for people with few GPU resources to level the playing field.\\n"
                "* The OSS Apache 2 license only supports one GPU - please obtain a commercial license.\\n"
                "* We're a 2 person team, so we still have to fund our development costs - thanks!\\n"
                "* If you don't, please consider at least sponsoring us through Ko-fi! Appreciate it!",
            )
        debug_info ="""
        debug_info = debug_info.split('\n')
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace("debug_info =", debug_info, 1)

        front_spaces = re.match(r"[\t\s]{1,}", inner_training_loop).group(0)
        inner_training_loop = re.sub(r"^" + front_spaces, "", inner_training_loop, flags = re.MULTILINE)
        inner_training_loop = inner_training_loop.replace(
            "train_dataloader = tpu_spmd_dataloader(train_dataloader)",
            "raise RuntimeError('Unsloth: TPUs are not yet supported!')"
        )
        inner_training_loop = inner_training_loop.replace(
            "self.accelerator.free_memory()",
            "self.accelerator.free_memory()\n" + \
            front_spaces + "if self.is_deepspeed_enabled:"\
            "raise RuntimeError('Unsloth: Deepspeed is not yet supported!')\n", 1,
        )

        check_batches = """train_dataloader = self.get_train_dataloader()
        ga  = args.gradient_accumulation_steps
        bsz = self._train_batch_size
        total_batches = bsz * ga * args.world_size
        n_total_devices = total_batches // ga // bsz
        if n_total_devices > 1:
            logger.warning_once(
                "* Our OSS was designed for people with few GPU resources to level the playing field.\\n"
                "* The OSS Apache 2 license only supports one GPU - please obtain a commercial license.\\n"
                "* We're a 2 person team, so we still have to fund our development costs - thanks!\\n"
                "* If you don't, please consider at least sponsoring us through Ko-fi! Appreciate it!",
            )
            divisor = n_total_devices / 1
            bsz = self._train_batch_size = max(int(bsz / divisor), 1)
            if total_batches // ga // bsz > 1:
                divisor = n_total_devices / 1
                ga = args.gradient_accumulation_steps = max(int(ga / divisor), 1)"""
        check_batches = check_batches.split('\n')
        check_batches = "\n".join([check_batches[0]] + [front_spaces + x[8:] for x in check_batches[1:]])
        inner_training_loop = inner_training_loop.replace(
            "train_dataloader = self.get_train_dataloader()",
            check_batches, 1,
        )
        inner_training_loop = inner_training_loop.replace(
            "_inner_training_loop",
            "_fast_inner_training_loop", 1,
        )
        exec(inner_training_loop, globals())

        Trainer._inner_training_loop = _fast_inner_training_loop
        inner_training_loop = inner_training_loop.replace(
            "is_torch_tpu_available()",
            "False",
        )
        if "n_total_devices >" not in inner_training_loop:
            raise RuntimeError(
                "Our OSS was designed for people with few GPU resources to level the playing field.\n"
                "The OSS Apache 2 license only supports one GPU - please obtain a commercial license.\n"
                "We're a 2 person team, so we still have to fund our development costs - thanks!\n"
                "If you don't, please consider at least sponsoring us through Ko-fi! Appreciate it!",
            )
        pass
        inner_training_loop = inner_training_loop.replace(
            "is_sagemaker_mp_enabled()",
            "False",
        )
        exec(inner_training_loop, globals())
        Trainer._inner_training_loop = _fast_inner_training_loop

        # Save max_seq_length
        model.language_model.max_seq_length = max_position_embeddings
        internal_model = model.language_model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_position_embeddings
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_position_embeddings

        # We check the tokenizer first for errors
        if False and fix_tokenizer:
            tokenizer = check_tokenizer(
                model            = model,
                tokenizer        = tokenizer,
                model_name       = model_name,
                model_max_length = max_position_embeddings,
                padding_side     = "right",
                token            = token,
            )
        pass
        # patch_saving_functions(tokenizer)

        # Fix up config for transformers uploading PEFT
        # Not necessary anymore since we require transformers>=4.37!
        if False:
            name = model.config._name_or_path
            if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                name = name[:len(name) - len("-bnb-4bit")]
                model.config.update({"_name_or_path" : name})
            pass
        pass

        # Log Unsloth version for future fastpaths for inference
        model.config.update({"unsloth_version" : __version__})

        # Add save modules
        patch_saving_functions(model)
        Trainer._inner_training_loop = _fast_inner_training_loop

        return model, processor
    pass


pass
