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
from ..kernels import patch_layernorm, unpatch_layernorm
from ..kernels import patch_rms_layernorm, unpatch_rms_layernorm
from ..kernels import patch_llama_for_causal_lm, unpatch_llama_for_causal_lm
from ._utils import patch_gradient_checkpointing

from transformers import AutoProcessor
try:
    from transformers import MllamaForConditionalGeneration
except:
    raise ImportError(
        "Unsloth: Please update your transformers version to 4.46.0 for Llama 3.2 support!"
    )
pass

class FastVisionModel:

    def pre_patch(self):
        patch_gradient_checkpointing()
        patch_layernorm()
        patch_rms_layernorm()
        patch_llama_for_causal_lm()
    pass

    def post_unpatch(self):
        unpatch_layernorm()
        unpatch_rms_layernorm()
        unpatch_llama_for_causal_lm()
    pass


    @staticmethod
    def from_pretrained(
        model_name        = "llava-hf/llava-1.5-7b-hf",
        max_seq_length    = None,
        dtype             = None,
        load_in_4bit      = True,
        token             = None,
        device_map        = "sequential",
        rope_scaling      = None,
        trust_remote_code = False,
        **kwargs,
    ):
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )
        pass
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_patcher.__name__[4:-5]} patching. Transformers = {transformers_version}.\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"\
           f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"\
           f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free Apache license: http://github.com/unslothai/unsloth'
        print(statistics)

        # Warn about fast transfers
        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")
        pass
        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer

        get_statistics() # For debugging - we use a download counter to see if environments are not breaking 

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        # We currently only support NVIDIA GPUs - AMD / Intel is a work in progress!
        pre_check = check_nvidia()

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
            )
        pass

        # Cannot be None, since HF now checks for the config
        if load_in_4bit: kwargs["quantization_config"] = bnb_config
        
        self.pre_patch()
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            device_map              = device_map,
            torch_dtype             = dtype,
            # quantization_config     = bnb_config,
            token                   = token,
            max_position_embeddings = max_position_embeddings,
            trust_remote_code       = trust_remote_code,
            attn_implementation     = "sdpa",
            **kwargs,
        )
        self.post_unpatch()

        # Return old flag
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer
        # We currently only support NVIDIA GPUs - AMD / Intel is a work in progress!
        post_check = check_nvidia()

        # Counteract saved tokenizers
        tokenizer = AutoProcessor.from_pretrained(
            model_name,
        )
        model = FastVisionModel.post_patch(model)

        # Patch Trainer
        from transformers.trainer import Trainer
        try:
            if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
                inner_training_loop = inspect.getsource(Trainer._inner_training_loop)
                Trainer._original_training_loop = inner_training_loop
            else:
                inner_training_loop = Trainer._original_training_loop
        except:
            raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
        pass

        if ((post_check - pre_check) >= 1).sum() > 1:
            raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')

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
        import subprocess, re, gc, numpy as np
        a = np.array([0,])
        try:
            a = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell = True)
            a = re.findall(rb'([\\d]{1,})[\\s]{1,}M', a)
            a = np.array([int(x.decode('utf-8'))/1024 for x in a])
        except:
            if not torch.cuda.is_available():
                raise RuntimeError('Unsloth: We do not support AMD / Intel machines yet - it is a work in progress!')
        if ((a - PRE_CHECK) >= 1).sum() > 1:
            raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()"""

        debug_info = debug_info.split('\n')
        debug_info = "\n".join([debug_info[0]] + [spaces + x[8:] for x in debug_info[1:]])
        inner_training_loop = inner_training_loop.replace(original_debug, debug_info)

        debug_info = """n_total_devices = total_train_batch_size // \\
            args.gradient_accumulation_steps // self._train_batch_size
        if n_total_devices > 1:
            logger.warning_once('Unsloth currently does not support multi GPU setups - but we are working on it!')
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
            logger.warning_once('Unsloth currently does not support multi GPU setups - but we are working on it!')
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
            raise RuntimeError('Unsloth currently does not support multi GPU setups - but we are working on it!')
        pass
        inner_training_loop = inner_training_loop.replace(
            "is_sagemaker_mp_enabled()",
            "False",
        )
        exec(inner_training_loop, globals())
        Trainer._inner_training_loop = _fast_inner_training_loop

        # Save max_seq_length
        model.max_seq_length = max_position_embeddings
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_position_embeddings
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_position_embeddings

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

        # Also fix torch_dtype
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "config"):
                if   internal_model.config.torch_dtype ==  "float32":
                    internal_model.config.torch_dtype = torch.float32
                elif internal_model.config.torch_dtype == "bfloat16":
                    internal_model.config.torch_dtype = torch.bfloat16
                elif internal_model.config.torch_dtype ==  "float16":
                    internal_model.config.torch_dtype = torch.float16
                pass
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "config"):
            if   internal_model.config.torch_dtype ==  "float32":
                internal_model.config.torch_dtype = torch.float32
            elif internal_model.config.torch_dtype == "bfloat16":
                internal_model.config.torch_dtype = torch.bfloat16
            elif internal_model.config.torch_dtype ==  "float16":
                internal_model.config.torch_dtype = torch.float16
            pass
        pass
        
        return model, tokenizer
    pass


    @staticmethod
    def post_patch(model):
        # Patch model
        layers = model.model.layers
        lm_head = model.get_output_embeddings().weight
        
        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        return model
    pass


    @staticmethod
    def get_peft_model(
        model,
        r                   = 16,
        target_modules      = "all-linear",
        lora_alpha          = 16,
        lora_dropout        = 0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = True,
        random_state        = 3407,
        max_seq_length      = 2048, # not used anymore
        use_rslora          = False,
        modules_to_save     = None,
        init_lora_weights   = True,
        loftq_config        = {},
        temporary_location  = "_unsloth_temporary_saved_buffers",
        **kwargs,
    ):
        transformers_set_seed(random_state)

        # Get LoRA
        arguments = dict(
            r                   = r,
            lora_alpha          = lora_alpha,
            target_modules      = target_modules,
            lora_dropout        = lora_dropout,
            bias                = bias,
            layers_to_transform = layers_to_transform,
            init_lora_weights   = init_lora_weights,
            # loftq_config        = loftq_config,
            # use_rslora          = use_rslora,
            modules_to_save     = modules_to_save,
            **kwargs,
        )

        lora_config = LoraConfig(**arguments)

        model = _get_peft_model(model, lora_config)

        model = FastVisionModel.patch_peft_model(model, use_gradient_checkpointing)

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass

        return model
    pass


    @staticmethod
    def patch_peft_model(
        model,
        use_gradient_checkpointing = True,
    ):

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )

        # Fix up config for transformers uploading PEFT
        for active_adapter in model.peft_config.keys():
            # Not necessary since we requires transformers >= 4.37
            if False:
                name = model.peft_config[active_adapter].base_model_name_or_path
                if name.startswith("unsloth/") and name.endswith("-bnb-4bit"):
                    name = name[:len(name) - len("-bnb-4bit")]
                    model.peft_config[active_adapter].base_model_name_or_path = name
                pass
            # Add revision to enable future fast inference paths
            # [TODO] Bugs out!see https://github.com/unslothai/unsloth/issues/492
            # model.peft_config[active_adapter].revision = f"unsloth"
        pass

        from transformers.trainer import Trainer 
        if Trainer._inner_training_loop.__name__ != "_fast_inner_training_loop":
            raise RuntimeError(
                'Unsloth currently does not work on multi GPU setups - sadly we are a 2 brother team so '\
                'enabling it will require much more work, so we have to prioritize. Please understand!\n'\
                'We do have a separate beta version, which you can contact us about!\n'\
                'Thank you for your understanding and we appreciate it immensely!'
            )
        pass

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "\
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )
        patch_saving_functions(model)

        # Patch cross entropy loss labels
        # Fixes https://github.com/unslothai/unsloth/issues/10
        max_seq_length = model.max_seq_length
        extra_ignored_labels = torch.full((max_seq_length, 1), -100, device = "cuda:0")
        model.model.extra_ignored_labels = extra_ignored_labels
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_seq_length        

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            pass
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        pass

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        pass
        return model
    pass


    @staticmethod
    def for_inference(model):
        # if model.config.model_type == "qwen2":
        #     FastLlamaModel.for_training(model)
        #     return
        # pass

        internal_model = model
        internal_model.gradient_checkpointing = False
        internal_model.training = False

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = False
            internal_model.training = False
        pass
        if hasattr(internal_model, "training"):
            internal_model.training = False
        pass

        # Also check if lm_head / embeddings are trained
        internal_model = model
        while not hasattr(internal_model, "lm_head"):
            internal_model = internal_model.model
        pass
        lm_head = internal_model.lm_head.weight
        device_type = lm_head.device.type
        dtype = model.config.torch_dtype
        
        if type(dtype) is str:
            if   dtype ==  "float16": dtype = torch.float16
            elif dtype == "bfloat16": dtype = torch.bfloat16
        pass

        # Also disable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        pass

        return model
    pass


    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        internal_model = model
        internal_model.gradient_checkpointing = use_gradient_checkpointing
        internal_model.training = True

        # Delete all fast inference loras
        for param in model.parameters():
            if hasattr(param, "_fast_lora"):
                del param._fast_lora
        pass

        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
            internal_model.gradient_checkpointing = use_gradient_checkpointing
            internal_model.training = True
        pass
        if hasattr(internal_model, "training"):
            internal_model.training = True
        pass

        # Also re-enable training for embeddings for NEFTune
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        pass

        return model
    pass
pass
