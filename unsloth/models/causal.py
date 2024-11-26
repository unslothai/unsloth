import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from .llama import *
from ..kernels import (
    post_patch_loss_function,
)
from ._utils import __version__
from typing import Optional

__all__ = ["FastBaseCausalModel"]

def _wrap_fast_inference(generate, device_type, dtype, model):
    @torch.inference_mode
    def _fast_generate(*args, **kwargs):
        kwargs.pop("token_type_ids", None)

        # Check pad_token
        model_eos_token_id = getattr(model.config, "eos_token_id", None)
        if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
            model_eos_token_id = model_eos_token_id[0]

        kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)

        # Autocasted
        with torch.autocast(device_type=device_type, dtype=dtype):
            output = generate(*args, **kwargs)
        return output
    return _fast_generate

class FastBaseCausalModel:
    @staticmethod
    def from_config(
        tokenizer,
        model_name: Optional[str] = None,
        context_length: int = 2048,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11008,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        initializer_range: float = 0.02,
        pretraining_tp: int = 1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        tie_word_embeddings: bool = False,
        model_type: str = "llama",
        trust_remote_code: bool = False,
        dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new causal language model from configuration for pretraining.
        
        Args:
            tokenizer: Tokenizer to determine vocabulary size
            model_name: Name of base model to derive config from (optional)
            context_length: Maximum sequence length (default: 2048)
            hidden_size: Size of hidden layers (default: 4096)
            num_hidden_layers: Number of transformer layers (default: 32)
            num_attention_heads: Number of attention heads (default: 32)
            num_key_value_heads: Number of key/value heads for grouped query attention (default: 32)
            intermediate_size: Size of intermediate feed-forward layer (default: 11008)
            hidden_act: Hidden layer activation function (default: "silu")
            attention_dropout: Attention dropout probability (default: 0.0)
            attention_bias: Whether to use bias in attention (default: False)
            initializer_range: Weight initialization range (default: 0.02)
            pretraining_tp: Tensor parallelism for pretraining (default: 1)
            rms_norm_eps: RMSNorm epsilon (default: 1e-6)
            rope_theta: RoPE theta parameter (default: 10000.0)
            tie_word_embeddings: Whether to tie input/output embeddings (default: False)
            model_type: Type of model architecture (default: "llama")
            trust_remote_code: Whether to trust remote code
            dtype: Model dtype (defaults to bfloat16 if supported, else float16)
            token: HuggingFace token for downloading
            tokenizer_name: Optional different name for tokenizer
        """
        if trust_remote_code:
            print(
                "Unsloth: WARNING `trust_remote_code` is True.\n"\
                "Are you certain you want to do remote code execution?"
            )

        if token is None: 
            token = get_token()
            
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast {model_type[0].title()} causal patching. Transformers = {transformers_version}.\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB. Platform = {platform_system}.\n"\
           f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit = {torch.version.cuda}.\n"\
           f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"\
           f' "-____-"     Free Apache license: http://github.com/unslothai/unsloth'
        print(statistics)

        # Warn about fast transfers
        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
            print("Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!")

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        get_statistics()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        pre_check = check_nvidia()

        # Create base config
        config = AutoConfig.from_pretrained(
            model_name if model_name is not None else model_type,
            vocab_size=len(tokenizer),
            max_position_embeddings=context_length,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            initializer_range=initializer_range,
            pretraining_tp=pretraining_tp,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=dtype,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        # Create model from config
        model = AutoModelForCausalLM.from_config(config)
        
        # Print model size
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_hf_transfer
        post_check = check_nvidia()

        # Get tokenizer if not already provided
        if tokenizer_name is None:
            tokenizer_name = model_name if model_name is not None else model_type
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                padding_side="right",
                token=token,
            )

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})

        patch_saving_functions(model, causal=True)
        patch_saving_functions(tokenizer, causal=True)

        tokenizer.padding_side = "left"  # Force inference
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model._saved_temp_tokenizer = tokenizer
            internal_model = internal_model.model
        internal_model._saved_temp_tokenizer = tokenizer
        
        return model, tokenizer, config

    @staticmethod
    def for_inference(model):
        """Prepares model for inference by disabling training features."""
        model.gradient_checkpointing = False
        model.training = False

        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            if hasattr(module, "training"):
                module.training = False

        dtype = model.config.torch_dtype
        if type(dtype) is str:
            if   dtype == "float16": dtype = torch.float16
            elif dtype == "bfloat16": dtype = torch.bfloat16
        device_type = model.device.type

        # Wrap model.generate
        if model.generate.__name__ != "_fast_generate":
            model._unwrapped_old_generate = model.generate
            model.generate = _wrap_fast_inference(model.generate, device_type, dtype, model)
        
        # Also disable training for embeddings
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = False

        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing=True):
        """Prepares model for training by enabling training features."""
        model.gradient_checkpointing = use_gradient_checkpointing
        model.training = True

        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = use_gradient_checkpointing
            if hasattr(module, "training"):
                module.training = True

        # Revert model.generate
        if hasattr(model, "_unwrapped_old_generate"):
            model.generate = model._unwrapped_old_generate
            del model._unwrapped_old_generate

        # Also re-enable training for embeddings
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True
        if hasattr(model, "get_output_embeddings"):
            embeddings = model.get_output_embeddings()
            if hasattr(embeddings, "training"): embeddings.training = True

        return model
