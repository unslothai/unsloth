import torch
import types 
from transformers import AutoModelForCausalLM, AutoConfig
from ._utils import (
    __version__, 
    is_bfloat16_supported, 
    HAS_FLASH_ATTENTION, 
    check_nvidia,
    get_statistics,
)
from .llama import * 

from ..kernels import post_patch_loss_function
from typing import Optional
import os
from transformers import __version__ as transformers_version
try:
    from huggingface_hub.utils import get_token
except:
    from huggingface_hub.utils._token import get_token

__all__ = ["FastBaseCausalModel"]

def _wrap_fast_inference(generate, device_type, dtype, model):
    @torch.inference_mode
    def _fast_generate(*args, **kwargs):
        kwargs.pop("token_type_ids", None)
        model_eos_token_id = getattr(model.config, "eos_token_id", None)
        if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
            model_eos_token_id = model_eos_token_id[0]
        kwargs["pad_token_id"] = kwargs.pop("pad_token_id", model_eos_token_id)
        with torch.autocast(device_type=device_type, dtype=dtype):
            output = generate(*args, **kwargs)
        return output
    return _fast_generate

class FastBaseCausalModel:
    @staticmethod
    def from_config(
        tokenizer,
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
        model_type: str = "meta-llama/Llama-2-7b-hf",
        trust_remote_code: bool = False,
        dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,
        **kwargs,
    ):
        """Creates a new causal language model from configuration."""
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
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB.\n"\
           f"O^O/ \_/ \\    Pytorch: {torch.__version__}. CUDA = {gpu_stats.major}.{gpu_stats.minor}.\n"\
           f"\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA2 = {HAS_FLASH_ATTENTION}\n"\
           f' "-____-"     Free Apache license: http://github.com/unslothai/unsloth'
        print(statistics)

        old_hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        get_statistics()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            print("Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        pre_check = check_nvidia()

        # Create base config
        config = AutoConfig.from_pretrained(
            model_type,
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

        model, tokenizer = patch_tokenizer(model, tokenizer)
        model = post_patch_loss_function(model)
        if hasattr(model, "config"):
            model.config.update({"unsloth_version" : __version__})
        
        # Add Llama optimizations
        for layer in model.model.layers:
            # Patch attention with fast Llama implementation
            layer.self_attn.forward = types.MethodType(LlamaAttention_fast_forward, layer.self_attn)
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o = original_apply_o
            
            # Patch MLP with fast SwiGLU
            layer.mlp.forward = types.MethodType(fast_swiglu_inference, layer.mlp)
            
            # Patch LayerNorm with fast RMS norm
            if hasattr(layer, 'input_layernorm'):
                layer.input_layernorm.forward = types.MethodType(fast_rms_layernorm_inference, layer.input_layernorm)
            if hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm.forward = types.MethodType(fast_rms_layernorm_inference, layer.post_attention_layernorm)

        # Add training/inference mode switching
        model.for_inference = types.MethodType(FastLlamaModel.for_inference, model)
        model.for_training = types.MethodType(FastLlamaModel.for_training, model)
        
        # TO DO : add patch saving for causal 
        #patch_saving_functions(model, causal=True)
        #patch_saving_functions(tokenizer, causal=True)

        tokenizer.padding_side = "left"  # Force inference
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model._saved_temp_tokenizer = tokenizer
            internal_model = internal_model.model
        internal_model._saved_temp_tokenizer = tokenizer
        
        return model, tokenizer, config