# Voxtral fast fine-tuning support for Unsloth.
#
# Voxtral (Mistral's open speech-to-text model) is a composite architecture:
#   - audio_tower:            Whisper-style audio encoder  (VoxtralEncoder / VoxtralAttention)
#   - language_model:         Llama-architecture decoder   (LlamaModel / LlamaAttention / ...)
#   - multi_modal_projector:  audio -> text space projector
#
# Unsloth support therefore means: fast-patch the Llama language backbone with
# the existing fused Llama kernels, while leaving the audio tower untouched and
# frozen (typical Voxtral SFT trains the language backbone via LoRA only).
# LoRA targeting already excludes the audio tower: FastVisionModel.get_peft_model
# defaults to finetune_audio_layers=False.

from .llama import (
    FastLlamaModel,
    _restore_uncompiled_transformers_classes,
    _snapshot_transformers_modules,
    _record_pre_patch_changes,
)


class FastVoxtralModel:
    """Fast LoRA / fine-tuning entry point for Voxtral speech-to-text models."""

    @staticmethod
    def pre_patch():
        # The language backbone is built from transformers' native Llama classes,
        # so the standard Llama patching covers it. The audio tower uses its own
        # VoxtralAttention / VoxtralEncoderLayer classes and is never touched.
        return FastLlamaModel.pre_patch()

    @staticmethod
    def post_patch(model, *args, **kwargs):
        # Freeze the Whisper-style audio encoder: Voxtral SFT trains the language
        # backbone (via LoRA); the encoder stays fixed. Runs before PEFT so the
        # adapter only ever sees the language model + projector.
        audio_tower = getattr(getattr(model, "model", model), "audio_tower", None)
        if audio_tower is not None:
            for param in audio_tower.parameters():
                param.requires_grad = False
        return model

    @staticmethod
    def from_pretrained(model_name, *args, **kwargs):
        from unsloth.models.loader import FastModel

        # Same pre-patch protocol as FastLlamaModel.from_pretrained: snapshot the
        # transformers modules, patch the Llama classes, record the changes.
        # Bookkeeping uses FastLlamaModel because unsloth.models.llama is the
        # module whose transformers classes actually get patched.
        model_patcher = FastLlamaModel
        kwargs.pop("model_patcher", None)
        _restore_uncompiled_transformers_classes(model_patcher)
        snapshot = _snapshot_transformers_modules(model_patcher)
        FastVoxtralModel.pre_patch()
        _record_pre_patch_changes(snapshot)

        # Load through the multimodal-capable path: it resolves the correct auto
        # class (AutoModelForMultimodalLM on transformers 5) for Voxtral.
        model, tokenizer = FastModel.from_pretrained(
            model_name,
            *args,
            **kwargs,
        )

        model = FastVoxtralModel.post_patch(model)
        return model, tokenizer
