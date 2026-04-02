# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))

        if "model" in weights:
            model_weights = weights["model"]
        else:
            model_weights = weights

        for key in [
            "vision_tower",
            "embed_vision",
            "audio_tower",
            "embed_audio",
        ]:
            model_weights.pop(key, None)

        if "language_model" in model_weights:
            source_lm_weights = dict(tree_flatten(model_weights["language_model"]))
        else:
            source_lm_weights = dict(tree_flatten(model_weights))

        lm_weights = {}
        for key, value in source_lm_weights.items():
            if key.startswith("model.") or key.startswith("lm_head."):
                lm_weights[key] = value
            else:
                lm_weights[f"model.{key}"] = value

        lm_head = model_weights.get("lm_head", weights.get("lm_head"))
        if isinstance(lm_head, dict) and "weight" in lm_head:
            lm_weights["lm_head.weight"] = lm_head["weight"]

        lm_weights = self.language_model.sanitize(lm_weights)
        return {f"language_model.{key}": value for key, value in lm_weights.items()}

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
