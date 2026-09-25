# Copyright 2026 Unsloth. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in merged-column SentenceTransformer ranking loss."""

import torch

try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
except ImportError:
    from sentence_transformers.losses import MultipleNegativesRankingLoss

try:
    from sentence_transformers.base.losses.merged_forward import merge_feature_batches
except ImportError:
    merge_feature_batches = None


class FastMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    """MNRL embedding anchor+positive in one forward; falls back to stock when parity is unproven."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_calls = 0
        self.fallback_calls = 0

    def forward(self, sentence_features, labels):
        features = list(sentence_features)
        parameter = next(self.model.parameters(), None)
        if (
            len(features) != 2
            or merge_feature_batches is None
            or not hasattr(self, "compute_loss_from_embeddings")
            or getattr(self, "gather_across_devices", False)
            or any(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for module in self.model.modules()
            )
            or parameter is None
            or parameter.device.type not in ("cpu", "cuda")
            or any(
                value.dtype != torch.float32 or value.device != parameter.device
                for value in self.model.parameters()
            )
            or torch.is_autocast_enabled("cuda")
            or torch.is_autocast_enabled("cpu")
        ):
            self.fallback_calls += 1
            return super().forward(features, labels)

        tensors = [feature.get("input_ids", feature.get("attention_mask")) for feature in features]
        if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 for tensor in tensors):
            self.fallback_calls += 1
            return super().forward(features, labels)
        widths = [tensor.shape[1] for tensor in tensors]
        # Padding a short query to a long document width regressed throughput.
        if max(widths) > 128 and max(widths) >= 2 * min(widths):
            self.fallback_calls += 1
            return super().forward(features, labels)

        merged = merge_feature_batches(features)
        if merged is None:
            self.fallback_calls += 1
            return super().forward(features, labels)

        rows = tensors[0].shape[0]
        embeddings = self.model(merged)["sentence_embedding"]
        if embeddings.shape[0] != 2 * rows:
            raise RuntimeError("Merged SentenceTransformer output has the wrong batch size")
        self.merged_calls += 1
        return self.compute_loss_from_embeddings(list(embeddings.split(rows, dim = 0)), labels)
