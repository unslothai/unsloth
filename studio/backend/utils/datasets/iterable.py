# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for dataset iterable detection."""


def is_streaming_dataset(dataset) -> bool:
    """Return True for iterable datasets that do not support eager map kwargs."""
    try:
        from datasets import IterableDataset as HfIterableDataset
        if isinstance(dataset, HfIterableDataset):
            return True
    except ImportError:
        pass

    try:
        from torch.utils.data import IterableDataset as TorchIterableDataset
        return isinstance(dataset, TorchIterableDataset)
    except ImportError:
        return False
