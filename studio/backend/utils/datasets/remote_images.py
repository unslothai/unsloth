# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep ``datasets`` from fetching untrusted Image() URL rows on its own, outside the SSRF guard."""


def is_remote_image_path(path):
    # hf:// paths resolve against the Hub itself; any other scheme is a fetch the row chose.
    return isinstance(path, str) and "://" in path and not path.startswith("hf://")


def undecode_remote_image_columns(dataset, columns = None):
    """Image() columns self-decode on access, so a bytes-less URL row is fetched unguarded. Turn
    decoding off for exactly the columns holding such rows; any other dataset is returned as is."""
    try:
        from datasets import Dataset
        from datasets import Image as ImageFeature

        if not isinstance(dataset, Dataset):
            return dataset
        for name, feature in dataset.features.items():
            if columns is not None and name not in columns:
                continue
            if not isinstance(feature, ImageFeature) or not feature.decode:
                continue
            for chunk in dataset.data.column(name).chunks:
                paths = chunk.field("path").to_pylist()
                missing = chunk.field("bytes").is_null().to_pylist()
                if any(m and is_remote_image_path(p) for p, m in zip(paths, missing)):
                    dataset = dataset.cast_column(name, ImageFeature(decode = False))
                    break
        return dataset
    except Exception:
        return dataset


def is_remote_image_cell(value):
    return (
        isinstance(value, dict)
        and not value.get("bytes")
        and is_remote_image_path(value.get("path"))
    )
