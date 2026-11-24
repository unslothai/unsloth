# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest


def _readers(designer):
    registry = designer._seed_reader_registry
    readers = getattr(registry, "_readers", None) or getattr(registry, "readers", None)
    if isinstance(readers, dict):
        readers = list(readers.values())
    return list(readers)


def test_each_designer_attaches_seeds_on_its_own_readers(tmp_path):
    # attach() stores the source on the reader: shared readers let one account's validation read another's seed.
    pytest.importorskip("data_designer")
    from core.data_recipe.service import create_data_designer
    from data_designer.interface.data_designer import DEFAULT_SEED_READERS

    recipe = {"columns": [{"column_type": "expression", "name": "copy", "expr": "{{ text }}"}]}
    first = _readers(create_data_designer(recipe, artifact_path = str(tmp_path / "a")))
    second = _readers(create_data_designer(recipe, artifact_path = str(tmp_path / "b")))

    shared = {id(reader) for reader in DEFAULT_SEED_READERS}
    assert first and second
    assert not shared & {id(reader) for reader in first + second}
    assert not {id(reader) for reader in first} & {id(reader) for reader in second}
    assert [type(reader) for reader in first] == [type(reader) for reader in DEFAULT_SEED_READERS]
