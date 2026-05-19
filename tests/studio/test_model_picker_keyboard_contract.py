# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pin the model picker keyboard navigation contract.

The Hub and Fine-tuned picker lists must expose a listbox/options shape and
keep a roving tab stop. This lets keyboard users Tab into the visible model
list once, then use ArrowUp / ArrowDown / Home / End to move between models.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PICKERS_TSX = (
    REPO / "studio/frontend/src/components/assistant-ui/model-selector/pickers.tsx"
)
SELECTOR_TSX = REPO / "studio/frontend/src/components/assistant-ui/model-selector.tsx"


def test_model_picker_rows_use_listbox_options_with_roving_tabindex():
    src = PICKERS_TSX.read_text()
    assert "function useRovingModelList" in src
    assert 'role: "listbox" as const' in src
    assert 'role: "option"' in src
    assert '"aria-activedescendant"' in src
    assert "tabIndex: optionKey === activeOptionKey ? 0 : -1" in src
    assert '"data-model-picker-option": true' in src


def test_model_picker_supports_arrow_and_home_end_keys():
    src = PICKERS_TSX.read_text()
    for key in ("ArrowDown", "ArrowUp", "Home", "End"):
        assert f'event.key === "{key}"' in src
    assert 'moveFocus(optionKey, "next")' in src
    assert 'moveFocus(optionKey, "previous")' in src
    assert 'moveFocus(optionKey, "first")' in src
    assert 'moveFocus(optionKey, "last")' in src


def test_hub_and_lora_pickers_are_wired_to_roving_listboxes():
    src = PICKERS_TSX.read_text()
    assert 'label: "Hub models"' in src
    assert 'label: "Fine-tuned models"' in src
    assert "{...hubModelList.listboxProps}" in src
    assert "{...loraModelList.listboxProps}" in src
    assert "hubModelList.getOptionProps" in src
    assert "loraModelList.getOptionProps" in src


def test_arrow_down_from_tab_enters_active_model_list():
    src = SELECTOR_TSX.read_text()
    assert "function handlePickerEntryKeyDown" in src
    assert 'event.key !== "ArrowDown"' in src
    assert '[data-model-picker-option][tabindex="0"]' in src
    assert "focusActiveModelOption(event.currentTarget)" in src
