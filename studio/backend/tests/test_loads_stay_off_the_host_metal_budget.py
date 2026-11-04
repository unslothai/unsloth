# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The suite runs on a macos-15 leg too. There the real unified-memory budget is non-zero, and
every GGUF load a test drives takes the Metal context-fit path, which the tests were not
written for. conftest pins the budget to 0; these check the pin holds on any host."""

import utils.hardware as hardware
import utils.hardware.hardware as hardware_impl
from core.inference.llama_cpp import LlamaCppBackend


def _host_is_apple_silicon(monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hardware_impl, "_metal_recommended_working_set_bytes", lambda: 16 * 1024**3)


def test_a_load_on_apple_silicon_sees_no_metal_budget_by_default(monkeypatch):
    _host_is_apple_silicon(monkeypatch)
    assert LlamaCppBackend._apple_metal_memory_budget_bytes() == 0


def test_the_real_probe_is_back_when_a_test_asks_for_it(monkeypatch, real_apple_metal_budget):
    _host_is_apple_silicon(monkeypatch)
    assert LlamaCppBackend._apple_metal_memory_budget_bytes() > 0
