# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model selection excludes connection settings and row action buttons."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_model_options_only_returns_selectable_rows():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    dom_js = Path(__file__).resolve().parents[1] / "dom.js"
    script = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const buttons = [
  { label: "Connections", option: false },
  { label: "Model A", option: true },
  { label: "Model settings", option: false },
  { label: "Model B", option: true },
];
let menuOpen = true;
const menu = {
  querySelectorAll(selector) {
    if (selector === "button") return buttons;
    if (selector === "[data-model-picker-option]") {
      return buttons.filter((button) => button.option);
    }
    throw new Error(`Unexpected selector: ${selector}`);
  },
};
const context = {
  setInterval() {},
  window: { addEventListener() {} },
  document: {
    querySelector(selector) {
      // The composer picker first, then the header picker other pages keep.
      assert.ok(
        ["[data-composer-model-picker]", ".unsloth-model-selector-menu"].includes(selector),
        selector,
      );
      return menuOpen && selector === ".unsloth-model-selector-menu" ? menu : null;
    },
  },
};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
const options = context.window.__sb.dom.modelOptions();
assert.deepEqual(Array.from(options, (button) => button.label), ["Model A", "Model B"]);
menuOpen = false;
assert.equal(context.window.__sb.dom.modelOptions().length, 0);
"""
    subprocess.run([node, "-e", script, str(dom_js)], check = True, timeout = 10)
