// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Export page shows the backend's export verdict in its banner and on the greyed method
// cards. For a no-torch install that verdict says to reinstall without --no-torch, which a Desktop
// Chat only user never typed; there, like the Train hint, it points at Repair installation.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { exportUnsupportedText } = await import("../src/config/hardware-verdict.ts");

const NO_TORCH_MESSAGE =
  "This install was set up without the training stack (--no-torch), so export is disabled. Reinstall Unsloth Studio without --no-torch to enable export.";

test("a Desktop Chat only install is pointed at Repair installation", () => {
  const text = exportUnsupportedText("no_torch", NO_TORCH_MESSAGE, true);
  assert.match(text, /Use Repair installation in Settings/);
  assert.doesNotMatch(text, /--no-torch/);
});

test("without a Repair action the backend's --no-torch advice stays", () => {
  // Web, CLI install.sh --no-torch, or a Desktop attached to a terminal-started backend.
  assert.equal(exportUnsupportedText("no_torch", NO_TORCH_MESSAGE, false), NO_TORCH_MESSAGE);
});

test("other reasons keep the backend's message even when Repair is available", () => {
  const msg = "Studio is running under Rosetta (x86_64 Python), so export is disabled.";
  assert.equal(exportUnsupportedText("rosetta", msg, true), msg);
  assert.equal(exportUnsupportedText(null, msg, true), msg);
});

test("no backend message falls back to the generic accelerator text", () => {
  assert.match(exportUnsupportedText(null, null, false), /^Export requires a supported accelerator/);
});
