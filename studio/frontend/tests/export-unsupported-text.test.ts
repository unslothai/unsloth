// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { exportUnsupportedText } = await import("../src/features/export/lib/unsupported-text.ts");

const NO_TORCH_MESSAGE =
  "This install was set up without the training stack (--no-torch), so export is disabled. Reinstall Unsloth Studio without --no-torch to enable export.";

test("a Desktop Chat only install is pointed at the full install upgrade", () => {
  const text = exportUnsupportedText("no_torch", NO_TORCH_MESSAGE, true);
  assert.match(text, /Use Upgrade to full install in Settings/);
  assert.doesNotMatch(text, /--no-torch/);
});

test("without the Install type row the backend's message stays", () => {
  // Web, CLI --no-torch, a terminal-started backend, or a Mac the MLX wheels skip.
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

test("a successful repair drops the cached hardware verdict before the backend restarts", async () => {
  // The hook cannot be rendered here, so the order is read off the source as in
  // repair-single-flight.test.ts.
  const hook = await readSrcAsync("hooks/use-tauri-backend.ts");
  const body = hook.slice(hook.indexOf("async function runRepair("), hook.indexOf("async function startServer()"));
  const native = body.indexOf('invoke("start_managed_repair"');
  const invalidate = body.indexOf("invalidateHardwareInfo();");
  const start = body.indexOf("await startManagedServer();");
  assert.ok(native > 0 && invalidate > native && start > invalidate);
});
