// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Selecting a preset is persisted through a short debounce. Both update paths can
// restart the backend or the whole Tauri renderer before page-exit events flush it,
// leaving the other presets intact while the active selection falls back to Default.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const LLAMA_UPDATE_HOOK = readFileSync(
  new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
  "utf8",
);
const TAURI_UPDATE_HOOK = readFileSync(
  new URL("../src/hooks/use-tauri-update.ts", import.meta.url),
  "utf8",
);

function applyBody(source: string): string {
  const start = source.indexOf("const apply = useCallback");
  const end = source.indexOf("\n  }, [applying,", start);
  if (start < 0) {
    throw new Error("llama update apply callback is missing");
  }
  if (end <= start) {
    throw new Error("llama update apply callback boundary moved");
  }
  return source.slice(start, end);
}

function installBody(source: string): string {
  const start = source.indexOf("async function installUpdate()");
  const end = source.indexOf("\n  async function retryUpdate", start);
  if (start < 0) {
    throw new Error("Tauri installUpdate function is missing");
  }
  if (end <= start) {
    throw new Error("Tauri installUpdate function boundary moved");
  }
  return source.slice(start, end);
}

test("llama.cpp updates flush chat settings before starting the job", () => {
  const body = applyBody(LLAMA_UPDATE_HOOK);
  const flush = body.indexOf("await flushPendingChatSettings()");
  const start = body.indexOf('authFetch("/api/llama/update"');

  assert.ok(flush >= 0, "the llama update path does not flush chat settings");
  assert.ok(start >= 0, "the llama update POST is missing");
  assert.ok(
    flush < start,
    "the llama update must not start before pending chat settings are flushed",
  );
});

test("desktop updates flush chat settings before stopping or restarting anything", () => {
  const body = installBody(TAURI_UPDATE_HOOK);
  const flush = body.indexOf("await flushPendingChatSettings()");
  const updateActionIndices = [
    "crashCleanupReady()",
    "start_backend_update",
    "installDesktopUpdate()",
  ]
    .map((needle) => body.indexOf(needle))
    .filter((index) => index >= 0);

  assert.ok(flush >= 0, "the desktop update path does not flush chat settings");
  assert.ok(
    updateActionIndices.length > 0,
    "the desktop update action is missing",
  );
  const firstUpdateAction = Math.min(
    ...updateActionIndices,
  );

  assert.ok(
    flush < firstUpdateAction,
    "the desktop update must not start before pending chat settings are flushed",
  );
});
