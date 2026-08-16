// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The load response says `disable_vision: false` for an MLX / safetensors model
// however the request was set, because the response field means "the projector
// WAS deliberately left unloaded" and on those backends it was not. The reason
// that is not a switch disagreeing with a response is asserted here: there is no
// switch. The Vision control exists only inside the GGUF half of Advanced
// Settings, so a non-GGUF target never renders it.
//
// Companion to studio/backend/tests/test_non_gguf_vision_fields_report_absent.py,
// which pins the response side. Structural (the repo has no render harness), so
// it reads the source of the one component that owns the control.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const SRC = new URL(
  "../src/features/model-picker/components/model-config-page.tsx",
  import.meta.url,
);
const text = readFileSync(SRC, "utf8");
const lines = text.split("\n");

/** Body of a top-level `function Name(`, up to the next top-level function. */
function bodyOf(name: string): string {
  const start = lines.findIndex((line) => line.startsWith(`function ${name}(`));
  assert.notEqual(start, -1, `no top-level function ${name}`);
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    if (/^(export )?function \w+\(/.test(lines[i])) {
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join("\n");
}

test("the Vision switch lives only inside GgufAdvancedSettings", () => {
  const gguf = bodyOf("GgufAdvancedSettings");
  const mlx = bodyOf("MlxAdvancedSettings");

  // By its exact wiring, so a second switch added elsewhere fails here rather
  // than quietly widening the control's reach.
  const wiring = "checked={!config.disableVision}";
  assert.equal(
    text.split(wiring).length - 1,
    1,
    "expected exactly one Vision switch in the file",
  );
  assert.ok(gguf.includes(wiring), "Vision switch is not in GgufAdvancedSettings");
  assert.ok(!mlx.includes(wiring), "Vision switch leaked into MlxAdvancedSettings");
  assert.ok(
    !mlx.includes("disableVision"),
    "MlxAdvancedSettings reads disableVision, which it cannot act on",
  );
  assert.ok(gguf.includes(">Vision</span>"));
  assert.ok(!mlx.includes(">Vision</span>"));
});

test("GgufAdvancedSettings renders only under target.isGguf", () => {
  const ggufAt = lines.findIndex((line) => line.includes("<GgufAdvancedSettings"));
  const mlxAt = lines.findIndex((line) => line.includes("<MlxAdvancedSettings"));
  assert.ok(ggufAt > 0, "GgufAdvancedSettings is never rendered");
  assert.ok(mlxAt > 0, "MlxAdvancedSettings is never rendered");

  // Nearest enclosing isGguf gate above each render site.
  const gateAbove = (index: number): string => {
    for (let i = index; i >= 0; i--) {
      const line = lines[i].trim();
      if (line === "{target.isGguf && (") return "isGguf";
      if (line === "{!target.isGguf && (") return "!isGguf";
    }
    return "none";
  };
  assert.equal(gateAbove(ggufAt), "isGguf");
  assert.equal(gateAbove(mlxAt), "!isGguf");

  // One Vision switch exists, it is inside GgufAdvancedSettings, and
  // GgufAdvancedSettings renders only when target.isGguf. So a non-GGUF target
  // shows no Vision switch, and `disable_vision: false` in its load response
  // contradicts nothing the user can see.
});

test("the store scrubs disableVision to false on every non-GGUF load", () => {
  // The other half of "nothing disagrees": the client does not keep a stale
  // true from a previous GGUF and show it against an MLX model.
  const runtime = readFileSync(
    new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
    "utf8",
  );
  assert.match(runtime, /disableVision:\s*loadResponse\.disable_vision \?\? false/);
});
