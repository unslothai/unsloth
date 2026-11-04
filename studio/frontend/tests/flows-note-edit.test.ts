// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

test("a new note stays in its editor past the first keystroke", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/frame-nodes.tsx", import.meta.url),
    "utf8",
  );
  const note = src.slice(src.indexOf("function NoteFrameImpl"));
  // The editor shows while editing or empty; focusing it must start editing, or typing one
  // character swaps it for the rendered note.
  assert.match(note, /onFocus=\{\(\) => setEditing\(true\)\}/);
});
