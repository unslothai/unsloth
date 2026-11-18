// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  apiCompactionRequestFields,
  normalizeCompactionMode,
} from "../src/features/chat/utils/auto-compaction.ts";

import { readSrc } from "./helpers/kit.ts";

test("both evicting modes are sent on auto-compaction and on /compact", () => {
  for (const compactionMode of ["rolling", "checkpoint"] as const) {
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: true,
      contextLength: 128_000,
      compactionMode,
    }),
    {
      context_overflow: "truncate_oldest",
      compaction_threshold: 96_000,
      context_window: 128_000,
      compaction_mode: compactionMode,
    },
  );
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: false,
      contextLength: null,
      compactionMode,
      compactNow: true,
    }),
    {
      context_overflow: "truncate_oldest",
      compact_context: true,
      compaction_mode: compactionMode,
    },
  );
  }
});

test("the default summary mode sends no compaction_mode; auto-compact off sends only the mode", () => {
  for (const compactionMode of [undefined, null, "summary"] as const) {
    assert.equal(
      "compaction_mode" in
        apiCompactionRequestFields({
          autoCompactEnabled: true,
          contextLength: 128_000,
          compactionMode,
        }),
      false,
    );
  }
  assert.deepEqual(
    apiCompactionRequestFields({
      autoCompactEnabled: false,
      contextLength: 128_000,
      compactionMode: "checkpoint",
    }),
    // A /compact epoch must stay in force on later turns with auto-compact off.
    { compaction_mode: "checkpoint" },
  );
  assert.deepEqual(
    apiCompactionRequestFields({ autoCompactEnabled: false, contextLength: 128_000 }),
    {},
  );
});

test("only the three known modes survive a saved or synced connection", () => {
  for (const mode of ["summary", "rolling", "checkpoint"] as const) {
    assert.equal(normalizeCompactionMode(mode), mode);
  }
  for (const value of [undefined, null, "", "evict", "drop", 1, true]) {
    assert.equal(normalizeCompactionMode(value), undefined);
  }
});

test("the connection's mode reaches the request and the saved connection", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /\?\.contextLength,\s*compactionMode: externalProvider\.compactionMode,\s*autoCompactTokens: externalProvider\.autoCompactTokens,/,
  );
  const sync = readSrc("features/chat/sync-external-providers.ts");
  assert.match(sync, /compactionMode: normalizeCompactionMode\(config\.compaction_mode\)/);
  const api = readSrc("features/chat/api/providers-api.ts");
  assert.equal(api.match(/compaction_mode: payload\.compactionMode/g)?.length, 2);
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  assert.match(dialog, /<SelectItem value="rolling">Drop oldest turns<\/SelectItem>/);
  assert.match(dialog, /<SelectItem value="checkpoint">Reset to latest turn<\/SelectItem>/);
  // Editing back to the default clears the column instead of leaving a mode behind.
  assert.match(
    dialog,
    /compactionMode: supportsAutoCompactTokens\s*\?\s*\(compactionModeDraft === "summary" \? null : compactionModeDraft\)/,
  );
});
