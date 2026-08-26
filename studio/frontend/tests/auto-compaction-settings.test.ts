// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  compactionStyleValue,
  ggufCompactionRequestFields,
  parseCompactionStyle,
} from "../src/features/chat/utils/auto-compaction.ts";

test("auto-compact off omits the rolling overflow field", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: false,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    {},
  );
});

test("checkpoint compaction sends truncate_oldest and no policy override", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "truncate_oldest" },
  );
});

test("a sliding window sends rolling policy and the extra-trim ratio", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0.05,
    }),
    {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: 0.05,
    },
  );
});

test("external models never opt into GGUF compaction", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: false,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0,
    }),
    {},
  );
});

test("the settings select round-trips style values", () => {
  assert.equal(compactionStyleValue("checkpoint", 0.25), "checkpoint");
  assert.equal(compactionStyleValue("rolling", 0), "rolling:0");
  assert.deepEqual(parseCompactionStyle("rolling:0.1"), {
    contextPolicy: "rolling",
    compactionHeadroomRatio: 0.1,
  });
});

test("the chat adapter still names the rolling overflow field", () => {
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(adapter, /context_overflow:\s*"truncate_oldest"/);
  assert.match(adapter, /context_policy:\s*"rolling"/);
});
