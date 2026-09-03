// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DEFAULT_CONTEXT_POLICY,
  compactionStyleValue,
  ggufCompactionRequestFields,
  parseCompactionStyle,
  sanitizeCompactionHeadroomRatio,
} from "../src/features/chat/utils/auto-compaction.ts";

test("the default preserves the server context policy", () => {
  assert.equal(DEFAULT_CONTEXT_POLICY, "inherit");
  assert.equal(compactionStyleValue("inherit", 0.25), "inherit");
  assert.deepEqual(parseCompactionStyle("inherit"), {
    contextPolicy: "inherit",
    compactionHeadroomRatio: 0.25,
  });
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "inherit",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "truncate_oldest" },
  );
});

test("auto-compact off sends an explicit error overflow policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: false,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "error" },
  );
});

test("checkpoint compaction sends truncate_oldest and the checkpoint policy", () => {
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "checkpoint",
      compactionHeadroomRatio: 0.25,
    }),
    { context_overflow: "truncate_oldest", context_policy: "checkpoint" },
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

test("unsupported headroom ratios snap to an exposed choice", () => {
  assert.equal(sanitizeCompactionHeadroomRatio(0.9), 0.25);
  assert.equal(sanitizeCompactionHeadroomRatio(0.07), 0.05);
  assert.equal(compactionStyleValue("rolling", 0.9), "rolling:0.25");
  assert.deepEqual(
    ggufCompactionRequestFields({
      isGguf: true,
      autoCompactEnabled: true,
      contextPolicy: "rolling",
      compactionHeadroomRatio: 0.9,
    }),
    {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: 0.25,
    },
  );
});

test("the chat adapter sends compaction fields through the shared helper", () => {
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(adapter, /ggufCompactionRequestFields\(/);
});
