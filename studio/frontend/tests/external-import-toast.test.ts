// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A partial import (non-empty `warnings`) must not toast as success.

import assert from "node:assert/strict";
import test from "node:test";

import { describeExternalImportToast } from "../src/features/settings/lib/external-import-toast.ts";

const copy = {
  none: "none",
  upToDate: "up to date",
  one: "one",
  many: "many",
  partial: "partial",
};

test("a clean import keeps the success wording", () => {
  assert.deepEqual(
    describeExternalImportToast({ chats: 0, newChats: 0, warnings: [] }, copy),
    { kind: "success", title: "none" },
  );
  assert.deepEqual(
    describeExternalImportToast({ chats: 4, newChats: 0, warnings: [] }, copy),
    { kind: "success", title: "up to date" },
  );
  assert.deepEqual(
    describeExternalImportToast({ chats: 4, newChats: 1, warnings: [] }, copy),
    { kind: "success", title: "one" },
  );
  assert.deepEqual(
    describeExternalImportToast({ chats: 4, newChats: 3, warnings: [] }, copy),
    { kind: "success", title: "many" },
  );
});

test("unread transcripts turn the toast into a warning", () => {
  assert.deepEqual(
    describeExternalImportToast(
      {
        chats: 4,
        newChats: 0,
        warnings: ["gone.jsonl: could not be read (No such file).", "bad.jsonl: could not be read."],
      },
      copy,
    ),
    {
      kind: "warning",
      title: "partial",
      description:
        "gone.jsonl: could not be read (No such file).\nbad.jsonl: could not be read.",
    },
  );
});
