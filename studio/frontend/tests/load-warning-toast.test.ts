// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { LOAD_WARNING_TOAST_ID, showLoadWarning } = await import(
  "../src/features/chat/utils/load-warning-toast.ts"
);

const NOTICE =
  "Not enough disk space to download BF16 (7.5 GB needed, 7.5 GB free), so Q4_1 (2.4 GB) was loaded instead.";
const FROM_LOAD_RESPONSE = /showLoadWarning\(loaded\.memory_warning\);/;
const FROM_STATUS_ON_MODEL_CHANGE =
  /if \(hydratingExistingModel\) \{\s*showLoadWarning\(status\.memory_warning\);/;

test("a load warning raises one warning toast carrying the backend's text", () => {
  calls.length = 0;
  showLoadWarning(NOTICE);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].kind, "warning");
  assert.equal(calls[0].options?.id, LOAD_WARNING_TOAST_ID);
  assert.equal(calls[0].options?.description, NOTICE);
});

test("a model with no warning takes the previous model's warning down", () => {
  calls.length = 0;
  showLoadWarning(null);
  showLoadWarning(undefined);
  assert.deepEqual(calls, [
    { kind: "dismiss", id: LOAD_WARNING_TOAST_ID },
    { kind: "dismiss", id: LOAD_WARNING_TOAST_ID },
  ]);
});

test("loads from this tab and models loaded elsewhere both reach the toast", () => {
  assert.match(readSrc("features/chat/api/chat-api.ts"), FROM_LOAD_RESPONSE);
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    FROM_STATUS_ON_MODEL_CHANGE,
  );
});
