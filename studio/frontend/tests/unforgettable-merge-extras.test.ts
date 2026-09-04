// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { mergeUnforgettableChatExtras } from "../src/features/unforgettable/lib/merge-extras.ts";
import { isVirtualModel } from "../src/features/unforgettable/lib/virtual-model.ts";
import { SETTINGS_SEARCH_INDEX } from "../src/features/settings/settings-search.ts";

test("virtual model ids match the Apache alias", () => {
  assert.equal(isVirtualModel("unforgettable"), true);
  assert.equal(isVirtualModel("unforgettable/qwen"), true);
  assert.equal(isVirtualModel("qwen"), false);
  assert.equal(isVirtualModel(null), false);
});

test("extras only attach when the model is unforgettable", () => {
  const extras = {
    planner: "on",
    filter: "off",
    judge_model: "judge-large",
    stakes: "high",
    skip_standing: true,
    confirm_retry: false,
    adapter_id: "ada-1",
  };
  assert.deepEqual(mergeUnforgettableChatExtras("qwen", extras), {});
  const merged = mergeUnforgettableChatExtras("unforgettable", extras);
  assert.equal(merged.planner, "on");
  assert.equal(merged.filter, "off");
  assert.equal(merged.judge_model, "judge-large");
  assert.equal(merged.stakes, "high");
  assert.equal(merged.skip_standing, true);
  assert.equal(merged.confirm_retry, false);
  assert.equal(merged.adapter_id, "ada-1");
});

test("null extras emit nothing; off planner is sent so env cannot override", () => {
  assert.deepEqual(mergeUnforgettableChatExtras("unforgettable", null), {});
  assert.deepEqual(mergeUnforgettableChatExtras("unforgettable", {}), {});
  assert.deepEqual(
    mergeUnforgettableChatExtras("unforgettable", { planner: "off" }),
    { planner: "off" },
  );
});

test("settings search indexes the Unforgettable tab", () => {
  assert.ok(SETTINGS_SEARCH_INDEX.unforgettable.includes("settings.unforgettable.title"));
  assert.ok(
    SETTINGS_SEARCH_INDEX.unforgettable.includes(
      "settings.unforgettable.approver.voter",
    ),
  );
  assert.ok(
    SETTINGS_SEARCH_INDEX.unforgettable.includes(
      "settings.unforgettable.episode.filter",
    ),
  );
  assert.ok(
    SETTINGS_SEARCH_INDEX.unforgettable.includes(
      "settings.unforgettable.episode.judgeModel",
    ),
  );
});
