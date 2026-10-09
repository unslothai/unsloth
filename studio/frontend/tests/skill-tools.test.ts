// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { skillToolNames } from "../src/features/chat/api/skill-tools.ts";

const enabled = { name: "guided", valid: true, shadowed: false, enabled: true };

test("Code on offers both skill tools for any usable skill", () => {
  assert.deepEqual(skillToolNames([enabled], true, []), ["read_skill", "create_skill"]);
});

test("Code off keeps a plain chat out of the tool loop (#11671)", () => {
  assert.deepEqual(skillToolNames([enabled], false, []), []);
  assert.deepEqual(skillToolNames([enabled], false, ["what is guided?"]), []);
  // Not a mention: no whitespace before the @, or not a spec-shaped name.
  assert.deepEqual(skillToolNames([enabled], false, ["mail me@guided"]), []);
  assert.deepEqual(skillToolNames([enabled], false, ["@Guided"]), []);
  // The backend would not preload these, so they must not open the loop either.
  assert.deepEqual(skillToolNames([enabled], false, ["use @guided!!"]), []);
  assert.deepEqual(skillToolNames([enabled], false, ["[@guided]"]), []);
});

test("Code off still reads a skill the user @mentions, but never creates one", () => {
  assert.deepEqual(skillToolNames([enabled], false, ["use @guided please"]), ["read_skill"]);
  // An earlier turn's mention keeps it readable: the preloaded SKILL.md is not replayed.
  assert.deepEqual(skillToolNames([enabled], false, ["@guided", "and now?"]), ["read_skill"]);
});

test("a mention of an unusable or unknown skill offers nothing with Code off", () => {
  for (const skill of [
    { ...enabled, enabled: false },
    { ...enabled, shadowed: true },
    { ...enabled, valid: false },
  ]) {
    assert.deepEqual(skillToolNames([skill], false, ["@guided"]), []);
  }
  assert.deepEqual(skillToolNames([enabled], false, ["@someone-else"]), []);
});

test("Code alone offers nothing without a usable skill", () => {
  assert.deepEqual(skillToolNames([], true, ["@guided"]), []);
  assert.deepEqual(skillToolNames([{ ...enabled, enabled: false }], true, []), []);
  assert.deepEqual(skillToolNames([{ ...enabled, shadowed: true }], true, []), []);
  assert.deepEqual(skillToolNames([{ ...enabled, valid: false }], true, []), []);
});
