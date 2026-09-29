// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { skillToolsOffered } from "../src/features/chat/api/skill-tools.ts";

const enabled = { valid: true, shadowed: false, enabled: true };

test("an enabled skill is offered only while Code is on (#11671)", () => {
  assert.equal(skillToolsOffered([enabled], true), true);
  assert.equal(skillToolsOffered([enabled], false), false);
});

test("Code alone offers nothing without a usable skill", () => {
  assert.equal(skillToolsOffered([], true), false);
  assert.equal(skillToolsOffered([{ ...enabled, enabled: false }], true), false);
  assert.equal(skillToolsOffered([{ ...enabled, shadowed: true }], true), false);
  assert.equal(skillToolsOffered([{ ...enabled, valid: false }], true), false);
});
