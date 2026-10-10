// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/skills-api.ts");
const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

test("bulk changes go through one PUT on the collection and publish its list", () => {
  const body = API.slice(
    API.indexOf("export async function setAllSkillsEnabled"),
    API.indexOf("export const SKILL_NAME_PATTERN"),
  );
  assert.match(body, /enabled: boolean \| null/);
  // sign-out drops the response; otherwise it supersedes any earlier read.
  assert.match(body, /const epoch = sessionEpoch;/);
  assert.match(body, /if \(epoch !== sessionEpoch\) return skills;\s*\/\/[^\n]*\n\s*requestGeneration \+= 1;/);
  assert.doesNotMatch(body, /\+\+requestGeneration/);
  assert.match(API, /AUTH_SESSION_CLEARED_EVENT, \(\) => \{\s*sessionEpoch \+= 1;/);
  assert.match(body, /authFetch\("\/api\/skills", \{\s*method: "PUT"/);
  assert.match(body, /JSON\.stringify\(\{ enabled \}\)/);
  assert.match(body, /publish\(\{ skills, loading: false, initialized: true, error: null \}\)/);
  // match single-toggle fan-out so other tabs and the context bar follow.
  assert.match(body, /channel\?\.postMessage\("changed"\)/);
  assert.match(body, /refreshContextUsage\(\{ invalidate: true \}\)/);
});

test("enable all, disable all and reset sit in one menu beside Refresh", () => {
  assert.match(DIALOG, /onSelect=\{\(\) => void toggleAll\(true\)\}/);
  assert.match(DIALOG, /onSelect=\{\(\) => void toggleAll\(false\)\}/);
  // disable actions with nothing to change instead of allowing silent no-ops.
  assert.match(DIALOG, /disabled=\{usable\.every\(\(skill\) => skill\.enabled\)\}/);
  assert.match(DIALOG, /disabled=\{!usable\.some\(\(skill\) => skill\.enabled\)\}/);
  assert.match(DIALOG, /aria-label=\{t\("skills\.bulkActions"\)\}/);
});

test("reset discards custom choices, so it asks first", () => {
  assert.match(DIALOG, /onSelect=\{\(\) => setConfirmingReset\(true\)\}/);
  assert.match(
    DIALOG,
    /setConfirmingReset\(false\);\s*void toggleAll\(null\);/,
  );
  assert.doesNotMatch(DIALOG, /onSelect=\{\(\) => void toggleAll\(null\)\}/);
  // match destructive confirmations because reset discards user choices.
  assert.match(
    DIALOG,
    /variant="destructive"\s*onClick=\{\(\) => \{\s*setConfirmingReset\(false\);/,
  );
  // serialize row and bulk changes so response order cannot diverge from server write order.
  assert.match(DIALOG, /<SkillRow[\s\S]*?changing=\{changing !== null\}/);
  // gate the detail switch too so it cannot race a library change.
  assert.match(DIALOG, /<Switch[\s\S]*?disabled=\{[\s\S]*?changing !== null[\s\S]*?\}/);
});
