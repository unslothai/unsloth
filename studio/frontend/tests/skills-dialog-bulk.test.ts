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
  // A sign-out during the request drops it; otherwise the server's list wins over any earlier read.
  assert.match(body, /const epoch = sessionEpoch;/);
  assert.match(body, /if \(epoch !== sessionEpoch\) return skills;\s*\/\/[^\n]*\n\s*requestGeneration \+= 1;/);
  assert.doesNotMatch(body, /\+\+requestGeneration/);
  assert.match(API, /AUTH_SESSION_CLEARED_EVENT, \(\) => \{\s*sessionEpoch \+= 1;/);
  assert.match(body, /authFetch\("\/api\/skills", \{\s*method: "PUT"/);
  assert.match(body, /JSON\.stringify\(\{ enabled \}\)/);
  assert.match(body, /publish\(\{ skills, loading: false, initialized: true, error: null \}\)/);
  // Same fan-out as a single toggle: other tabs and the context bar follow.
  assert.match(body, /channel\?\.postMessage\("changed"\)/);
  assert.match(body, /refreshContextUsage\(\{ invalidate: true \}\)/);
});

test("enable all, disable all and reset sit in one menu beside Refresh", () => {
  assert.match(DIALOG, /onSelect=\{\(\) => void toggleAll\(true\)\}/);
  assert.match(DIALOG, /onSelect=\{\(\) => void toggleAll\(false\)\}/);
  // Nothing to do is disabled rather than a silent no-op.
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
  // Styled like the dialog's other confirms that throw choices away.
  assert.match(
    DIALOG,
    /variant="destructive"\s*onClick=\{\(\) => \{\s*setConfirmingReset\(false\);/,
  );
  // Every row waits for every change: overlapping a single toggle and a bulk write could
  // otherwise let their responses publish in an order different from the server writes.
  assert.match(DIALOG, /<SkillRow[\s\S]*?changing=\{changing !== null\}/);
  // The detail switch follows the same gate, so it cannot race a change started in the library.
  assert.match(DIALOG, /<Switch[\s\S]*?disabled=\{[\s\S]*?changing !== null[\s\S]*?\}/);
});
