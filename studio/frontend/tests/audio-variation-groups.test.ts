// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { groupHistory, historyRows, variationLabel, variationSiblings } =
  await import("../src/features/audio/music/variation-groups.ts");

const clip = (id: string, group_id?: string | null, variation?: number) => ({
  id,
  group_id,
  settings: variation === undefined ? null : { variation },
});

const clips = [
  clip("a"),
  clip("v3", "g1", 3),
  clip("v1", "g1", 1),
  clip("b", "solo"),
  clip("v2", "g1", 2),
  clip("c", null),
];

test("a group of variations is one row where its first clip was", () => {
  const entries = groupHistory(clips);
  assert.deepEqual(
    entries.map((entry) =>
      entry.kind === "clip" ? entry.clip.id : `group:${entry.groupId}`,
    ),
    ["a", "group:g1", "b", "c"],
  );
  const group = entries[1];
  assert.equal(group.kind, "group");
  if (group.kind === "group") {
    assert.deepEqual(
      group.clips.map((item) => item.id),
      ["v1", "v2", "v3"],
    );
  }
});

test("a group of one, or clips without a group, render as before", () => {
  assert.deepEqual(
    groupHistory([clip("x", "only"), clip("y")]).map((entry) => entry.kind),
    ["clip", "clip"],
  );
  assert.deepEqual(groupHistory([]), []);
});

test("without recorded variation numbers the listed order stands", () => {
  const entries = groupHistory([
    clip("p", "g"),
    clip("q", "g"),
    clip("r", "g"),
  ]);
  assert.equal(entries.length, 1);
  const group = entries[0];
  if (group.kind === "group") {
    assert.deepEqual(
      group.clips.map((item) => item.id),
      ["p", "q", "r"],
    );
  }
});

test("labels and siblings", () => {
  assert.equal(variationLabel(3), "3 variations");
  assert.equal(variationLabel(1), "1 variation");
  assert.deepEqual(
    variationSiblings(clips, "v3").map((item) => item.id),
    ["v1", "v2", "v3"],
  );
  assert.deepEqual(variationSiblings(clips, "b"), []);
  assert.deepEqual(variationSiblings(clips, "a"), []);
  assert.deepEqual(variationSiblings(clips, null), []);
  assert.deepEqual(variationSiblings(clips, "missing"), []);
});

test("an open group lists its clips nested under the header", () => {
  const closed = historyRows(clips, new Set());
  assert.equal(closed.length, 4);
  assert.equal(closed[1].header?.open, false);
  const open = historyRows(clips, new Set(["g1"]));
  assert.deepEqual(
    open.map((row) =>
      row.header ? "header" : `${row.clip.id}${row.nested ? "*" : ""}`,
    ),
    ["a", "header", "v1*", "v2*", "v3*", "b", "c"],
  );
});

test("TtsOutput groups history rows and offers Send to for music clips", () => {
  const source = readSrc("features/audio/pages/tts-workspace.tsx");
  assert.match(source, /historyRows\(clips, openGroups\)/);
  assert.match(source, /<VariationChips/);
  assert.match(source, /<MusicSendToMenuItems clip=\{clip\} \/>/);
  assert.match(source, /menu=\{clipMenu\(selectedClip, "row"\)\}/);
  const send = readSrc("features/audio/components/music-send-to.tsx");
  assert.match(send, /pushClipToEdit\(/);
  assert.match(send, /edit: "Edit"/);
  assert.match(send, /extend: "Extend"/);
  assert.match(send, /sendActionsFor\(editActions\)/);
  assert.match(send, /clipWorkflow\(clip\) === "music"/);
  assert.match(send, /requestWorkflow\("music"\)/);
});
