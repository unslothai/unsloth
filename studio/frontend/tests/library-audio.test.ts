// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { audioDetail, audioSummary, audioWorkflow, audioWorkflowOptions, runSiblings } =
  await import("../src/features/library/audio-items.ts");
const { EMPTY_FILTERS, filtersActive, matchesFilters } = await import(
  "../src/features/library/filters.ts"
);

type Item = Parameters<typeof audioDetail>[0];

function clip(id: string, audio: Partial<NonNullable<Item["audio"]>> | null): Item {
  return {
    id,
    name: `${id}.wav`,
    source: audio ? "generated" : "uploaded",
    contentType: "audio/wav",
    sizeBytes: 4,
    createdAt: 1,
    updatedAt: 1,
    fileUrl: `/files/${id}`,
    threadId: null,
    threadTitle: null,
    textOnly: false,
    favorite: false,
    folderId: null,
    openedAt: null,
    model: null,
    audio: audio && {
      workflow: "speak",
      role: "output",
      groupId: null,
      durationS: 42,
      model: "unsloth/orpheus-3b",
      mode: null,
      variation: null,
      ...audio,
    },
  };
}

const stem = (id: string, role: string, groupId = "run") =>
  clip(id, { workflow: "separate", role, groupId });

test("a clip names the Audio page workflow that made it; other audio names none", () => {
  for (const id of ["speak", "clone", "edit", "convert", "music", "separate"]) {
    assert.equal(audioWorkflow(clip(id, { workflow: id }))?.id, id);
  }
  assert.equal(audioWorkflow(clip("upload", null)), null);
  // A workflow a newer server adds falls back to the plain audio icon.
  assert.equal(audioWorkflow(clip("future", { workflow: "dub" })), null);
});

test("a stem shows its stem, music its kind and take, speech nothing extra", () => {
  assert.equal(audioDetail(stem("v", "vocals")), "Vocals");
  assert.equal(audioDetail(clip("m", { workflow: "music", mode: "song", variation: 2 })), "Song 2");
  assert.equal(audioDetail(clip("s", { workflow: "music", mode: "sfx" })), "Sound effect");
  assert.equal(audioDetail(clip("e", { workflow: "music", role: "edit", mode: "edit" })), "Edit");
  assert.equal(audioDetail(clip("t", {})), null);
  assert.deepEqual(audioSummary(stem("v", "vocals")), ["0:42", "Vocals"]);
  assert.deepEqual(audioSummary(clip("t", { durationS: null })), ["Speak"]);
  assert.deepEqual(audioSummary(clip("upload", null)), []);
});

test("a run lists its own clips in the Audio page's order", () => {
  const items = [
    stem("other", "other"),
    stem("bass", "bass"),
    stem("vocals", "vocals"),
    stem("drums", "drums"),
    stem("elsewhere", "vocals", "another-run"),
    clip("loose", {}),
  ];
  const ids = (list: Item[]) => list.map((item) => item.id);
  assert.deepEqual(ids(runSiblings(items, items[1]!)), ["vocals", "drums", "bass", "other"]);
  assert.deepEqual(runSiblings(items, items[5]!), []);
  const takes = [2, 1, 3].map((n) =>
    clip(`take${n}`, { workflow: "music", role: "variation", groupId: "song", variation: n }),
  );
  assert.deepEqual(ids(runSiblings(takes, takes[0]!)), ["take1", "take2", "take3"]);
});

test("the Workflow filter offers only workflows present and keeps only their clips", () => {
  const items = [stem("v", "vocals"), clip("speech", {}), clip("upload.mp3", null)];
  assert.deepEqual(
    audioWorkflowOptions(items).map((workflow) => workflow.id),
    ["speak", "separate"],
  );
  const filters = { ...EMPTY_FILTERS, workflows: new Set(["separate"]) };
  assert.equal(filtersActive(filters), true);
  assert.equal(filtersActive(EMPTY_FILTERS), false);
  assert.deepEqual(
    items.filter((item) => matchesFilters(item, filters, false)).map((item) => item.id),
    ["v"],
  );
  assert.equal(
    items.every((item) => matchesFilters(item, EMPTY_FILTERS, false)),
    true,
  );
});

test("View in Audio opens the clip's own workflow instead of passing through Speak", () => {
  const origin = readSrc("features/library/origin.ts");
  assert.doesNotMatch(origin, /task: "text-to-speech"/);
  assert.match(origin, /isAudioWorkflowId\(workflow\) \? \{ \.\.\.search, workflow \} : search/);
});

test("Chat with a speech fine-tune loads it on its Audio page", () => {
  const actions = readSrc("features/library/actions.ts");
  assert.match(
    actions,
    /model: model\.path,\s*loadId: model\.path,\s*workflow: audioWorkflowForAudioType\(audioType\)/,
  );
});

test("the preview plays clips with the Audio page's waveform and stops a card that is playing", () => {
  const preview = readSrc("features/library/components/library-preview.tsx");
  assert.doesNotMatch(preview, /<audio src=\{url!\} controls/);
  assert.match(preview, /<Waveform[\s\S]*?onError=\{handleMediaError\}/);
  assert.match(preview, /if \(open\) stopLibraryAudio\(\);/);
  // The card's play button sits beside the card's own button, never inside it.
  const cards = readSrc("features/library/components/library-cards.tsx");
  assert.match(cards, /<\/button>\n\s*\{select && \([\s\S]*?\)\}\n\s*\{control\}/);
});

test("a separation is deleted in one call", () => {
  const separate = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(separate, /await handleDeleteGroup\(\s*groupId,/);
  assert.match(readSrc("features/audio/api.ts"), /\/api\/inference\/audio\/gallery\/group\//);
});
