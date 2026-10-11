// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  editBeyondEndS,
  editMaxRanges,
  editUsesRanges,
  editUsesStrength,
  formatEditLimit,
  musicEditProblem,
  MUSIC_EDIT_ACTION_LABEL,
} = await import("../src/features/audio/music/music-edit-rules.ts");

type Rule = Parameters<typeof musicEditProblem>[0];
type Draft = Parameters<typeof musicEditProblem>[1];

const aceRule: Rule = {
  id: "edit",
  actions: ["repaint", "extend", "cover", "continue"],
  max_source_s: 240,
};
const stableRule: Rule = {
  id: "edit",
  actions: ["inpaint", "restyle"],
  max_ranges: 3,
};

const source = { kind: "clip" as const, id: "c1", name: "Clip", durationS: 20 };

function draft(patch: Partial<Draft> = {}): Draft {
  return {
    source,
    action: "repaint",
    ranges: [{ start_s: 2, end_s: 4 }],
    strength: null,
    extendS: 15,
    prompt: "brighter chorus",
    ...patch,
  };
}

test("a complete edit has no problem", () => {
  assert.equal(musicEditProblem(aceRule, draft(), 20), null);
  assert.equal(
    musicEditProblem(stableRule, draft({ action: "restyle", ranges: [] }), 20),
    null,
  );
});

test("each missing piece is named, in the order it is fixed", () => {
  assert.equal(
    musicEditProblem(aceRule, draft({ source: null }), null),
    "Pick a clip to edit.",
  );
  assert.equal(
    musicEditProblem(aceRule, draft({ action: null }), 20),
    "Pick what to do with the clip.",
  );
  assert.equal(
    musicEditProblem(aceRule, draft({ action: "inpaint" }), 20),
    "Pick what to do with the clip.",
  );
  assert.equal(
    musicEditProblem({ id: "edit", actions: [] }, draft(), 20),
    "This model cannot edit clips.",
  );
  assert.equal(
    musicEditProblem(aceRule, draft({ ranges: [] }), 20),
    "Select the part to change on the waveform.",
  );
  assert.equal(
    musicEditProblem(stableRule, draft({ action: "inpaint", ranges: [] }), 20),
    "Select the parts to change on the waveform.",
  );
  assert.equal(
    musicEditProblem(aceRule, draft({ prompt: "  " }), 20),
    "Describe what the new part should sound like.",
  );
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ action: "cover", ranges: [], prompt: "" }),
      20,
    ),
    "Describe the new style.",
  );
});

test("a clip past the model's limit asks for a trim", () => {
  assert.equal(
    musicEditProblem(aceRule, draft(), 300),
    "Edit clips up to 4 minutes. Trim it first.",
  );
  assert.equal(musicEditProblem(aceRule, draft(), 240), null);
  assert.equal(formatEditLimit(60), "1 minute");
  assert.equal(formatEditLimit(90), "90 s");
});

test("invalid ranges are refused", () => {
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ ranges: [{ start_s: 2, end_s: 2.1 }] }),
      20,
    ),
    "Make each selected part at least 0.25 s long.",
  );
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ ranges: [{ start_s: 21, end_s: 22 }] }),
      20,
    ),
    "A selected part starts after the clip ends. Move it inside the clip.",
  );
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({
        ranges: [
          { start_s: 1, end_s: 2 },
          { start_s: 3, end_s: 4 },
        ],
      }),
      20,
    ),
    "Select one part to change.",
  );
  assert.equal(
    musicEditProblem(
      { ...stableRule, max_ranges: 1 },
      draft({
        action: "inpaint",
        ranges: [
          { start_s: 1, end_s: 2 },
          { start_s: 3, end_s: 4 },
        ],
      }),
      20,
    ),
    "Select one part to change.",
  );
});

test("only a repaint may run past the end, and only so far", () => {
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ ranges: [{ start_s: 18, end_s: 30 }] }),
      20,
    ),
    null,
  );
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ ranges: [{ start_s: 18, end_s: 60 }] }),
      20,
    ),
    "A repaint can reach at most 30 s past the end.",
  );
  assert.equal(
    musicEditProblem(
      stableRule,
      draft({ action: "inpaint", ranges: [{ start_s: 18, end_s: 22 }] }),
      20,
    ),
    "A selected part runs past the end of the clip.",
  );
  assert.ok(editBeyondEndS("repaint") > 0);
  assert.equal(editBeyondEndS("inpaint"), 0);
  assert.equal(editBeyondEndS("extend"), 0);
});

test("extend asks for 5 to 120 seconds", () => {
  assert.equal(
    musicEditProblem(
      aceRule,
      draft({ action: "extend", ranges: [], extendS: 2 }),
      20,
    ),
    "Add between 5 and 120 seconds.",
  );
  assert.equal(
    musicEditProblem(aceRule, draft({ action: "extend", ranges: [] }), 20),
    null,
  );
});

test("which controls each action shows", () => {
  assert.equal(editUsesRanges("repaint"), true);
  assert.equal(editUsesRanges("inpaint"), true);
  assert.equal(editUsesRanges("cover"), false);
  assert.equal(editUsesStrength("cover"), true);
  assert.equal(editUsesStrength("restyle"), true);
  assert.equal(editUsesStrength("repaint"), false);
  assert.equal(editMaxRanges(stableRule, "inpaint"), 3);
  assert.equal(editMaxRanges(stableRule, "repaint"), 1);
  assert.equal(editMaxRanges(stableRule, "restyle"), 0);
  assert.equal(MUSIC_EDIT_ACTION_LABEL.continue, "Add parts");
});

test("the inputs reuse the source card and keep lyrics out of edit", () => {
  const ui = readSrc("features/audio/components/music-edit-inputs.tsx");
  assert.match(ui, /allowSavedVoice=\{false\}/);
  assert.match(ui, /label="Clip to edit"/);
  assert.match(ui, /renderWaveform=/);
  assert.match(ui, /"How much to change"/);
  assert.match(ui, /"Add seconds"/);
  assert.match(ui, /Extends the clip by/);
  assert.match(ui, /export \{ musicEditProblem \}/);
  assert.doesNotMatch(ui, /draft\.lyrics/);
});

test("clip actions follow what the loaded model's Edit mode can do", async () => {
  const { sendActionsFor } = await import(
    "../src/features/audio/music/music-edit-rules.ts"
  );
  assert.deepEqual(sendActionsFor([]), []);
  assert.deepEqual(sendActionsFor(["inpaint", "restyle"]), ["edit"]);
  assert.deepEqual(sendActionsFor(["repaint", "extend", "cover", "continue"]), [
    "edit",
    "extend",
  ]);
});
