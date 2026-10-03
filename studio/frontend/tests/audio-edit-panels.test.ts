// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  AudioModelContext,
  CoreInputs,
} from "../src/features/audio/tools/types.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { EDIT_PANEL_LOGIC } = await import(
  "../src/features/audio/tools/edit-panel-logic.ts"
);
const { claimedOptionNames, collectToolRequest, panelApplies } = await import(
  "../src/features/audio/tools/select.ts"
);
const {
  DELIVERY_NEEDS_FIRERED,
  DOTS_BAD_CHARACTERS,
  FIRERED_INSERT_AT_END,
  FIRERED_TOO_MANY_CHANGES,
} = await import("../src/features/audio/edit-adapters.ts");

const ctx = (
  audioFamily: string | null,
  audioWorkflows: string[],
): AudioModelContext => ({
  audioType: "audiocpp_tts",
  audioFamily,
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
  audioWorkflows,
  requiredInputs: [],
  referenceTextMode: null,
});

const S2 = "Okay, I'm Cemo and what you just heard wasn't a human voice.";
const core = (
  edited: string,
  mode: "words" | "delivery" = "words",
): CoreInputs => ({
  text: edited,
  edit: { transcript: S2, edited, mode, speed: 1, pitchSteps: 0 },
});

const shown = (
  c: AudioModelContext,
  workflow: "edit" | "speak" | "clone" = "edit",
) =>
  EDIT_PANEL_LOGIC.filter((panel) => panelApplies(panel, workflow, c)).map(
    (panel) => panel.id,
  );

test("each edit panel applies only to its family, and only when the model can edit", () => {
  assert.deepEqual(shown(ctx("dots_tts", ["speak", "edit"])), [
    "edit-dots_tts",
  ]);
  assert.deepEqual(shown(ctx("vevo2", ["clone", "edit"])), ["edit-vevo2"]);
  assert.deepEqual(shown(ctx("firered_audio", ["clone", "edit"])), [
    "edit-firered_audio",
  ]);
  // DotTTS-MF is dots_tts but lists no edit.
  assert.deepEqual(shown(ctx("dots_tts", ["speak"])), []);
  assert.deepEqual(shown(ctx("qwen3_tts", ["clone", "edit"])), []);
  // Never on Speak or Clone, even for an edit model.
  assert.deepEqual(shown(ctx("dots_tts", ["speak", "edit"]), "speak"), []);
  assert.deepEqual(shown(ctx("vevo2", ["clone", "edit"]), "clone"), []);
});

test("the panels claim the options the adapters set, so Advanced hides them", () => {
  const claims = (family: string) => [
    ...claimedOptionNames(
      EDIT_PANEL_LOGIC.filter((p) => p.families.includes(family)),
    ),
  ];
  assert.deepEqual(claims("firered_audio"), ["template_name", "instruction"]);
  for (const name of [
    "template_name",
    "instruction",
    "source_text",
    "target_text",
  ]) {
    assert.ok(claims("dots_tts").includes(name), name);
  }
});

test("collectToolRequest returns the adapter's reason and adds nothing to the request", () => {
  const fire = ctx("firered_audio", ["clone", "edit"]);
  const dots = ctx("dots_tts", ["speak", "edit"]);
  const check = (c: AudioModelContext, inputs: CoreInputs) =>
    collectToolRequest(
      EDIT_PANEL_LOGIC.filter((panel) =>
        panelApplies(panel, "edit", c),
      ) as never,
      {},
      inputs,
      c,
    );
  const atEnd = check(fire, core(`${S2} Really.`));
  assert.equal(atEnd.error, FIRERED_INSERT_AT_END);
  assert.deepEqual(atEnd.patch, {});
  // Six separate changes: every other word.
  const six = "OKAY, I'm CEMO and WHAT you JUST heard WASN'T a HUMAN voice.";
  const cases: [AudioModelContext, CoreInputs, string | null][] = [
    [fire, core(six), FIRERED_TOO_MANY_CHANGES],
    [fire, core(S2.replace("human", "robot")), null],
    [dots, core(S2.replace("human", "<robot>")), DOTS_BAD_CHARACTERS],
    [dots, core(S2, "delivery"), DELIVERY_NEEDS_FIRERED],
    [fire, core(S2, "delivery"), null],
    // Without the page's edit inputs a panel never blocks.
    [fire, { text: "" }, null],
  ];
  for (const [c, inputs, error] of cases) {
    assert.equal(check(c, inputs).error, error);
  }
});

test("the registry lists the edit panels and the panel keeps its labels", () => {
  const registry = readSrc("features/audio/tools/registry.tsx");
  assert.match(registry, /\.\.\.EDIT_TOOL_PANELS,/);
  const editor = readSrc(
    "features/audio/components/transcript-diff-editor.tsx",
  );
  assert.match(editor, /aria-label="Changes from the transcript"/);
  assert.match(
    editor,
    /<ins className="[^"]*underline[^"]*"|<ins className="[^"]*bg-secondary/,
  );
  assert.match(editor, /<del className="[^"]*line-through/);
  assert.match(editor, /sr-only/);
});
