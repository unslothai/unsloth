// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const L = await import("../src/features/audio/tools/edit-panel-logic.ts");
const S = await import("../src/features/audio/tools/select.ts");
const P = await import("../src/features/audio/edit-policy.ts");
const A = await import("../src/features/audio/edit-adapters.ts");

type Input = Parameters<typeof P.editBlocker>[0];
type Action = { id: string; label: string };

const S2 = "Okay, I'm Cemo and what you just heard wasn't a human voice.";
const EDITED = S2.replace("human", "robot");
const LONG = Array.from({ length: 401 }, (_, i) => `w${i}`).join(" ");
const ready: Input = {
  source: { kind: "input", id: "in_1", name: "take.wav", durationS: 4.7 },
  sourceBusy: false,
  sourceExpired: false,
  sourceError: null,
  sourceDurationS: 4.7,
  transcribing: false,
  transcript: S2,
  edited: EDITED,
  mode: "words",
  delivery: { speed: 1, pitchSteps: 0 },
  panelError: null,
};

const NONE = { source: null, sourceDurationS: null };
// Mid-transcription, before any transcript.
const BLANK = { transcript: "", edited: "", transcribing: true };

// Actions as ids, or as full objects where the label matters.
const check = (rows: [Input, string | null, (string | Action)[]?][]) => {
  for (const [input, reason, actions] of rows) {
    const blocker = P.editBlocker(input);
    assert.equal(blocker?.reason ?? null, reason);
    if (actions) {
      const got = blocker?.actions ?? [];
      const ids = typeof actions[0] === "string";
      assert.deepEqual(ids ? got.map((a) => a.id) : got, actions);
    }
  }
};

test("Edit's blockers come in rail order, each with its action", () => {
  // Every step is wrong at once; fixing one reveals the next.
  const empty: Input = { ...ready, ...NONE, ...BLANK, panelError: "no" };
  const picked = { ...empty, source: ready.source, sourceDurationS: 31 };
  // A take the 30 s timer stopped can run a frame over; the server allows that too.
  const fits = { ...picked, sourceDurationS: 30.02 };
  const transcribed = { ...fits, transcribing: false };
  const typed = {
    ...transcribed,
    transcript: S2,
    edited: `${S2} `,
  };
  const changed = { ...typed, edited: EDITED };
  assert.equal(P.editBlocker(changed)?.kind, "panel");
  check([
    [empty, P.EDIT_NO_SOURCE, ["add-recording"]],
    [{ ...empty, sourceError: "Not audio." }, "Not audio."],
    // Uploading or recording: wait, rather than ask for a recording.
    [{ ...empty, sourceBusy: true }, P.EDIT_SOURCE_BUSY],
    [
      { ...picked, sourceExpired: true },
      P.EDIT_SOURCE_EXPIRED,
      [{ id: "add-recording", label: "Add it again" }],
    ],
    [picked, P.EDIT_SOURCE_TOO_LONG, ["choose-recording"]],
    [fits, P.EDIT_TRANSCRIBING],
    [transcribed, P.EDIT_TRANSCRIPT_EMPTY, ["transcribe", "type-transcript"]],
    [typed, P.EDIT_NO_CHANGES, [{ id: "focus-changes", label: "Go to ②" }]],
    // Deleting every word would send an empty text the run route refuses with a 422.
    [{ ...typed, edited: "  " }, P.EDIT_CHANGES_EMPTY],
    [changed, "no"],
    [
      { ...changed, panelError: A.FIRERED_TOO_MANY_CHANGES },
      A.FIRERED_TOO_MANY_CHANGES,
    ],
    [{ ...changed, panelError: null }, null],
    [ready, null],
    // An unknown length does not block; the server checks it again.
    [{ ...ready, sourceDurationS: null }, null],
    [{ ...ready, transcript: LONG, edited: `${LONG} x` }, A.EDIT_TOO_LONG],
  ]);
});

test("Delivery needs a speed or a pitch change, but no transcript", () => {
  const delivery: Input = { ...ready, ...BLANK, mode: "delivery" };
  const needsFireRed = "Delivery changes need FireRedAudio.";
  check([
    [delivery, P.EDIT_DELIVERY_EMPTY],
    [{ ...delivery, delivery: { speed: 1.5, pitchSteps: 0 } }, null],
    [{ ...delivery, delivery: { speed: 1, pitchSteps: 3 } }, null],
    [{ ...delivery, panelError: needsFireRed }, needsFireRed],
    [{ ...delivery, ...NONE }, P.EDIT_NO_SOURCE],
  ]);
});

const page = {
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
};
const ctx = (audio_family: string, ...audio_workflows: string[]) =>
  S.audioModelContextFor(
    { audio_type: "audiocpp_tts", audio_family, audio_workflows },
    page,
  );
type Ctx = ReturnType<typeof ctx>;
type Inputs = Parameters<typeof S.collectToolRequest>[2];
const DOTS = ctx("dots_tts", "speak", "edit");
const VEVO = ctx("vevo2", "clone", "edit");
const FIRE = ctx("firered_audio", "clone", "edit");

const core = (edited: string, mode: "words" | "delivery" = "words") => ({
  text: edited,
  edit: { transcript: S2, edited, mode, speed: 1, pitchSteps: 0 },
});
const applying = (c: Ctx, workflow = "edit") =>
  L.EDIT_PANEL_LOGIC.filter((panel) =>
    S.panelApplies(panel, workflow as "edit", c),
  );

test("each edit panel applies only to its family, and only when the model can edit", () => {
  const rows: [Ctx, string, string[]][] = [
    [DOTS, "edit", ["edit-dots_tts"]],
    [VEVO, "edit", ["edit-vevo2"]],
    [FIRE, "edit", ["edit-firered_audio"]],
    // DotTTS-MF is dots_tts but lists no edit.
    [ctx("dots_tts", "speak"), "edit", []],
    [ctx("qwen3_tts", "clone", "edit"), "edit", []],
    [DOTS, "speak", []],
    [VEVO, "clone", []],
  ];
  for (const [c, workflow, ids] of rows) {
    const got = applying(c, workflow).map((panel) => panel.id);
    assert.deepEqual(got, ids, `${c.audioFamily} on ${workflow}`);
  }
});

test("the panels claim the options the adapters set, so Advanced hides them", () => {
  const claims = (c: Ctx) => [...S.claimedOptionNames(applying(c))];
  const fire = ["template_name", "instruction"];
  assert.deepEqual(claims(FIRE), fire);
  for (const name of [...fire, "source_text", "target_text"]) {
    assert.ok(claims(DOTS).includes(name), name);
  }
});

test("collectToolRequest returns the adapter's reason and adds nothing to the request", () => {
  // Six separate changes: every other word.
  const six = "OKAY, I'm CEMO and WHAT you JUST heard WASN'T a HUMAN voice.";
  const rows: [Ctx, Inputs, string | null][] = [
    [FIRE, core(`${S2} Really.`), A.FIRERED_INSERT_AT_END],
    [FIRE, core(six), A.FIRERED_TOO_MANY_CHANGES],
    [FIRE, core(EDITED), null],
    [DOTS, core(S2.replace("human", "<robot>")), A.DOTS_BAD_CHARACTERS],
    [DOTS, core(S2, "delivery"), A.DELIVERY_NEEDS_FIRERED],
    [FIRE, core(S2, "delivery"), null],
    // Without the page's edit inputs a panel never blocks.
    [FIRE, { text: "" }, null],
  ];
  for (const [c, inputs, error] of rows) {
    const got = S.collectToolRequest(applying(c) as never, {}, inputs, c);
    assert.equal(got.error, error);
    assert.deepEqual(got.patch, {});
  }
});

test("the registry lists the edit panels and the editor keeps its label", () => {
  const registry = readSrc("features/audio/tools/registry.tsx");
  assert.match(registry, /\.\.\.EDIT_TOOL_PANELS,/);
  const editor = readSrc(
    "features/audio/components/transcript-diff-editor.tsx",
  );
  assert.match(editor, /aria-label="Changes from the transcript"/);
});
