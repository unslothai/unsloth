// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DECISION_TYPES,
  PRESETS,
  buildRequest,
  draftFromRequest,
  initialDrafts,
  presetsFor,
  requestText,
  scoreLevels,
  stateValue,
} from "../src/features/settings/lib/decision-request.ts";
import { readSrc } from "./helpers/kit.ts";

const BLANK = {
  name: "check",
  state: "Charged twice",
  instructions: " Refund it? ",
  yesWhen: "",
  noWhen: "",
  options: [],
  levels: [],
};

test("yes/no criteria are sent as true and false, and only when written", () => {
  assert.deepEqual(buildRequest("noul", BLANK, "default"), {
    state: "Charged twice",
    model: "default",
    questions: { check: { type: "noul", instructions: "Refund it?" } },
  });
  const both = buildRequest(
    "noul",
    { ...BLANK, yesWhen: " A duplicate charge ", noWhen: "A single charge" },
    "laya-english",
  );
  assert.equal(both.model, "laya-english");
  assert.deepEqual(both.questions.check.criteria, {
    true: "A duplicate charge",
    false: "A single charge",
  });
  const yesOnly = buildRequest(
    "noul",
    { ...BLANK, yesWhen: "Duplicate" },
    "default",
  );
  assert.deepEqual(yesOnly.questions.check.criteria, { true: "Duplicate" });
});

test("choice options become a name to description map, skipping unnamed rows", () => {
  const request = buildRequest(
    "choice",
    {
      ...BLANK,
      options: [
        { name: " billing ", description: " Charges " },
        { name: "technical", description: "" },
        { name: " ", description: "ignored" },
      ],
    },
    "default",
  );
  assert.deepEqual(request.questions.check, {
    type: "choice",
    instructions: "Refund it?",
    criteria: { billing: "Charges", technical: "" },
  });
});

test("score levels are sent lowest first without blanks", () => {
  const request = buildRequest(
    "score",
    { ...BLANK, levels: ["calm", " ", " annoyed ", "furious"] },
    "default",
  );
  assert.deepEqual(request.questions.check.criteria, [
    "calm",
    "annoyed",
    "furious",
  ]);
});

test("JSON state is sent as an object or array, anything else as text", () => {
  assert.deepEqual(stateValue('{"rating": 2}'), { rating: 2 });
  assert.deepEqual(stateValue(" [1, 2] "), [1, 2]);
  assert.equal(stateValue("{not json"), "{not json");
  assert.equal(stateValue("42"), "42");
});

test("the JSON view reads back into the form for every type", () => {
  for (const preset of PRESETS) {
    const request = buildRequest(preset.type, preset.draft, "laya-english");
    const restored = draftFromRequest(JSON.parse(requestText(request)));
    assert.ok(restored, preset.id);
    assert.equal(restored.type, preset.type, preset.id);
    assert.equal(restored.model, "laya-english");
    assert.deepEqual(
      buildRequest(restored.type, restored.draft, "laya-english"),
      request,
      preset.id,
    );
  }
});

test("hand-edited JSON switches the form to the question's type", () => {
  const restored = draftFromRequest({
    state: { ticket: 7 },
    model: "default",
    questions: { mood: { type: "score", criteria: ["calm", "upset"] } },
  });
  assert.ok(restored);
  assert.equal(restored.type, "score");
  assert.equal(restored.draft.name, "mood");
  assert.equal(restored.draft.state, '{\n  "ticket": 7\n}');
  assert.deepEqual(restored.draft.levels, ["calm", "upset"]);
});

test("JSON the form cannot show is left alone", () => {
  assert.equal(draftFromRequest(null), null);
  assert.equal(draftFromRequest({ state: "x", questions: {} }), null);
  assert.equal(
    draftFromRequest({
      state: "x",
      questions: { a: { type: "noul" }, b: { type: "noul" } },
    }),
    null,
  );
  assert.equal(
    draftFromRequest({ state: "x", questions: { a: { type: "yes_no" } } }),
    null,
  );
});

test("every type starts from its first preset and every preset is a valid request", () => {
  const drafts = initialDrafts();
  for (const type of DECISION_TYPES) {
    assert.ok(presetsFor(type).length > 0, type);
    assert.deepEqual(drafts[type], presetsFor(type)[0].draft);
  }
  assert.equal(new Set(PRESETS.map((p) => p.id)).size, PRESETS.length);
  for (const preset of PRESETS) {
    const question = Object.values(
      buildRequest(preset.type, preset.draft, "default").questions,
    )[0];
    assert.ok(preset.draft.state.trim(), preset.id);
    assert.ok(question.instructions, preset.id);
    if (preset.type === "noul") {
      assert.deepEqual(Object.keys(question.criteria as object).sort(), [
        "false",
        "true",
      ]);
    } else if (preset.type === "choice") {
      assert.ok(
        Object.keys(question.criteria as object).length >= 2,
        preset.id,
      );
    } else {
      const levels = question.criteria as string[];
      assert.ok(levels.length >= 2 && levels.length <= 10, preset.id);
    }
  }
});

test("a score lands on the nearest level of its legend", () => {
  const answer = {
    type: "score" as const,
    score: 2.6,
    confidence: 0.7,
    legend: { "10": "x", "0": "low", "1": "mid", "2": "high" },
    probabilities: {},
  };
  assert.deepEqual(scoreLevels(answer), {
    keys: ["0", "1", "2", "10"],
    nearest: "2",
    max: 10,
  });
  assert.equal(scoreLevels({ ...answer, score: -1 }).nearest, "0");
});

test("Try it in Settings > API opens the decision playground", () => {
  const section = readSrc(
    "features/settings/components/decision-api-section.tsx",
  );
  assert.match(section, /disabled=\{!enabled\}\s+onClick=\{\(\) => setTryOpen\(true\)\}/);
  assert.match(section, /<DecisionTryDialog\s+open=\{tryOpen && enabled\}/);
  assert.match(
    readSrc("features/settings/components/decision-try-dialog.tsx"),
    /runDecision\(/,
  );
});
