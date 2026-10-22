// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Invariant: isCandidate() is never false when stripTrailingTemplatePlaceholder would cut.
// The reseed looks back a bounded window, so these cases cross that window repeatedly.

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRAILING_PLACEHOLDER_WINDOW as W,
  createTrailingPlaceholderWatch,
  stripTrailingTemplatePlaceholder,
} from "../src/features/chat/utils/trailing-template-placeholder.ts";

type Watch = ReturnType<typeof createTrailingPlaceholderWatch>;

const failures: string[] = [];
let checks = 0;
let strips = 0;

function check(where: string, watch: Watch, text: string) {
  checks += 1;
  const stripped = stripTrailingTemplatePlaceholder(text);
  const wouldCut = stripped.length !== text.length;
  if (wouldCut && !watch.isCandidate()) {
    failures.push(
      `${where}: the strip would cut ${text.length - stripped.length} characters but ` +
        `isCandidate() said no. tail=${JSON.stringify(text.slice(-60))}`,
    );
  }
  return { stripped, wouldCut };
}

function drive(where: string, arrivals: string[], seedText = ""): void {
  const watch = createTrailingPlaceholderWatch();
  let text = seedText;
  if (text) watch.append(text);
  arrivals.forEach((arrival, index) => {
    if (!arrival) return;
    text += arrival;
    watch.append(arrival);
    const { stripped, wouldCut } = check(`${where}@${index}`, watch, text);
    if (watch.isCandidate() && wouldCut) {
      strips += 1;
      text = stripped;
      watch.retract(text);
      // A second fragment can already be at the end, so check right after the retract too.
      check(`${where}@${index}:after-retract`, watch, text);
    }
  });
}

const filler = (n: number, ch = "x") => ch.repeat(n);

test("the gate never closes on an arrival the strip would cut", () => {
  for (const gap of [
    0,
    1,
    5,
    100,
    W - 10,
    W,
    W + 10,
    2 * W - 10,
    2 * W,
    2 * W + 10,
  ]) {
    drive(`two fragments gap=${gap}`, [
      `start ${filler(gap)}\${first}`,
      "${second}",
    ]);
  }

  for (const gap of [
    0,
    10,
    W - 4,
    W,
    W + 4,
    2 * W - 4,
    2 * W,
    2 * W + 4,
    3 * W,
  ]) {
    drive(`fragment then whitespace gap=${gap}`, [
      `head ${filler(gap)}\${a}\${b}`,
      "   ",
      "\n",
      " ",
    ]);
  }

  for (let back = 2 * W - 6; back <= 2 * W + 6; back += 1) {
    drive(`reseed edge back=${back}`, [
      `\${outer${filler(Math.max(0, back))}`,
      "}",
      "${inner}",
      "  ",
    ]);
  }

  for (const gap of [0, W, 2 * W, 3 * W]) {
    drive("resumed", ["${b}", "  "], `resumed ${filler(gap)}\${a}`);
  }

  assert.deepEqual(
    failures,
    [],
    `${failures.length} violations over ${checks} states`,
  );
  assert.ok(
    strips > 0,
    "no strip ever fired, so the invariant was never actually exercised",
  );
});

test("randomised streams far longer than the reseed window", () => {
  const makeRandom = (seed: number) => {
    let value = seed >>> 0;
    return () => {
      value = (value * 1664525 + 1013904223) >>> 0;
      return value / 0x100000000;
    };
  };
  const alphabet = [
    "a",
    " ",
    "\n",
    "$",
    "{",
    "}",
    "${",
    "}$",
    "${}",
    "x".repeat(50),
    "y".repeat(700),
  ];

  const before = failures.length;
  for (let seed = 1; seed <= 120; seed += 1) {
    const random = makeRandom(seed);
    const arrivals: string[] = [];
    for (let i = 0; i < 200; i += 1) {
      let piece = "";
      const n = 1 + Math.floor(random() * 3);
      for (let k = 0; k < n; k += 1) {
        piece += alphabet[Math.floor(random() * alphabet.length)];
      }
      arrivals.push(piece);
    }
    drive(`random seed=${seed}`, arrivals);
  }
  assert.deepEqual(failures.slice(before), []);
});
