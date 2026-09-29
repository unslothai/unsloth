// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The watch exists so the adapter can skip the strip on almost every arrival. That is only
// sound if the gate never closes on an arrival the strip WOULD have cut:
//
//   isCandidate() is never false when stripTrailingTemplatePlaceholder would cut.
//
// If it is ever false at such a moment the reply keeps a `${...}` fragment that the previous
// build removed, which is a visible difference in the user's text rather than a slower path.
//
// The cases beside this one work on short buffers, where the reseed after a strip never comes
// under pressure: it deliberately looks back a bounded window and forgets everything older, so
// only a buffer longer than that window can catch it forgetting something it still needs. These
// cases cross that window repeatedly.

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

/** Drive the watch exactly as the adapter does, checking at every arrival. */
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
      // A second fragment can be sitting at the end already, so the invariant has to
      // hold again immediately after the retract, not only after the next arrival.
      check(`${where}@${index}:after-retract`, watch, text);
    }
  });
}

const filler = (n: number, ch = "x") => ch.repeat(n);

test("the gate never closes on an arrival the strip would cut", () => {
  // Two fragments back to back: the first strip is followed by a second the reseed has to
  // rediscover from the buffer rather than from what it was tracking.
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

  // A fragment, a strip, then only whitespace, so the next cut is of the fragment the
  // reseed had to remember across the strip.
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

  // The opener landing on each offset either side of the reseed edge.
  for (let back = 2 * W - 6; back <= 2 * W + 6; back += 1) {
    drive(`reseed edge back=${back}`, [
      `\${outer${filler(Math.max(0, back))}`,
      "}",
      "${inner}",
      "  ",
    ]);
  }

  // A resumed turn: the buffer starts non-empty and the watch is caught up with one
  // append of the whole partial, which is what the adapter does.
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
  // Seeded, so a failure is reproducible from the name in the message.
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
