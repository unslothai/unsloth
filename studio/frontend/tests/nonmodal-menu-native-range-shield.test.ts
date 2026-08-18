// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A native <input type="range"> COMMITS ON POINTERDOWN. Pressing the track moves the thumb and
// fires `input` there and then, which is why the platform advice for range inputs has always
// been "use oninput, it fires during the press". So it belongs to the same family as Radix
// Slider and Radix Select: controls that have already acted by the time `swallowClick` in
// lib/menu-dismiss.ts runs, and which therefore have to leave the hit test while a non-modal
// menu is open rather than rely on the click swallow.
//
// nonmodal-menu-pointerdown-shield.test.ts pins the two RADIX primitives by name. It cannot see
// a native range, and the one in the AudioPlayer was missed: measured on chromium with the
// composer "+" menu open and a generated-audio message in the thread, one press on the visible
// scrubber closed the menu AND seeked the audio, currentTime 0 -> 4.08 s read BEFORE the
// release. On the merge base the same press landed on HTML with body pointer-events: none.
//
// So this sweeps for the SHAPE rather than for a list: every `type="range"` under src/ has to be
// shielded, and a new one added tomorrow is covered without anyone remembering this file.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

function sources(dir: string, found: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      sources(full, found);
    } else if (entry.endsWith(".tsx")) {
      found.push(full);
    }
  }
  return found;
}

const RANGE = /type=["{]?["']?range["']?\}?/;

/**
 * A range that only ever renders inside the file's own `modal={true}` overlay is out of scope
 * and must NOT be shielded. That overlay parks `pointer-events: none` on the body itself, so no
 * press outside it reaches anything; and shielding it anyway would take the control away from
 * the user for as long as any non-modal menu anywhere happened to be open. The settings
 * appearance colour picker is the live example: its hue slider lives in a `modal={true}`
 * Popover.
 *
 * The limit of the rule, said out loud: if such a file ever grows a SECOND range outside its
 * modal overlay, this sweep will not see it.
 */
const BEHIND_A_MODAL_LAYER = /modal=\{true\}/;

function filesWithANativeRange(): string[] {
  const found: string[] = [];
  for (const file of sources(SRC)) {
    const source = readFileSync(file, "utf8");
    if (BEHIND_A_MODAL_LAYER.test(source)) continue;
    // `<input` and a `type="range"` on the same element. Splitting on `<input` keeps a `range`
    // elsewhere in the file from claiming one exists.
    for (const chunk of source.split("<input").slice(1)) {
      const element = chunk.slice(0, chunk.indexOf(">"));
      if (RANGE.test(element)) {
        found.push(file);
        break;
      }
    }
  }
  return found;
}

test("every native range control takes itself out of the dismissing press", () => {
  const files = filesWithANativeRange();
  // A sweep that resolves nothing would pass without measuring anything. The audio scrubber is
  // the one this was written for; if it ever stops being an <input type="range"> this number is
  // the thing that says so rather than the suite quietly testing an empty set.
  assert.ok(
    files.length >= 1,
    "found no native range control at all; the sweep is not reaching the tree",
  );
  const offenders: string[] = [];
  for (const file of files) {
    const source = readFileSync(file, "utf8");
    const relative = path.relative(SRC, file);
    if (!/useShieldedFromDismissingPress\(\)/.test(source)) {
      offenders.push(`${relative}: never asks whether a non-modal menu is open`);
      continue;
    }
    if (!/pointerEvents: "none"/.test(source)) {
      offenders.push(`${relative}: subscribes to the shield but never applies it`);
    }
    if (!/"pointerdown-commits"/.test(source)) {
      offenders.push(
        `${relative}: missing the class the static popper exception in index.css keys on, so a ` +
          "range inside an open menu would be shielded from the user as well",
      );
    }
  }
  assert.deepEqual(
    offenders,
    [],
    "a native <input type=\"range\"> commits during the press, so the click swallow cannot undo " +
      "it: the AudioPlayer scrubber both dismissed the menu and seeked the audio. Offenders: " +
      offenders.join("; "),
  );
});
