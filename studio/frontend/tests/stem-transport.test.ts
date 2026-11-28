// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { applyStemGain, volumeIsWritable } =
  await import("../src/features/audio/hooks/use-stem-transport.ts");

test("stems play plain elements, and only iOS builds a Web Audio graph for the gains", () => {
  const hook = readSrc("features/audio/hooks/use-stem-transport.ts");
  assert.match(
    hook,
    /const context = elementVolumeWorks\(\) \? null : ensureGraph\(\);/,
  );
  assert.match(hook, /await context\?\.resume\(\);/);
  assert.match(
    hook,
    /if \(elementVolumeWorks\(\)\)\s*applyStemGain\(element, gainsRef\.current\[index\] \?\? 1\);\s*element\.src = src;/,
  );
  assert.match(
    hook,
    /if \(context\) setGain\(voice, values\[index\] \?\? 1, context\);\s*else if \(elementVolumeWorks\(\)\)\s*applyStemGain\(voice\.element, values\[index\] \?\? 1\);/,
  );
});

test("a stem's gain becomes its element's volume, and silence mutes it", () => {
  const element = { volume: 1, muted: false };
  applyStemGain(element, 0.4);
  assert.deepEqual(element, { volume: 0.4, muted: false });
  applyStemGain(element, 0);
  assert.deepEqual(element, { volume: 0, muted: true });
  applyStemGain(element, 1);
  assert.deepEqual(element, { volume: 1, muted: false });
  applyStemGain(element, 3);
  assert.deepEqual(element, { volume: 1, muted: false });
  applyStemGain(element, Number.NaN);
  assert.deepEqual(element, { volume: 1, muted: false });
});

test("a media element that ignores volume writes (iOS Safari) is detected", () => {
  assert.equal(volumeIsWritable({ volume: 1 }), true);
  const pinned = {
    get volume() {
      return 1;
    },
    set volume(_value: number) {},
  };
  assert.equal(volumeIsWritable(pinned), false);
});
