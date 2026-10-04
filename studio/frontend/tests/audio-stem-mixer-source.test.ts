// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const mixer = readSrc("features/audio/components/stem-mixer.tsx");
const sources = readSrc("features/audio/hooks/use-stem-sources.ts");
const transport = readSrc("features/audio/hooks/use-stem-transport.ts");
const waveform = readSrc("features/audio/components/waveform.tsx");

test("Solo and Mute are named toggle buttons", () => {
  assert.match(
    mixer,
    /aria-label=\{`Solo \$\{stem\.label\}`\}\s+aria-pressed=\{level\.solo\}/,
  );
  assert.match(
    mixer,
    /aria-label=\{`Mute \$\{stem\.label\}`\}\s+aria-pressed=\{level\.muted\}/,
  );
  assert.match(
    mixer,
    /thumbValueText=\{\(value\) => `\$\{stem\.label\} volume \$\{value\} %`\}/,
  );
  assert.match(mixer, /aria-label=\{`Download \$\{stem\.label\}`\}/);
});

test("the mixer is a keyboard target: Space or K plays, arrows seek, only on itself or the bars", () => {
  assert.match(mixer, /aria-label="Stem mixer"/);
  assert.match(mixer, /tabIndex=\{0\}/);
  assert.match(
    mixer,
    /event\.key === " " \|\| event\.key === "k" \|\| event\.key === "K"/,
  );
  assert.match(mixer, /event\.key === "ArrowRight"/);
  assert.match(mixer, /event\.key === "ArrowLeft"/);
  assert.match(mixer, /target === event\.currentTarget/);
  assert.match(mixer, /closest\("\[data-stem-bars\]"\)/);
  assert.match(mixer, /aria-live="polite"/);
  assert.match(mixer, /playRef\.current\?\.focus\(\)/);
});

test("waveforms are neutral, sizes scale, rows stack when narrow", () => {
  assert.match(mixer, /<WaveformBars/);
  assert.match(waveform, /export function WaveformBars/);
  assert.match(
    waveform,
    /<WaveformBars/,
    "Waveform renders through WaveformBars",
  );
  assert.match(waveform, /"text-foreground"/);
  assert.match(waveform, /"text-muted-foreground"/);
  assert.match(mixer, /@container/);
  assert.match(mixer, /@\[30rem\]:grid-cols-/);
  assert.match(mixer, /calc\(\d+px\*var\(--ui-space-scale,1\)\)/);
  assert.match(mixer, /rounded-4xl/);
  assert.match(mixer, /corner-squircle/);
  assert.doesNotMatch(mixer, /shadow-/);
});

test("every animation has a reduced-motion pair", () => {
  for (const line of mixer.split("\n")) {
    if (/\banimate-(?!none)/.test(line))
      assert.match(line, /motion-reduce:animate-none/, line.trim());
    if (/\btransition-/.test(line))
      assert.match(line, /motion-reduce:transition-none/, line.trim());
  }
});

test("stem audio uses its own pinned cache, not the gallery's LRU", () => {
  assert.match(sources, /new BlobUrlCache\(/);
  assert.doesNotMatch(sources, /galleryCache/);
  assert.match(sources, /fetchAudioBlob\(/);
  assert.match(sources, /cache\.clear\(\)/);
  assert.match(sources, /URL\.createObjectURL\(blob\)/);
  // The CSP's connect-src refuses fetch() on blob: URLs, so peaks decode from the Blob itself.
  assert.match(sources, /blob\.arrayBuffer\(\)/);
  assert.doesNotMatch(sources, /fetch\(objectUrl\)/);
  assert.match(sources, /PEAKS_CAP = 64/);
  assert.match(sources, /decodeAudioData/);
  assert.match(sources, /computePeaks\(/);
});

test("the transport is one context with a gain per stem and a drift check", () => {
  assert.match(transport, /new AudioContext\(\)/);
  assert.equal(transport.match(/new AudioContext\(/g)?.length, 1);
  assert.match(transport, /createMediaElementSource/);
  assert.match(transport, /createGain\(\)/);
  assert.match(transport, /DRIFT_TOLERANCE_S = 0\.04/);
  assert.match(transport, /DRIFT_CHECK_MS = 250/);
  assert.match(transport, /"canplay"/);
  assert.match(transport, /if \(!active\) pause\(\)/);
  assert.match(transport, /context\?\.close\(\)/);
});

test("a stem that failed to load is left out of playback, so the rest still play", () => {
  assert.match(
    mixer,
    /const playable = useMemo\(\s*\(\) => stems\.filter\(\(stem\) => !stem\.failed\)/,
  );
  assert.match(
    mixer,
    /playable\.map\(\(stem\) => \(\{ id: stem\.clipId, src: stem\.src \}\)\)/,
  );
  assert.match(
    mixer,
    /const loadingCount = playable\.filter\(\(stem\) => !stem\.src\)\.length/,
  );
  assert.match(mixer, /aria-busy=\{!\(stem\.src \|\| stem\.failed\)\}/);
  // The zip needs every stem, so Download all waits for all of them.
  assert.match(mixer, /disabled=\{stems\.some\(\(stem\) => !stem\.src\)\}/);
});
