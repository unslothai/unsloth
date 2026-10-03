// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const output = readSrc("features/audio/pages/transcribe-output.tsx");
const view = readSrc("features/audio/components/transcript-view.tsx");
const player = readSrc("features/audio/components/transcript-player.tsx");
const chip = readSrc("features/audio/components/speaker-chip.tsx");
const gallery = readSrc("features/audio/transcript-gallery.tsx");

test("timestamps are mono buttons that seek and play, or plain text without audio", () => {
  const button = view.match(
    /<button[\s\S]*?aria-label=\{`Play from \$\{stamp\}`\}[\s\S]*?<\/button>/,
  )?.[0];
  assert.ok(button, "each timestamp is a labelled button");
  assert.match(
    button,
    /onClick=\{\(\) => player\.current\?\.seek\(segment\.start, true\)\}/,
  );
  assert.match(button, /font-mono/);
  assert.match(button, /tabular-nums/);
  assert.match(button, /stampWidth/);
  assert.match(
    view,
    /"min-w-\[5ch\]"/,
    "a fixed width keeps rows aligned at every UI size",
  );
  assert.match(view, /\{player \? \(/);
  assert.match(output, /player=\{audioAvailable \? player : null\}/);
});

test("the playing row is marked by aria-current and a border, not colour alone", () => {
  assert.match(view, /aria-current=\{current \? "true" : undefined\}/);
  assert.match(
    view,
    /current \? "border-foreground bg-muted" : "border-transparent"/,
  );
  assert.match(view, /border-l-2/);
});

test("following along respects reduced motion and only runs while playing", () => {
  assert.match(
    view,
    /import \{ prefersReducedMotion \} from "@\/features\/settings";/,
  );
  assert.match(
    view,
    /if \(!playing \|\| active < 0 \|\| shown !== "segments"\) return;/,
  );
  assert.match(view, /behavior: prefersReducedMotion\(\) \? "auto" : "smooth"/);
  assert.match(view, /block: "nearest"/);
  assert.match(view, /motion-reduce:transition-none/);
});

test("the Segments tab explains itself when a transcript has no timing", () => {
  assert.match(view, /<PillTabs[\s\S]*?disabled=\{!hasSegments\}/);
  assert.match(
    view,
    /No timestamps in this transcript\. Turn on Timestamps with a model\s+that supports them and transcribe again\./,
  );
});

test("a speaker colour always comes with the speaker's name", () => {
  assert.match(chip, /backgroundColor: `var\(--chart-\$\{color\}\)`/);
  assert.match(chip, /% SPEAKER_COLORS\) \+ 1/);
  assert.match(chip, /<span className="min-w-0 truncate">\{label\}<\/span>/);
  assert.match(chip, /aria-label=\{`Rename \$\{label\}`\}/);
  assert.match(chip, /aria-hidden="true"[\s\S]*?--chart-/);
  assert.match(chip, /Renames every line by this speaker\./);
  assert.match(chip, /event\.key === "Enter"/);
  assert.match(chip, /onRename\(id, sanitizeSpeakerName\(draft\)\)/);
});

test("SRT and VTT say why they are off when there are no timestamps", () => {
  assert.match(
    output,
    /const blocked = formatNeedsTimestamps\(format\) && !timed;/,
  );
  assert.match(output, /disabled=\{blocked\}[\s\S]*?Needs timestamps/);
  assert.match(gallery, /disabled=\{blocked\}[\s\S]*?Needs timestamps/);
});

test("a download marks only the transcript it saved", () => {
  assert.match(
    output,
    /const version = transcriptVersion\.current;[\s\S]*?if \(await downloadTranscriptFile\([\s\S]*?\)\)\s*markExported\(version\);/,
  );
  assert.match(output, /Not saved/);
});

test("the result is focusable for the host, and errors are announced", () => {
  assert.match(
    output,
    /id="transcribe-result"\s+tabIndex=\{-1\}\s+aria-label="Transcript"/,
  );
  assert.match(output, /role="alert"/);
  assert.match(output, /onSelect=\{selectRecord\}/);
});

test("the player fetches by source, cleans up, notes missing audio and never autoplays", () => {
  assert.match(player, /fetchAudioBlob\(\s*sourceFileUrl\(/);
  assert.match(player, /decodePeaks\(blob\)/);
  assert.match(
    player,
    /controller\.abort\(\);\s*if \(url\) URL\.revokeObjectURL\(url\);/,
  );
  assert.match(
    player,
    /The audio for this transcript is no longer available\. Timestamps still\s+export\./,
  );
  assert.match(player, /controlRef=\{controlRef\}/);
  assert.doesNotMatch(player + output + view, /autoPlay|\.play\(/);
});

test("history rows show duration and what the transcript carries", () => {
  assert.match(
    gallery,
    /font-mono[^"]*tabular-nums[\s\S]*?formatTimestamp\(record\.duration\)/,
  );
  assert.match(gallery, /"Timestamps"/);
  assert.match(gallery, /`\$\{speakers\} speakers`/);
  assert.match(gallery, /full = await getTranscript\(record\.id\);/);
});
