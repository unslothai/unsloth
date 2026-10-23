// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { hubAudioTaskSearch, matchesHubAudioTask } from "../src/lib/hub-audio-task.ts";
import { readSrc } from "./helpers/kit.ts";

const routeSource = readSrc("app/routes/hub.tsx");
const hubPageSource = readSrc("features/hub/hub-page.tsx");
const discoverSource = readSrc("features/hub/hooks/use-discover-search.ts");

test("the Hub link keeps only known audio tasks, and only with the Audio filter", () => {
  assert.deepEqual(hubAudioTaskSearch(["text-to-speech", "text-to-audio"], "audio"), [
    "text-to-speech",
    "text-to-audio",
  ]);
  assert.deepEqual(hubAudioTaskSearch("automatic-speech-recognition", "audio"), [
    "automatic-speech-recognition",
  ]);
  assert.deepEqual(hubAudioTaskSearch(["text-to-speech", "image-to-text", 3], "audio"), [
    "text-to-speech",
  ]);
  assert.equal(hubAudioTaskSearch(["image-to-text"], "audio"), undefined);
  assert.equal(hubAudioTaskSearch(["text-to-speech"], "vision"), undefined);
  assert.equal(hubAudioTaskSearch(undefined, "audio"), undefined);
  assert.match(routeSource, /hubAudioTaskSearch\(search\.audioTask, next\.capability\)/);
});

test("an audio page's Search Hub link narrows the Audio filter to its tasks", () => {
  assert.match(hubPageSource, /urlCapability === "audio" && urlTaskKey/);
  assert.match(hubPageSource, /capability: undefined,\s*audioTask: undefined,/);
  assert.match(hubPageSource, /if \(!capabilityFilters\.includes\("audio"\) && linkedAudioTask !== null\) \{\s*setLinkedAudioTask\(null\);/);
  assert.match(
    hubPageSource,
    // Only with Audio as the one filter: asking the Hub for audio tasks would drop the others' models.
    /capabilityFilters\.length === 1 && capabilityFilters\[0\] === "audio"\s*\? \(linkedAudioTask \?\? undefined\)\s*: undefined,/,
  );
  assert.match(discoverSource, /channel,\s*filters,\s*task,\s*\}\);/);
});

test("a typed Hub search still keeps only the linked audio tasks", () => {
  const asr = ["automatic-speech-recognition"] as const;
  assert.equal(matchesHubAudioTask("automatic-speech-recognition", asr), true);
  assert.equal(matchesHubAudioTask("text-to-speech", asr), false);
  assert.equal(matchesHubAudioTask("text-to-audio", asr), false);
  // An untagged repo stays, as the task listing keeps it.
  assert.equal(matchesHubAudioTask(undefined, asr), true);
  assert.equal(matchesHubAudioTask("text-to-speech", null), true);
  assert.equal(matchesHubAudioTask("text-to-speech", []), true);
  assert.match(
    hubPageSource,
    /key !== "audio" \|\|\s*matchesHubAudioTask\(row\.result\.pipelineTag, linkedAudioTask\)/,
  );
});

test("Separate's Search Hub link narrows to audio-to-audio, not the speak picker's tasks", () => {
  // The speak tasks (text-to-speech, text-to-audio) filtered out every separation model.
  const audioPage = readSrc("features/audio/audio-page.tsx");
  const selector = readSrc("features/model-picker/components/model-selector.tsx");
  assert.match(audioPage, /hubTask=\{pageWorkflow === "separate" \? "audio-to-audio" : undefined\}/);
  assert.match(selector, /const linkedTask = hubTask \?\? task;[\s\S]{0,200}?audioTask:\s*hubCapability === "audio" && linkedTask \? \[linkedTask\]\.flat\(\)/);
  assert.deepEqual(hubAudioTaskSearch("audio-to-audio", "audio"), ["audio-to-audio"]);
});
