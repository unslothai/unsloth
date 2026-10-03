// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const host = readSrc("features/audio/audio-page.tsx");
const slot = readSrc("features/audio/hooks/use-audio-model-slot.ts");
const handoff = readSrc("features/audio/hooks/use-audio-handoff.ts");
const generation = readSrc("features/audio/hooks/use-speech-generation.ts");
const gallery = readSrc("features/audio/hooks/use-audio-gallery.tsx");
const tour = readSrc("features/audio/tour/steps.tsx");

test("the host calls its hooks in the order their effects ran in the single page", () => {
  const order = [
    "useSttSidecar(",
    "useTranscription(",
    "useAudioGallery(",
    "useAudioModelSlot(",
    "useSpeechGeneration(",
    "useAudioHandoff(",
  ].map((call) => host.indexOf(call));
  assert.ok(order.every((at) => at > 0), String(order));
  assert.deepEqual([...order].sort((a, b) => a - b), order);
  assert.ok(host.indexOf("// Resync on activation") < host.indexOf("useAudioModelSlot("));
});

test("only the page commits a workflow; the sidebar's request goes through the same gate", () => {
  assert.match(
    host,
    /useAudioWorkspaceStore\.getState\(\)\.clearRequestedWorkflow\(\);\s*transitionWorkflow\(requestedWorkflow\);/,
  );
  assert.match(
    slot,
    /const transitionWorkflow = useCallback\(\s*\(next: AudioWorkflowId\) => \{\s*if \(!transitionMode\(slotForWorkflow\(next\)\)\) return false;\s*useAudioWorkspaceStore\.getState\(\)\.commitWorkflow\(next\);/,
  );
  assert.match(host, /mode === "transcribe" \? "transcribe" : musicGeneration \? "music" : "speak"/);
});

test("a loaded model opens the page that fits it", () => {
  assert.match(
    slot,
    /const loadedWorkflow = workflowForLoadedModel\(\{\s*current: useAudioWorkspaceStore\.getState\(\)\.workflow,\s*audioWorkflows: res\.audio_workflows,\s*music: isMusicGenerationModel\(repoId, res\.audio_type\),\s*\}\);[\s\S]{0,200}?rememberModel\(loadedWorkflow, repoId\);\s*if \(modeRef\.current === "speak"\) workspace\.commitWorkflow\(loadedWorkflow\);/,
  );
  assert.match(host, /adoptWorkflow\("music"\)/);
});

test("?workflow= names the page ahead of ?task=, and both clear the URL", () => {
  const workflowAt = handoff.indexOf("const routedWorkflow = routeSearch.workflow;");
  const taskAt = handoff.indexOf("const task = routeSearch.task;");
  assert.ok(workflowAt > 0 && workflowAt < taskAt);
  assert.match(
    handoff,
    /if \(!transitionWorkflow\(routedWorkflow\)\) return;\s*void navigateSelf\(\{ to: "\/audio", search: \{\}, replace: true \}\);/,
  );
  assert.match(
    handoff,
    /void navigateSelf\(\{ to: "\/audio", search: \{\}, replace: true \}\);\s*\/\/[^\n]*\n\s*useAudioWorkspaceStore\s*\.getState\(\)\s*\.commitWorkflow\(audioWorkflowForTask\(task\) \?\? intended\);/,
  );
  assert.match(handoff, /transitionWorkflow\(clipWorkflow\(clip\)\)/);
});

test("Speak and Music keep separate drafts that survive a reload", () => {
  assert.match(generation, /readLastPrompt\(ttsDraftKey\("prompt", "speak"\)\)/);
  assert.match(generation, /readLastPrompt\(ttsDraftKey\("prompt", "music"\)\)/);
  assert.match(generation, /const base = workflow === "music" \? "audio:music" : "audio";/);
  assert.match(generation, /saveLastPrompt\(ttsDraftKey\(field, page\), next\);/);
});

test("history is per page, and Clear all clears only that page", () => {
  assert.match(gallery, /clips\.filter\(\(clip\) => clipWorkflow\(clip\) === workflow\)/);
  assert.match(gallery, /await clearAudioGallery\(workflow\);/);
  assert.match(host, /\(\) => handleClearGallery\(ttsWorkflow\)/);
});

test("Create|Train shows only where the page trains", () => {
  assert.match(host, /\{workflowTab\.createTrain \? \(\s*<PillTabs\s*ariaLabel="Page mode"/);
});

test("Generate says why it is off and runs from Mod+Enter anywhere on the page", () => {
  assert.match(host, /: "Type the text to speak\."/);
  assert.match(host, /const chooseModelAction = \{ label: "Choose a model", onClick: openSelector \};/);
  assert.match(host, /\{ label: "Choose a music model", onClick: openSelector \}/);
  assert.match(host, /if \(busy === "generating"\) setGenerationError\(null\);/);
  assert.match(host, /!pageRootRef\.current\?\.contains\(event\.target\)/);
  // Clone runs through its own hook; the shortcut follows the page.
  assert.match(
    host,
    /const handlePageGenerate =\s*ttsWorkflow === "clone" \? clone\.handleGenerate : handleGenerate;/,
  );
  assert.match(host, /if \(canGenerate\) void handlePageGenerate\(\);/);
});

test("each page tours its own model and settings", () => {
  assert.match(tour, /workflow: AudioWorkflowId;/);
  assert.match(tour, /if \(workflow === "transcribe"\)/);
  assert.match(tour, /modelStep\(workflow\)/);
  assert.match(host, /buildAudioTourSteps\(\{ workflow: pageWorkflow \}\)/);
});
