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
  // A refused request is kept and retried when busy settles; the commit is what clears it.
  assert.match(
    host,
    /if \(!active \|\| requestedWorkflow === null\) return;\s*transitionWorkflow\(requestedWorkflow\);\s*\}, \[active, busy, requestedWorkflow, transitionWorkflow\]\);/,
  );
  assert.doesNotMatch(host, /clearRequestedWorkflow\(\)/);
  assert.match(
    slot,
    /\} else if \(!transitionMode\(slotForWorkflow\(next\)\)\) \{\s*return false;\s*\}\s*store\.commitWorkflow\(next\);/,
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

test("?workflow= names the page ahead of ?task=, and both go through the workflow gate", () => {
  // Task-only links take the same gate as ?workflow=, so a Speak/Music switch never skips the busy check.
  assert.match(
    handoff,
    /const routedWorkflow = audioRouteIntent\(routeSearch\);\s*if \(routedWorkflow === null\) return;[\s\S]{0,120}?if \(!transitionWorkflow\(routedWorkflow\)\) return;\s*void navigateSelf\(\{ to: "\/audio", search: \{\}, replace: true \}\);/,
  );
  assert.doesNotMatch(handoff, /commitWorkflow\(/);
  // A Library clip asks for its page through ?workflow=, so a busy page retries it later.
  assert.match(
    handoff,
    /search: \(prev\) => \(\{ \.\.\.prev, workflow: clipWorkflow\(clip\) \}\)/,
  );
});

test("Speak and Music keep separate drafts that survive a reload", () => {
  assert.match(generation, /readLastPrompt\(ttsDraftKey\("prompt", "speak"\)\)/);
  assert.match(generation, /readLastPrompt\(ttsDraftKey\("prompt", "music"\)\)/);
  assert.match(generation, /const base = workflow === "music" \? "audio:music" : "audio";/);
  assert.match(generation, /saveLastPrompt\(ttsDraftKey\(field, page\), next\);/);
});

test("history is per page, and Clear all clears only that page", () => {
  assert.match(
    gallery,
    /clips\.filter\(\(clip\) => clipWorkflow\(clip\) === workflow/,
  );
  assert.match(gallery, /await clearAudioGallery\(workflow\);/);
  assert.match(host, /\(\) => handleClearGallery\(ttsWorkflow\)/);
});

test("Create|Train shows only where the page trains", () => {
  assert.match(
    host,
    /\{workflowTab\.createTrain \? \(\s*<PillTabs\s*ariaLabel="Page mode"/,
  );
});

test("Generate says why it is off and runs from Mod+Enter anywhere on the page", () => {
  assert.match(host, /: "Type the text to speak\."/);
  assert.match(host, /const chooseModelAction = \{ label: "Choose a model", onClick: openSelector \};/);
  assert.match(host, /\{ label: "Choose a music model", onClick: openSelector \}/);
  assert.match(host, /if \(busy === "generating"\) setGenerationError\(null\);/);
  assert.match(host, /!pageRootRef\.current\?\.contains\(event\.target\)/);
  assert.match(
    host,
    /const handlePageGenerate =\s*ttsWorkflow === "separate" \? separate\.handleGenerate :\s*ttsWorkflow === "clone" \? clone\.handleGenerate :\s*ttsWorkflow === "edit" \? edit\.handleGenerate :\s*ttsWorkflow === "convert" \? convert\.handleGenerate : handleGenerate;/,
  );
  assert.match(host, /if \(canGenerate\) void handlePageGenerate\(\);/);
});

test("each page tours its own model and settings", () => {
  assert.match(tour, /workflow: AudioWorkflowId;/);
  assert.match(tour, /if \(workflow === "transcribe"\)/);
  assert.match(tour, /modelStep\(workflow\)/);
  assert.match(host, /buildAudioTourSteps\(\{ workflow: pageWorkflow \}\)/);
});

test("the rail heading switches pages from a menu of every workflow", () => {
  // A tab row of 5-7 pages truncated every label in the default rail; the title menu lists them all.
  const host = readSrc("features/audio/audio-page.tsx");
  assert.doesNotMatch(host, /ariaLabel="Audio workflow"/);
  assert.match(
    host,
    /<WorkflowTitleMenu\s+workflow=\{pageWorkflow\}\s+onSelect=\{transitionWorkflow\}/,
  );
  const menu = readSrc("features/audio/components/workflow-title-menu.tsx");
  assert.match(menu, /data-tour="audio-mode"/);
  assert.match(menu, /AUDIO_WORKFLOWS\.map\(\(tab\) =>/);
  assert.match(menu, /\{tab\.hint\}/);
  // Radio items: the current page is ticked and announced as checked.
  assert.match(menu, /<DropdownMenuRadioGroup\s+value=\{current\.id\}/);
});

test("results show as a clip card with a waveform, never autoplaying", () => {
  const output = readSrc("features/audio/pages/tts-workspace.tsx");
  const card = readSrc("features/audio/components/clip-card.tsx");
  assert.doesNotMatch(output, /<audio\b/);
  assert.doesNotMatch(output + card, /autoPlay/);
  assert.match(output, /<ClipCard\s+\/\/[^\n]*\n\s*key=\{selectedClip\.id\}/);
  assert.match(card, /<Waveform\s+peaks=\{peaks\}/);
  // A run in progress stands where its clip will appear, with Stop, and no second live region.
  assert.match(
    output,
    /\{pending \? \(\s*<PendingClipCard \{\.\.\.pending\} \/>/,
  );
  assert.match(
    host,
    /pending:\s*busy === "generating" && generationPresentation/,
  );
  assert.match(host, /onStop: handleStopGeneration,/);
  assert.doesNotMatch(card, /aria-live/);
});

test("Send to lists the other Audio pages from the shared workflow list", () => {
  const card = readSrc("features/audio/components/clip-card.tsx");
  assert.match(
    card,
    /AUDIO_WORKFLOWS\.filter\(\s*\(tab\) => tab\.id !== current && handlers\[tab\.id\],?\s*\)/,
  );
  assert.match(
    host,
    /if \(!transitionWorkflow\("transcribe"\)\) return;[\s\S]{0,200}?useAudioTranscribeStore\.setState\(\{\s*source: \{\s*kind: "clip",\s*id: clip\.id,/,
  );
});

test("an unlisted clip stays on the page that made it", () => {
  assert.equal((generation.match(/saved: (?:true|false),\s*workflow,/g) ?? []).length, 2);
  assert.match(gallery, /const pageFallbackClip =\s*fallbackClip\?\.workflow === workflow \? fallbackClip : null;/);
  assert.match(gallery, /if \(!enabled \|\| selectedClip \|\| pageFallbackClip\) return;/);
  assert.match(host, /fallbackClip: pageFallbackClip,\s*\} = useWorkflowHistory\(/);
  assert.match(host, /handleDeleteClip,\s*fallbackClip: pageFallbackClip,/);
});

test("switching between Speak and Music passes the same busy gate as a mode switch", () => {
  assert.match(
    slot,
    /next !== store\.workflow &&\s*mode === "speak" &&\s*slotForWorkflow\(next\) === "speak"\s*\) \{\s*if \(\s*!canTransitionAudioMode\(busyRef\.current, generationPhaseRef\.current\)\s*\) \{[\s\S]{0,200}?return false;\s*\}\s*if \(busyRef\.current === "generating"\) handleStopGeneration\(\);/,
  );
});

test("the Max tokens stop warning from main lives in the generation hook", () => {
  assert.match(
    generation,
    /if \(generated\.choices\[0\]\?\.finish_reason === "length"\)\s*toast\.warning\(\s*maxTokens < TTS_MAX_TOKENS/,
  );
});

test("auto-selecting a page's history keeps another page's unsaved clip", () => {
  assert.match(gallery, /if \(first\) selectClip\(first\.id, fallbackClip !== null\);/);
});

test("Send to waits for a running task instead of stopping it and dropping the clip", () => {
  // A switch mid-run stops the run, but the page stays busy until the stopped run settles, so the
  // transcription was refused and the clip silently dropped.
  assert.match(
    host,
    /transcribe: \(\) => \{[\s\S]*?if \(busyRef\.current !== null\) \{\s*toast\.info\([^)]*\);\s*return;\s*\}\s*if \(!transitionWorkflow\("transcribe"\)\) return;/,
  );
});

test("switching between Speak and Music disowns a pending pick", () => {
  assert.match(
    slot,
    /if \(busyRef\.current === "generating"\) handleStopGeneration\(\);\s*invalidatePendingTtsSelection\(\);\s*\} else if \(!transitionMode/,
  );
});

test("trained checkpoints are split between Speak and Music like the catalog", () => {
  assert.match(
    host,
    /trainedTtsModels\.filter\(\s*\(model\) =>\s*isMusicGenerationModel\(model\.id, model\.audioType\) ===\s*\(ttsWorkflow === "music"\),\s*\)/,
  );
});
