// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  STEM_ORDER,
  SEPARATION_STEMS_BY_FAMILY,
  groupSeparationClips,
  orderStems,
  stemLabel,
} = await import("../src/features/audio/separation-stems.ts");
const {
  SEPARATE_MAX_SECONDS,
  estimateSeparateSeconds,
  overlapReloads,
  overlapRequest,
  roformerOverlapLogic,
  separateBlocker,
  separateCpuWarning,
  separatePresentation,
} = await import("../src/features/audio/separate-policy.ts");
const { buildAudioRunBody } = await import(
  "../src/features/audio/audio-run-request.ts"
);
const { audioRowMatchesWorkflow } = await import(
  "../src/features/audio/picker-filter.ts"
);
const {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_SEP_AUDIO_TYPE,
  audioCppModelFor,
  audioCppModelSpeaks,
  audioCppWorkflowsFor,
} = await import("../src/features/audio/audio-cpp-catalog.ts");
const { isTtsAudioType } = await import(
  "../src/features/audio/audio-page-policy.ts"
);
const { loadedAudioKind } = await import(
  "../src/features/audio/audio-workspace-utils.ts"
);
const { audioCapabilityLine } = await import(
  "../src/features/audio/catalog.ts"
);

const clip = (overrides: Record<string, unknown>) => ({
  id: "x",
  url: "/api/inference/audio/gallery/x/file",
  prompt: "song.wav",
  model: "audio-cpp/audio.cpp-gguf/HTDemucs-GGUF",
  audio_type: "audiocpp_sep",
  sample_rate: 44100,
  duration_s: 180,
  created_at: "2026-10-03T00:00:00Z",
  workflow: "separate",
  ...overrides,
});

test("stems show in a fixed order with readable names", () => {
  assert.deepEqual(orderStems(["other", "bass", "drums", "vocals"]), [
    "vocals",
    "drums",
    "bass",
    "other",
  ]);
  assert.deepEqual(orderStems(["instrumental", "zzz", "vocals", "aaa"]), [
    "vocals",
    "instrumental",
    "zzz",
    "aaa",
  ]);
  assert.equal(stemLabel("vocals"), "Vocals");
  assert.equal(stemLabel("lead_guitar"), "Lead guitar");
  assert.equal(stemLabel(""), "Stem");
  assert.equal(STEM_ORDER[0], "vocals");
});

test("each separation family lists the stems the runtime returns (spike S3)", () => {
  assert.deepEqual([...SEPARATION_STEMS_BY_FAMILY.htdemucs.stems].sort(), [
    "bass",
    "drums",
    "other",
    "vocals",
  ]);
  assert.equal(SEPARATION_STEMS_BY_FAMILY.htdemucs_6stems.stems.length, 6);
  assert.deepEqual(SEPARATION_STEMS_BY_FAMILY.bs_roformer.stems, [
    "vocals",
    "instrumental",
  ]);
  assert.deepEqual(SEPARATION_STEMS_BY_FAMILY.mel_band_roformer.stems, [
    "vocals",
    "instrumental",
  ]);
});

test("history shows one item per separation, stems in display order", () => {
  const groups = groupSeparationClips([
    clip({
      id: "a2",
      group_id: "g1",
      role: "drums",
      settings: { stems: ["drums", "vocals"] },
    }),
    clip({ id: "b1", group_id: "g2", role: "instrumental", pinned: true }),
    clip({ id: "a1", group_id: "g1", role: "vocals" }),
    clip({ id: "b2", group_id: "g2", role: "vocals" }),
    clip({ id: "solo", role: "vocals" }),
  ]);
  assert.deepEqual(
    groups.map((group) => [group.groupId, group.stems.map((stem) => stem.id)]),
    [
      ["g1", ["a1", "a2"]],
      ["g2", ["b2", "b1"]],
      ["clip:solo", ["solo"]],
    ],
  );
  assert.equal(groups[0].complete, true);
  assert.equal(groups[1].pinned, true);
  assert.equal(groups[0].title, "song.wav");
});

test("a separation missing stems says so", () => {
  const [group] = groupSeparationClips([
    clip({
      id: "a",
      group_id: "g",
      role: "vocals",
      settings: { stems: ["vocals", "drums", "bass", "other"] },
    }),
  ]);
  assert.equal(group.complete, false);
  assert.equal(group.expectedStems, 4);
});

test("a group cut by the page boundary waits for the next page", () => {
  const clips = [
    clip({
      id: "a",
      group_id: "g1",
      role: "vocals",
      settings: { stems: ["vocals"] },
    }),
    clip({
      id: "b",
      group_id: "g2",
      role: "vocals",
      settings: { stems: ["vocals", "drums"] },
    }),
  ];
  assert.deepEqual(
    groupSeparationClips(clips, true).map((group) => group.groupId),
    ["g1"],
  );
  assert.deepEqual(
    groupSeparationClips(clips, false).map((group) => group.groupId),
    ["g1", "g2"],
  );
});

test("Generate says what the track needs, in order", () => {
  const source = {
    kind: "input" as const,
    id: "i",
    name: "a.wav",
    durationS: 30,
  };
  const base = {
    source,
    sourceBusy: false,
    sourceExpired: false,
    sourceError: null,
  };
  assert.equal(separateBlocker(base), null);
  assert.equal(
    separateBlocker({ ...base, source: null })?.reason,
    "Add a track to separate.",
  );
  assert.equal(
    separateBlocker({ ...base, sourceBusy: true })?.kind,
    "source-busy",
  );
  assert.equal(
    separateBlocker({ ...base, sourceExpired: true })?.reason,
    "This track expired. Add it again.",
  );
  assert.equal(
    separateBlocker({ ...base, sourceError: "Not audio." })?.reason,
    "Not audio.",
  );
  const long = separateBlocker({
    ...base,
    source: { ...source, durationS: 601 },
  });
  assert.equal(long?.kind, "too-long");
  assert.match(long?.reason ?? "", /up to 10 minutes.*10:01/);
  assert.equal(
    separateBlocker({
      ...base,
      source: { ...source, durationS: SEPARATE_MAX_SECONDS },
    }),
    null,
  );
});

test("the estimate follows the model, the length and the overlap", () => {
  assert.equal(estimateSeparateSeconds("htdemucs", 180, true), 9);
  assert.equal(estimateSeparateSeconds("bs_roformer", 180, true), 13);
  assert.equal(estimateSeparateSeconds("bs_roformer", 180, false), 6);
  assert.equal(estimateSeparateSeconds("htdemucs", 1, true), 1);
  assert.equal(estimateSeparateSeconds("unknown", 180, true), null);
  assert.equal(estimateSeparateSeconds("htdemucs", null, true), null);
  assert.match(separateCpuWarning("bs_roformer", "cpu") ?? "", /very slow/);
  assert.equal(separateCpuWarning("bs_roformer", "auto"), null);
});

test("Overlap off asks for one pass; on sends nothing; a change is a reload", () => {
  assert.deepEqual(overlapRequest({ overlap: false }), {
    options: { num_overlap: 1 },
  });
  assert.deepEqual(overlapRequest({ overlap: true }), {});
  assert.deepEqual(overlapRequest(undefined), {});
  assert.equal(overlapReloads(undefined, true), false);
  assert.equal(overlapReloads(undefined, false), true);
  assert.equal(overlapReloads(false, false), false);
  assert.equal(overlapReloads(false, true), true);
  assert.deepEqual(roformerOverlapLogic.families, [
    "bs_roformer",
    "mel_band_roformer",
  ]);
  assert.deepEqual(roformerOverlapLogic.workflows, ["separate"]);
  assert.deepEqual(roformerOverlapLogic.initial([]), { overlap: true });
});

test("progress copy names the separation phases", () => {
  const presentation = {
    status: "Generating audio…",
    actionLabel: "Stop",
    canStop: true,
  };
  assert.equal(
    separatePresentation(presentation, "generating", false)?.status,
    "Separating…",
  );
  assert.match(
    separatePresentation(presentation, "generating", true)?.status ?? "",
    /Reloading/,
  );
  assert.equal(
    separatePresentation(presentation, "finishing", false)?.status,
    "Saving stems…",
  );
  assert.equal(separatePresentation(null, "generating", false), null);
});

test("a separate run sends the source and options only", () => {
  const body = buildAudioRunBody({
    workflow: "separate",
    inputs: { source: { input_id: "abc" } },
    options: { num_overlap: 1 },
    seed: undefined,
  });
  assert.deepEqual(body, {
    workflow: "separate",
    inputs: { source: { input_id: "abc" } },
    options: { num_overlap: 1 },
  });
  assert.equal("text" in body, false);
  assert.equal(buildAudioRunBody({ workflow: "speak", text: "hi" }).text, "hi");
});

test("separation rows list only on Separate", () => {
  const pages = ["speak", "clone", "music", "separate", "transcribe"] as const;
  const on = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
    pages.filter((page) => audioRowMatchesWorkflow(row, page));
  assert.deepEqual(on({ audioWorkflows: ["separate"] }), ["separate"]);
  assert.deepEqual(on({ audioType: AUDIO_CPP_SEP_AUDIO_TYPE }), ["separate"]);
  assert.deepEqual(on({ id: "audio-cpp/audio.cpp-gguf/HTDemucs-GGUF" }), [
    "separate",
  ]);
  assert.equal(
    on({ task: "text-to-speech", audioType: "audiocpp_tts" }).includes(
      "separate",
    ),
    false,
  );
  assert.equal(on({}).includes("separate"), false);
});

test("the catalog seeds the four separation models", () => {
  const seps = AUDIO_CPP_MODELS.filter((model) => model.task === "sep");
  assert.deepEqual(
    seps.map((model) => model.id.split("/").pop()),
    [
      "HTDemucs-GGUF",
      "BS-RoFormer-ep368-GGUF",
      "HTDemucs-6stems-GGUF",
      "Mel-Band-RoFormer-GGUF",
    ],
  );
  for (const model of seps) {
    assert.deepEqual(audioCppWorkflowsFor(model), ["separate"]);
    assert.ok(model.stems && model.stems.length >= 2);
    assert.equal(audioCppModelSpeaks(model.id), false);
  }
  assert.equal(
    audioCppModelFor("audio-cpp/audio.cpp-gguf/HTDemucs-6stems-GGUF")?.stems
      ?.length,
    6,
  );
});

test("a loaded separation model is a main-slot audio model", () => {
  assert.equal(isTtsAudioType("audiocpp_sep"), true);
  assert.equal(isTtsAudioType("audiocpp_sep", true), true);
  assert.equal(loadedAudioKind("audiocpp_sep"), "separation");
  assert.equal(
    audioCapabilityLine("separate", "audiocpp_sep"),
    "Source separation · GGUF",
  );
});

test("the host renders Separate's rail, footer and output and the page reuses the shared input", () => {
  const host = readSrc("features/audio/audio-page.tsx");
  // Leaving Audio stops a Separate recording and releases the mic, as on Clone.
  assert.match(
    host,
    /ttsWorkflow === "separate" \? \(\s*<AudioActiveProvider value=\{active\}>\s*<SeparateRail/,
  );
  assert.match(host, /ttsWorkflow === "separate" \? \(\s*<SeparateFooter/);
  assert.match(host, /ttsWorkflow === "separate" \? \(\s*<SeparateOutput/);
  assert.match(host, /ttsWorkflow === "separate"\s*\? separate\.blocker/);
  assert.match(host, /reason: "The loaded model separates audio\."/);
  // Trained speech checkpoints are not separation models.
  assert.match(
    host,
    /ttsWorkflow === "clone" \|\| ttsWorkflow === "separate"\s*\? \[\]/,
  );
  // A stem sent to Transcribe mid-run would be dropped when the run stops.
  assert.match(
    host,
    /target\.workflow === "transcribe"\) \{[\s\S]*?busyRef\.current === null[\s\S]*?fetchAudioBlob\(clip\.url\)[\s\S]*?if \(busyNow\(\)\) return;\s*if \(!transitionWorkflow\("transcribe"\)\)/,
  );
  const page = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(page, /<AudioSourceInput\s+id="separate-source"/);
  assert.match(page, /Converted to 44\.1 kHz automatically/);
  assert.match(page, /allowSavedVoice=\{false\}/);
  // The track card speaks of a track, not a clone reference, and records up to the cap.
  assert.match(page, /expiredMessage=\{SEPARATE_TRACK_EXPIRED_MESSAGE\}/);
  assert.match(page, /maxRecordSeconds=\{SEPARATE_MAX_SECONDS - 1\}/);
  assert.match(page, /recordHint="[^"]*track[^"]*"/);
  assert.match(page, /<StemMixer/);
  assert.match(page, /groupSeparationClips\(clips, hasMore\)/);
  assert.match(page, /aria-live="polite"/);
  // No inline fetch of the stems for playback: the mixer's own sources pin the group.
  assert.match(page, /useStemSources\(inputs, attempt\)/);
  // A failed stem fetch says so and retries instead of loading forever.
  assert.match(page, /sources\.failedIds\.length > 0/);
  assert.match(page, /setAttempt\(\(n\) => n \+ 1\)/);
  // A stem that cannot load is marked so the mixer plays the others.
  assert.match(page, /failed: sources\.failedIds\.includes\(clip\.id\)/);
});

test("a separation that a refresh missed falls back on the Separate page", () => {
  const generation = readSrc("features/audio/hooks/use-separate-generation.ts");
  assert.match(generation, /showRunResult\(\{[^}]*workflow: "separate"/);
  const run = readSrc("features/audio/hooks/use-clone-generation.ts");
  assert.match(run, /workflow: "speak" \| "clone" \| "separate";/);
});

test("a cut oldest run loads the next page itself, since a short list never scrolls", () => {
  const page = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(
    page,
    /hasMore && groupSeparationClips\(clips\)\.length > groups\.length/,
  );
  assert.match(
    page,
    /if \(tailHidden\) void loadMore\(\);\s*\}, \[tailHidden, clips, loadMore\]\)/,
  );
  // The same clips as the page-boundary case: one run hidden only while more pages exist.
  const clips = [
    clip({
      id: "a",
      group_id: "g1",
      role: "vocals",
      settings: { stems: ["vocals"] },
    }),
    clip({
      id: "b",
      group_id: "g2",
      role: "vocals",
      settings: { stems: ["vocals", "drums"] },
    }),
  ];
  assert.ok(
    groupSeparationClips(clips).length >
      groupSeparationClips(clips, true).length,
  );
});
