// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { groupSeparationClips, orderStems, separationGroupSummary, stemLabel } = await import(
  "../src/features/audio/separation-stems.ts"
);
const {
  SEPARATE_MAX_SECONDS,
  estimateSeparateSeconds,
  overlapReloads,
  overlapRequest,
  roformerOverlapLogic,
  separateBlocker,
  separateCpuWarning,
  separateEmptyTitle,
  separatePresentation,
  separateAnnouncement,
  separateRunOptions,
  separateVerb,
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
const { pageModels } = await import("../src/features/audio/page-models.ts");
const { isTtsAudioType } = await import(
  "../src/features/audio/audio-page-policy.ts"
);
const { loadedAudioKind } = await import(
  "../src/features/audio/audio-workspace-utils.ts"
);
const { audioCapabilityLine } = await import(
  "../src/features/audio/catalog.ts"
);

const clip = (
  id: string,
  role: string,
  extra: Record<string, unknown> = {},
) => ({
  id,
  role,
  url: `/api/inference/audio/gallery/${id}/file`,
  prompt: "song.wav",
  model: "audio-cpp/audio.cpp-gguf/HTDemucs-GGUF",
  audio_type: "audiocpp_sep",
  sample_rate: 44100,
  duration_s: 180,
  created_at: "2026-10-03T00:00:00Z",
  workflow: "separate",
  ...extra,
});
const ids = (groups: { groupId: string }[]) => groups.map((g) => g.groupId);

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
  assert.equal(stemLabel("lead_guitar"), "Lead guitar");
  assert.equal(stemLabel(""), "Stem");
});

test("history shows one item per separation, stems in display order", () => {
  const groups = groupSeparationClips([
    clip("a2", "drums", {
      group_id: "g1",
      settings: { stems: ["drums", "vocals"] },
    }),
    clip("b1", "instrumental", { group_id: "g2", pinned: true }),
    clip("a1", "vocals", { group_id: "g1" }),
    clip("b2", "vocals", { group_id: "g2" }),
    clip("solo", "vocals"),
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
  const [partial] = groupSeparationClips([
    clip("a", "vocals", {
      group_id: "g",
      settings: { stems: ["vocals", "drums", "bass", "other"] },
    }),
  ]);
  assert.equal(partial.complete, false);
  assert.equal(partial.expectedStems, 4);
});

test("a group cut by the page boundary waits for the next page, which the page loads itself", () => {
  const clips = [
    clip("a", "vocals", { group_id: "g1", settings: { stems: ["vocals"] } }),
    clip("b", "vocals", {
      group_id: "g2",
      settings: { stems: ["vocals", "drums"] },
    }),
  ];
  assert.deepEqual(ids(groupSeparationClips(clips, true)), ["g1"]);
  assert.deepEqual(ids(groupSeparationClips(clips, false)), ["g1", "g2"]);
  const page = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(
    page,
    /hasMore && groupSeparationClips\(clips\)\.length > groups\.length/,
  );
  assert.match(
    page,
    /if \(tailHidden\) void loadMore\(\);\s*\}, \[tailHidden, clips, loadMore\]\)/,
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
  const long = { ...source, durationS: 601 };
  const cases: [
    Partial<Parameters<typeof separateBlocker>[0]>,
    string | null,
    RegExp | string | null,
  ][] = [
    [{}, null, null],
    [{ source: { ...source, durationS: SEPARATE_MAX_SECONDS } }, null, null],
    [{ sourceBusy: true, sourceError: "x", source: null }, "source-busy", null],
    [{ sourceError: "Not audio.", source: null }, "source-error", "Not audio."],
    [
      { source: null, sourceExpired: true },
      "source",
      "Add a recording.",
    ],
    [
      { sourceExpired: true, source: long },
      "source-expired",
      "This recording expired. Add it again.",
    ],
    [{ source: long }, "too-long", /up to 10 minutes.*10:01/],
  ];
  for (const [patch, kind, reason] of cases) {
    const blocker = separateBlocker({ ...base, ...patch });
    assert.equal(blocker?.kind ?? null, kind, JSON.stringify(patch));
    if (typeof reason === "string") {
      assert.equal(blocker?.reason, reason);
    }
    if (reason instanceof RegExp) {
      assert.match(blocker?.reason ?? "", reason);
    }
  }
});

test("the estimate follows the model, the length and the overlap", () => {
  const cases: [string, number | null, boolean, number | null][] = [
    ["htdemucs", 180, true, 9],
    ["bs_roformer", 180, true, 13],
    ["bs_roformer", 180, false, 6],
    ["htdemucs", 1, true, 1],
    ["unknown", 180, true, null],
    ["htdemucs", null, true, null],
  ];
  for (const [family, seconds, overlap, want] of cases) {
    assert.equal(
      estimateSeparateSeconds(family, seconds, overlap),
      want,
      `${family} ${seconds} ${overlap}`,
    );
  }
  assert.match(separateCpuWarning("bs_roformer", "cpu") ?? "", /very slow/);
  assert.equal(separateCpuWarning("bs_roformer", "auto"), null);
  // The per-second rate was measured on HTDemucs only; other models get no promise.
  assert.match(separateCpuWarning("htdemucs", "cpu") ?? "", /a second per second/);
  assert.doesNotMatch(separateCpuWarning("audiosr", "cpu") ?? "", /second per second/);
  assert.match(separateCpuWarning("audiosr", "cpu") ?? "", /slower on the CPU/);
});

test("Overlap off asks for one pass; on sends nothing; a change is a reload", () => {
  assert.deepEqual(overlapRequest({ overlap: false }), {
    options: { num_overlap: 1 },
  });
  assert.deepEqual(overlapRequest({ overlap: true }), {});
  assert.deepEqual(overlapRequest(undefined), {});
  const reloads: [boolean | undefined, boolean, boolean][] = [
    [undefined, true, false],
    [undefined, false, true],
    [false, false, false],
    [false, true, true],
  ];
  for (const [loaded, wanted, want] of reloads) {
    assert.equal(overlapReloads(loaded, wanted), want);
  }
  assert.deepEqual(roformerOverlapLogic.initial([]), { overlap: true });
});

test("progress copy names the separation phases", () => {
  const p = { status: "Generating audio…", actionLabel: "Stop", canStop: true };
  assert.equal(
    separatePresentation(p, "generating", false)?.status,
    "Separating…",
  );
  assert.match(
    separatePresentation(p, "generating", true)?.status ?? "",
    /Reloading/,
  );
  assert.equal(
    separatePresentation(p, "finishing", false)?.status,
    "Saving stems…",
  );
  assert.equal(separatePresentation(null, "generating", false), null);
});

test("a separate run sends the source and options only", () => {
  const body = {
    workflow: "separate" as const,
    inputs: { source: { input_id: "abc" } },
    options: { num_overlap: 1 },
  };
  assert.deepEqual(buildAudioRunBody({ ...body, seed: undefined }), body);
  assert.equal(buildAudioRunBody({ workflow: "speak", text: "hi" }).text, "hi");
});

test("separation rows list only on Separate; sep catalog models only separate", () => {
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
  const seps = AUDIO_CPP_MODELS.filter((model) => model.task === "sep");
  for (const model of seps) {
    assert.deepEqual(audioCppWorkflowsFor(model), ["separate"]);
    assert.equal(audioCppModelSpeaks(model.id), false);
  }
});

test("a filter (voice, restore) names what it does and saves one file", () => {
  assert.equal(separateVerb("voice"), "Isolating the voice");
  assert.equal(separateVerb("restore"), "Restoring");
  assert.equal(separateVerb("stems"), "Separating");
  assert.equal(separateVerb(null), "Separating");
  const shown = { title: "t", subtitle: "s", status: "x" } as never;
  assert.equal(separatePresentation(shown, "finishing", false, "restore")?.status, "Saving…");
  assert.equal(separatePresentation(shown, "finishing", false, "stems")?.status, "Saving stems…");
  assert.equal(separateAnnouncement(1), "Saved the result.");
  assert.equal(separateAnnouncement(4), "Separated into 4 stems.");
});

test("a run merges the spec options over the tool panels', and sends none when empty", () => {
  assert.equal(separateRunOptions(undefined, {}), undefined);
  assert.deepEqual(separateRunOptions({ num_overlap: 1 }, {}), { num_overlap: 1 });
  assert.deepEqual(separateRunOptions(undefined, { audio_chunk_duration_sec: 5 }), {
    audio_chunk_duration_sec: 5,
  });
});

test("a history row names a lone output instead of counting stems", () => {
  const group = (roles: string[], complete = true, expectedStems = roles.length) => ({
    stems: roles.map((role, i) => clip(String(i), role)),
    complete,
    expectedStems,
  });
  assert.equal(separationGroupSummary(group(["voice"])), "Voice");
  assert.equal(separationGroupSummary(group(["restored"])), "Restored");
  assert.equal(separationGroupSummary(group(["vocals", "drums"])), "2 stems");
  assert.equal(separationGroupSummary(group(["vocals"], false, 4)), "1 of 4 stems");
});

test("Separate lists its models grouped by goal: stems, voice, restore, speakers", () => {
  const rows = AUDIO_CPP_MODELS.filter((model) => model.task === "sep")
    .reverse()
    .map((model) => ({ id: model.id }));
  const goals = pageModels("separate", rows, { isMac: false }).map(
    (row) => audioCppModelFor(row.id)?.goal ?? "stems",
  );
  const order = ["stems", "voice", "restore", "speakers"];
  assert.deepEqual(
    goals,
    [...goals].sort((a, b) => order.indexOf(a) - order.indexOf(b)),
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

test("host and page wiring for Separate", () => {
  const host = readSrc("features/audio/audio-page.tsx");
  assert.match(
    host,
    /ttsWorkflow === "separate" \? \(\s*<AudioActiveProvider value=\{active\}>\s*<SeparateRail/,
  );
  assert.match(host, /ttsWorkflow === "separate"\s*\? separate\.blocker/);
  assert.match(host, /reason: "The loaded model separates audio\."/);
  assert.match(
    host,
    /ttsWorkflow !== "speak" && ttsWorkflow !== "music"\s*\? \[\]/,
  );
  // switching to Transcribe mid-run would stop the run and drop the stem
  assert.match(
    host,
    /const handleSendStem = useCallback\([\s\S]*?if \(runBusy\(\)\) return;[\s\S]*?target\.workflow === "transcribe"\) \{\s*if \(!transitionWorkflow\("transcribe"\)\) return;[\s\S]*?useAudioTranscribeStore\.setState\(\{\s*source: \{\s*kind: "clip",\s*id: clip\.id/,
  );
  const generation = readSrc("features/audio/hooks/use-separate-generation.ts");
  assert.match(generation, /showRunResult\(\{[^}]*workflow: "separate"/);
  const page = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(page, /allowSavedVoice=\{false\}/);
  assert.match(page, /expiredMessage=\{SEPARATE_TRACK_EXPIRED_MESSAGE\}/);
  assert.match(page, /maxRecordSeconds=\{SEPARATE_MAX_SECONDS - 1\}/);
  // mixer sources pin the group; failed fetches retry, while unavailable stems are skipped
  assert.match(page, /useStemSources\(inputs, attempt\)/);
  assert.match(page, /sources\.failedIds\.length > 0/);
  assert.match(page, /setAttempt\(\(n\) => n \+ 1\)/);
  assert.match(page, /failed: sources\.failedIds\.includes\(clip\.id\)/);
});

test("a stem sent to Clone is adopted, so the old reference's transcript goes with it", () => {
  const host = readSrc("features/audio/audio-page.tsx");
  const send = host.slice(host.indexOf("const handleSendStem = useCallback("));
  assert.match(
    send,
    /if \(!transitionWorkflow\("clone"\)\) return;[\s\S]{0,160}?useAudioCloneStore\.getState\(\)\.adoptReference\(\{/,
  );
});

test("the empty results card names what the loaded model makes", () => {
  assert.equal(separateEmptyTitle("voice"), "The clean voice will appear here");
  assert.equal(separateEmptyTitle("restore"), "The restored audio will appear here");
  assert.equal(separateEmptyTitle("speakers"), "Each speaker will appear here");
  assert.equal(separateEmptyTitle("stems"), "Stems will appear here");
  assert.equal(separateEmptyTitle(null), "Stems will appear here");
});
