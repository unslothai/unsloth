// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { audioDetail, audioSummary, audioWorkflow, audioWorkflowOptions, runSiblings } =
  await import("../src/features/library/audio-items.ts");
const { EMPTY_FILTERS, filtersActive, matchesFilters } = await import(
  "../src/features/library/filters.ts"
);
const policy = await import("../src/features/audio/audio-page-policy.ts");
const workflows = await import("../src/features/audio/workflows.ts");

type Item = Parameters<typeof audioDetail>[0];

function clip(id: string, audio: Partial<NonNullable<Item["audio"]>> | null): Item {
  return {
    id,
    name: `${id}.wav`,
    source: audio ? "generated" : "uploaded",
    contentType: "audio/wav",
    sizeBytes: 4,
    createdAt: 1,
    updatedAt: 1,
    fileUrl: `/files/${id}`,
    threadId: null,
    threadTitle: null,
    textOnly: false,
    favorite: false,
    folderId: null,
    openedAt: null,
    model: null,
    audio: audio && {
      workflow: "speak",
      role: "output",
      groupId: null,
      durationS: 42,
      model: "unsloth/orpheus-3b",
      mode: null,
      variation: null,
      ...audio,
    },
  };
}

const stem = (id: string, role: string, groupId = "run") =>
  clip(id, { workflow: "separate", role, groupId });

test("a clip names the Audio page workflow that made it; other audio names none", () => {
  for (const id of ["speak", "clone", "edit", "convert", "music", "separate"]) {
    assert.equal(audioWorkflow(clip(id, { workflow: id }))?.id, id);
  }
  assert.equal(audioWorkflow(clip("upload", null)), null);
  // A workflow a newer server adds falls back to the plain audio icon.
  assert.equal(audioWorkflow(clip("future", { workflow: "dub" })), null);
});

test("a stem shows its stem, music its kind and take, speech nothing extra", () => {
  assert.equal(audioDetail(stem("v", "vocals")), "Vocals");
  assert.equal(audioDetail(clip("m", { workflow: "music", mode: "song", variation: 2 })), "Song 2");
  assert.equal(audioDetail(clip("s", { workflow: "music", mode: "sfx" })), "Sound effect");
  assert.equal(audioDetail(clip("e", { workflow: "music", role: "edit", mode: "edit" })), "Edit");
  assert.equal(audioDetail(clip("t", {})), null);
  assert.deepEqual(audioSummary(stem("v", "vocals")), ["0:42", "Vocals"]);
  assert.deepEqual(audioSummary(clip("t", { durationS: null })), ["Speak"]);
  assert.deepEqual(audioSummary(clip("upload", null)), []);
});

test("a run lists its own clips in the Audio page's order", () => {
  const items = [
    stem("other", "other"),
    stem("bass", "bass"),
    stem("vocals", "vocals"),
    stem("drums", "drums"),
    stem("elsewhere", "vocals", "another-run"),
    clip("loose", {}),
  ];
  const ids = (list: Item[]) => list.map((item) => item.id);
  assert.deepEqual(ids(runSiblings(items, items[1]!)), ["vocals", "drums", "bass", "other"]);
  assert.deepEqual(runSiblings(items, items[5]!), []);
  const takes = [2, 1, 3].map((n) =>
    clip(`take${n}`, { workflow: "music", role: "variation", groupId: "song", variation: n }),
  );
  assert.deepEqual(ids(runSiblings(takes, takes[0]!)), ["take1", "take2", "take3"]);
});

test("the Workflow filter offers only workflows present and keeps only their clips", () => {
  const items = [stem("v", "vocals"), clip("speech", {}), clip("upload.mp3", null)];
  assert.deepEqual(
    audioWorkflowOptions(items).map((workflow) => workflow.id),
    ["speak", "separate"],
  );
  const filters = { ...EMPTY_FILTERS, workflows: new Set(["separate"]) };
  assert.equal(filtersActive(filters), true);
  assert.equal(filtersActive(EMPTY_FILTERS), false);
  assert.deepEqual(
    items.filter((item) => matchesFilters(item, filters, false)).map((item) => item.id),
    ["v"],
  );
  assert.equal(
    items.every((item) => matchesFilters(item, EMPTY_FILTERS, false)),
    true,
  );
});

test("View in Audio opens the clip's own workflow instead of passing through Speak", () => {
  const origin = readSrc("features/library/origin.ts");
  assert.doesNotMatch(origin, /task: "text-to-speech"/);
  assert.match(origin, /isAudioWorkflowId\(workflow\) \? \{ \.\.\.search, workflow \} : search/);
});

function chatWithModelFor(lora: { audio_type: string; export_type: string }, deviceType = "linux") {
  const navigations: unknown[] = [];
  const toasts: unknown[] = [];
  const { chatWithModel } = loadWithStubs<{
    chatWithModel: (navigate: (to: unknown) => Promise<void>, item: unknown) => Promise<void>;
  }>(new URL("../src/features/library/actions.ts", import.meta.url), {
    fflate: {},
    "@/config/env": { usePlatformStore: { getState: () => ({ deviceType }) } },
    "@/features/audio/audio-page-policy": policy,
    "@/features/audio/workflows": workflows,
    "@/features/auth": { getAuthSessionEpoch: () => 0 },
    "@/features/chat": {
      listLoras: async () => ({ loras: [{ adapter_path: "/out/my-voice", ...lora }] }),
    },
    "@/features/model-picker": {},
    "@/i18n": { translate: (key: string) => key },
    "@/lib/audio-utils": {},
    "@/lib/api-base": { isTauri: false },
    "@/lib/native-files": {},
    "@/lib/toast": { toast: (title: unknown) => toasts.push(title) },
    "@/lib/video-utils": {},
    "./api": {},
    "./file-kind": {},
    "./file-name": {},
    "./start-chat": {},
  });
  const item = { name: "my-voice", model: { path: "/out/my-voice", origin: "training", exportType: "merged" } };
  return chatWithModel(async (to) => void navigations.push(to), item).then(() => ({ navigations, toasts }));
}

test("Chat with a speech fine-tune loads it on its Audio page with its audio type", async () => {
  // A native checkpoint needs its audio type for the custom-code approval and runtime checks.
  const native = await chatWithModelFor({ audio_type: "moss_tts_local", export_type: "merged" });
  assert.deepEqual(native.navigations, [
    {
      to: "/audio",
      search: { model: "/out/my-voice", loadId: "/out/my-voice", audioType: "moss_tts_local", workflow: "speak" },
    },
  ]);
  assert.deepEqual(native.toasts, []);
  // One the Audio page cannot load keeps the old pointer to its model menu.
  const other = await chatWithModelFor({ audio_type: "whisper", export_type: "lora" });
  assert.deepEqual(other.navigations, [{ to: "/audio" }]);
  assert.deepEqual(other.toasts, ["library.toast.speechModel"]);
});

test("Chat with on a Mac skips a fine-tune the Audio page cannot run there", async () => {
  const music = await chatWithModelFor({ audio_type: "minimax_music3", export_type: "merged" }, "mac");
  assert.deepEqual(music.navigations, [{ to: "/audio" }]);
  assert.deepEqual(music.toasts, ["library.toast.speechModel"]);
  const merged = await chatWithModelFor({ audio_type: "moss_tts_local", export_type: "merged" }, "mac");
  assert.equal((merged.navigations[0] as { search?: { loadId?: string } }).search?.loadId, "/out/my-voice");
});

test("the preview plays clips with the Audio page's waveform and stops a card that is playing", () => {
  const preview = readSrc("features/library/components/library-preview.tsx");
  assert.doesNotMatch(preview, /<audio src=\{url!\} controls/);
  assert.match(preview, /<Waveform[\s\S]*?onError=\{handleMediaError\}/);
  assert.match(preview, /if \(open\) stopLibraryAudio\(\);/);
  // The card's play button sits beside the card's own button, never inside it.
  const cards = readSrc("features/library/components/library-cards.tsx");
  assert.match(cards, /<\/button>\n\s*\{select && \([\s\S]*?\)\}\n\s*\{control\}/);
});

test("a separation is deleted in one call", () => {
  const separate = readSrc("features/audio/pages/separate-page.tsx");
  assert.match(separate, /await handleDeleteGroup\(\s*groupId,/);
  assert.match(readSrc("features/audio/api.ts"), /\/api\/inference\/audio\/gallery\/group\//);
});

test("the card player stops a clip whose card is gone and reports a failed stream once", async () => {
  const toasts: unknown[] = [];
  const players: FakeAudio[] = [];
  class FakeAudio {
    src = "";
    paused = true;
    error: { message: string } | null = null;
    onended: (() => void) | null = null;
    onerror: (() => void) | null = null;
    rejectPlay: ((reason: unknown) => void) | null = null;
    constructor() {
      players.push(this);
    }
    play() {
      this.paused = false;
      return new Promise<void>((_resolve, reject) => {
        this.rejectPlay = reject;
      });
    }
    pause() {
      this.paused = true;
    }
  }
  (globalThis as { Audio?: unknown }).Audio = FakeAudio;
  const playback = loadWithStubs<{
    toggleLibraryAudio: (item: unknown) => Promise<void>;
    stopLibraryAudioUnlessShown: (shown: ReadonlySet<string>) => void;
    useLibraryAudioPlaying: (id: string) => boolean;
  }>(new URL("../src/features/library/audio-playback.ts", import.meta.url), {
    "@/i18n": { translate: (key: string) => key },
    "@/lib/toast": { toast: { error: (title: unknown) => toasts.push(title) } },
    react: { useSyncExternalStore: (_subscribe: unknown, get: () => boolean) => get() },
    "./api": { errorMessage: String, fetchLibraryStreamUrl: async () => "blob:clip" },
  });
  const playing = (id: string) => playback.useLibraryAudioPlaying(id);

  await playback.toggleLibraryAudio({ id: "audio:a", name: "a.wav" });
  assert.equal(playing("audio:a"), true);
  playback.stopLibraryAudioUnlessShown(new Set(["audio:a", "audio:b"]));
  assert.equal(playing("audio:a"), true);
  playback.stopLibraryAudioUnlessShown(new Set(["audio:b"]));
  assert.equal(playing("audio:a"), false);
  assert.equal(players[0]!.paused, true);

  await playback.toggleLibraryAudio({ id: "audio:b", name: "b.wav" });
  players[0]!.error = { message: "network" };
  players[0]!.onerror?.();
  players[0]!.rejectPlay?.(new Error("aborted"));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(playing("audio:b"), false);
  assert.deepEqual(toasts, ["library.audio.playFailed"]);
});
