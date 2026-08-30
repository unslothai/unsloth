// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  canTransitionAudioMode,
  exactGgufLoadSelector,
  expectedGgufDownloadBytes,
  isTtsAudioType,
  macTtsPickAction,
  MINIMAX_MUSIC_DEFAULT_SECONDS,
  MINIMAX_MUSIC_FRAMES_PER_SECOND,
  MINIMAX_MUSIC_MAX_FRAMES,
  MINIMAX_MUSIC_MAX_SECONDS,
  mergeGalleryPage,
  micStreamRequestIsCurrent,
  minimaxMusicFramesForSeconds,
  nativeAudioInstructionsKind,
  persistedClipForGeneration,
  reconcileSttSelection,
  resolveSttLoadedModel,
  resolveSttResidency,
  selectAutoGgufVariant,
  stagedTtsLoadIsOwned,
  sttSelectionReady,
} from "../src/features/audio/audio-page-policy.ts";

const audioPageSource = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);
const chatApiSource = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);

test("mode transitions cancel generation but wait for non-cancellable work", () => {
  assert.equal(canTransitionAudioMode(null), true);
  assert.equal(canTransitionAudioMode("generating"), true);
  assert.equal(canTransitionAudioMode("loading"), false);
  assert.equal(canTransitionAudioMode("unloading"), false);
  assert.equal(canTransitionAudioMode("transcribing"), false);
});

test("staged TTS completion requires the same Speak ownership generation", () => {
  assert.equal(stagedTtsLoadIsOwned(4, 4, "speak"), true);
  assert.equal(stagedTtsLoadIsOwned(4, 5, "speak"), false);
  assert.equal(stagedTtsLoadIsOwned(4, 4, "transcribe"), false);
  assert.equal(stagedTtsLoadIsOwned(null, 4, "speak"), false);
});

test("cached GGUF quant labels remain exact when no filename is present", () => {
  assert.equal(exactGgufLoadSelector({ ggufVariant: "Q4_K_M" }), "Q4_K_M");
  assert.equal(
    exactGgufLoadSelector({
      ggufVariant: "Q4_K_M",
      ggufFilename: "model-q4_k_m.gguf",
    }),
    "model-q4_k_m.gguf",
  );
});

test("TTS load context matches the advertised generation ceiling", () => {
  assert.match(audioPageSource, /const TTS_MAX_TOKENS = 8192/);
  assert.match(audioPageSource, /max_seq_length: TTS_MAX_TOKENS/);
  assert.match(audioPageSource, /label="Max tokens"[\s\S]*max=\{TTS_MAX_TOKENS\}/);
});

test("MiniMax duration converts seconds to the official frame budget", () => {
  assert.equal(MINIMAX_MUSIC_FRAMES_PER_SECOND, 25);
  assert.equal(MINIMAX_MUSIC_DEFAULT_SECONDS, 30);
  assert.equal(MINIMAX_MUSIC_MAX_FRAMES, 9000);
  assert.equal(MINIMAX_MUSIC_MAX_SECONDS, 360);
  assert.equal(
    minimaxMusicFramesForSeconds(MINIMAX_MUSIC_DEFAULT_SECONDS),
    750,
  );
  assert.equal(minimaxMusicFramesForSeconds(1.99), 49);
  assert.equal(minimaxMusicFramesForSeconds(1.16), 29);
  assert.equal(minimaxMusicFramesForSeconds(360), 9000);
  assert.equal(minimaxMusicFramesForSeconds(999), 9000);
  assert.equal(minimaxMusicFramesForSeconds(0), 1);
  for (let frames = 1; frames <= MINIMAX_MUSIC_MAX_FRAMES; frames += 1) {
    assert.equal(
      minimaxMusicFramesForSeconds(frames / MINIMAX_MUSIC_FRAMES_PER_SECOND),
      frames,
    );
  }
});

test("native audio instruction fields match the runtime payload contract", () => {
  assert.equal(nativeAudioInstructionsKind("higgs_tts2"), "scene");
  assert.equal(nativeAudioInstructionsKind("moss_tts_local"), "style");
  assert.equal(nativeAudioInstructionsKind("minimax_music3"), "music");
  assert.equal(nativeAudioInstructionsKind("higgs_tts3"), null);
  assert.equal(nativeAudioInstructionsKind("moss_tts_nano"), null);
  assert.equal(nativeAudioInstructionsKind(null), null);
});

test("Audio sends model-specific duration and instruction payloads", () => {
  assert.match(
    audioPageSource,
    /musicGeneration\s*\? minimaxMusicFramesForSeconds\(minimaxMaxSeconds\)/,
  );
  assert.match(audioPageSource, /max=\{MINIMAX_MUSIC_MAX_SECONDS\}/);
  assert.match(
    audioPageSource,
    /instructionsKind !== null && instructions[\s\S]*audio_instructions: instructions/,
  );
  assert.match(audioPageSource, /\? "Scene description"/);
});

test("Mac rejects safetensors-only TTS and redirects sibling families", () => {
  assert.equal(
    macTtsPickAction({ isMac: true, isGguf: false, ggufSibling: null }),
    "reject",
  );
  assert.equal(
    macTtsPickAction({
      isMac: true,
      isGguf: false,
      ggufSibling: "org/model-GGUF",
    }),
    "use-gguf-sibling",
  );
  assert.equal(
    macTtsPickAction({ isMac: true, isGguf: true, ggufSibling: null }),
    "allow",
  );
});

test("Mac sibling resolution returns an exact managed-download file", () => {
  const variants = [
    {
      filename: "model-q4.gguf",
      quant: "Q4_K_M",
      size_bytes: 4_000,
      download_size_bytes: 3_500,
      downloaded: false,
    },
    {
      filename: "model-q8.gguf",
      quant: "Q8_0",
      size_bytes: 8_000,
      downloaded: true,
    },
  ];
  const cached = selectAutoGgufVariant(variants, "Q4_K_M");
  assert.equal(cached?.filename, "model-q8.gguf");

  const remote = selectAutoGgufVariant(
    variants.map((variant) => ({ ...variant, downloaded: false })),
    "Q4_K_M",
  );
  assert.equal(remote?.filename, "model-q4.gguf");
  assert.equal(remote && expectedGgufDownloadBytes(remote), 3_500);
});

test("only the generation's persisted gallery id is selected", () => {
  const refreshed = [{ id: "other-client" }, { id: "ours" }];
  assert.deepEqual(persistedClipForGeneration("ours", refreshed), {
    id: "ours",
  });
  assert.equal(persistedClipForGeneration(null, refreshed), null);
  assert.equal(persistedClipForGeneration("missing", refreshed), null);
});

test("Speak requires a supported TTS codec, not any audio model", () => {
  for (const codec of ["snac", "csm", "bicodec", "dac"])
    assert.equal(isTtsAudioType(codec), true);
  assert.equal(isTtsAudioType("csm", true), false);
  for (const codec of ["snac", "bicodec", "dac"])
    assert.equal(isTtsAudioType(codec, true), true);
  for (const codec of ["whisper", "audio_vlm", "", null])
    assert.equal(isTtsAudioType(codec), false);
});

test("STT controls require the selected sidecar to be resident", () => {
  const keyFor = (repo: string) => (repo === "org/asr" ? "asr" : repo);
  assert.equal(sttSelectionReady("org/asr", "asr", keyFor), true);
  assert.equal(sttSelectionReady("org/asr", null, keyFor), false);
  assert.equal(sttSelectionReady("org/asr", "other", keyFor), false);
  assert.equal(
    sttSelectionReady("org/asr", "asr", keyFor, "transformers", "gguf"),
    false,
  );
});

test("STT activation adopts resident sidecars and preserves only pending selections", () => {
  const sidecarKeyFor = (repo: string) =>
    repo === "org/asr-a" ? "asr-a" : repo === "org/asr-b" ? "asr-b" : repo;
  const repoIdForSidecarKey = (key: string) =>
    key === "asr-a" ? "org/asr-a" : key === "asr-b" ? "org/asr-b" : key;

  assert.equal(
    reconcileSttSelection({
      selectedRepo: null,
      loadedModel: "asr-b",
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-b",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: "asr-b",
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-b",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: null,
      preservePending: true,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-a",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: null,
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    null,
  );

  assert.equal(
    reconcileSttSelection({
      selectedRepo: "unsloth/whisper-small",
      loadedModel: "small",
      loadedEngine: "gguf",
      preservePending: false,
      sidecarKeyFor: () => "small",
      repoIdForSidecarKey: (_key, engine) =>
        engine === "gguf"
          ? "unslothai/whisper-small-GGUF"
          : "unsloth/whisper-small",
      engineForRepo: (repo) =>
        repo.endsWith("-GGUF") ? "gguf" : "transformers",
    }),
    "unslothai/whisper-small-GGUF",
  );
});

test("STT residency reads the selected engine instead of Transformers-only legacy fields", () => {
  const qwenStatus = {
    loaded_model: null,
    loading: false,
    transformers: { loaded_model: null, loading: false },
    gguf: { loaded_model: null, loading: false },
    mtmd: { loaded_model: "qwen3-asr-0.6b", loading: false },
  };
  assert.equal(
    resolveSttLoadedModel(qwenStatus, "mtmd", false),
    "qwen3-asr-0.6b",
  );

  const whisperStatus = {
    ...qwenStatus,
    mtmd: { loaded_model: null, loading: false },
    gguf: { loaded_model: "small", loading: false },
  };
  assert.equal(resolveSttLoadedModel(whisperStatus, null, false), "small");
  assert.deepEqual(resolveSttResidency(whisperStatus, "gguf", false), {
    model: "small",
    engine: "gguf",
  });
});

test("a pending engine selection is not replaced by an older resident sidecar", () => {
  const status = {
    loaded_model: "small",
    loading: false,
    transformers: { loaded_model: "small", loading: false },
    mtmd: { loaded_model: null, loading: false },
  };
  assert.equal(resolveSttLoadedModel(status, "mtmd", true), null);
  assert.equal(resolveSttLoadedModel(status, "mtmd", false), "small");
});

test("a resolved microphone permission stream is accepted only by its live request", () => {
  assert.equal(micStreamRequestIsCurrent(4, 4, true), true);
  assert.equal(micStreamRequestIsCurrent(4, 5, true), false);
  assert.equal(micStreamRequestIsCurrent(4, 4, false), false);
});

test("MediaRecorder setup failures release the acquired microphone stream", () => {
  // start() now takes a timeslice so the byte cap is observable; the release on failure
  // is what this test is about and is unchanged.
  assert.match(
    audioPageSource,
    /recorder\.start\(RECORDING_CHUNK_MS\);[\s\S]*?catch \{[\s\S]*?stopRecordStream\(\);/,
  );
});

test("leaving Audio clears an unresolved microphone permission wait", () => {
  assert.match(
    audioPageSource,
    /const stopAndDiscardRecording[\s\S]*micRequestGeneration\.current \+= 1;[\s\S]*micPendingGeneration\.current = null;[\s\S]*setMicRequestPending\(false\)/,
  );
});

test("routed picks wait in the URL until Audio is idle", () => {
  assert.match(
    audioPageSource,
    /if \(busyRef\.current !== null\) return;[\s\S]*handledRouteModel\.current = key;[\s\S]*handleModelSelect\(wanted/,
  );
  assert.match(audioPageSource, /\[\s*active,\s*busy,/);
});

test("file transcription cannot overlap a pending microphone permission", () => {
  assert.match(
    audioPageSource,
    /type="file"[\s\S]*disabled=\{[\s\S]*micRequestPending[\s\S]*onChange=/,
  );
});

test("gallery refresh preserves fallback selection and pagination identity", () => {
  assert.match(
    audioPageSource,
    /const generation = \+\+galleryRefreshGeneration\.current;[\s\S]*listAudioGallery\(0, PAGE_SIZE\);[\s\S]*if \(generation !== galleryRefreshGeneration\.current\) return/,
  );
  assert.match(
    audioPageSource,
    /!fallbackClipRef\.current[\s\S]*merged\.length > 0/,
  );
  assert.match(
    audioPageSource,
    /listAudioGallery\([\s\S]*galleryCache\.nextCursor[\s\S]*galleryCache\.nextCursor =[\s\S]*page\.next_before_mtime[\s\S]*new Set\(galleryCache\.clips\.map[\s\S]*filter\(\(clip\) => !known\.has\(clip\.id\)\)/,
  );
});

test("Audio transcription uses backend language auto-detection", () => {
  assert.match(
    audioPageSource,
    /transcribeAudioBlob\(blob, \{[\s\S]*model: key,[\s\S]*engine,[\s\S]*language: ""/,
  );
});

test("older STT status requests cannot overwrite newer residency", () => {
  assert.match(
    audioPageSource,
    /const generation = \+\+sttStatusRefreshGeneration\.current;[\s\S]*await fetchSttStatus[\s\S]*generation !== sttStatusRefreshGeneration\.current\) return;[\s\S]*catch \{[\s\S]*generation !== sttStatusRefreshGeneration\.current\) return;/,
  );
});

test("older TTS status requests cannot overwrite newer residency", () => {
  assert.match(
    audioPageSource,
    /const generation = \+\+ttsStatusRefreshGeneration\.current;[\s\S]*await getInferenceStatus\(\)[\s\S]*generation !== ttsStatusRefreshGeneration\.current\) return;[\s\S]*catch \{[\s\S]*generation !== ttsStatusRefreshGeneration\.current\) return;/,
  );
});

test("leaving Audio cancels an owned TTS load without touching a pre-request prompt", () => {
  assert.match(
    audioPageSource,
    /const loadRequestId = crypto\.randomUUID\(\);[\s\S]*load_request_id: loadRequestId/,
  );
  assert.match(
    audioPageSource,
    // Cancels under the target the request actually sent. The display id only maps back
    // to the load for a standard HF cache snapshot; a pinned directory elsewhere does not,
    // and the backend then refuses the cancellation and keeps loading.
    /const pending = pendingTtsLoad\.current;[\s\S]*pending\.controller\.abort\(\);[\s\S]*if \(pending\.requestStarted\)[\s\S]*unloadModel\(\{[\s\S]*model_path: pending\.loadTarget,[\s\S]*cancel_load_request_id: pending\.loadRequestId/,
  );
  assert.match(
    chatApiSource,
    /if \(options\?\.signal\?\.aborted\)[\s\S]*options\?\.onRequestStart\?\.\(\);[\s\S]*authFetch\("\/api\/inference\/load", \{[\s\S]*signal: options\?\.signal/,
  );
});

test("history-only downloads revoke their temporary blob URL", () => {
  assert.match(
    audioPageSource,
    /handleDownloadClipById[\s\S]*temporaryUrl = fetched\.url;[\s\S]*anchor\.click\(\);[\s\S]*URL\.revokeObjectURL\(url\)/,
  );
  assert.doesNotMatch(
    audioPageSource,
    /handleDownloadClipById[\s\S]{0,500}galleryCache\.srcById\.set/,
  );
});

test("a gallery refresh keeps the loaded scrollback instead of collapsing it", () => {
  const page = [{ id: "e" }, { id: "d" }, { id: "c" }];
  const cached = [
    { id: "d" },
    { id: "c" },
    { id: "b" },
    { id: "a" },
  ];

  // Newest page first, then the pages the user scrolled to.
  assert.deepEqual(mergeGalleryPage(page, cached), {
    clips: [{ id: "e" }, { id: "d" }, { id: "c" }, { id: "b" }, { id: "a" }],
    stitched: true,
  });
});

test("a clip this client deleted leaves the merged gallery", () => {
  const merged = mergeGalleryPage(
    [{ id: "c" }, { id: "a" }],
    [{ id: "b" }, { id: "a" }],
    "b",
  );
  assert.deepEqual(merged.clips, [{ id: "c" }, { id: "a" }]);
});

test("the selection only moves when its clip left the merged gallery", () => {
  assert.match(
    audioPageSource,
    /const \{ clips: merged, stitched \} = mergeGalleryPage\([\s\S]*!merged\.some\(\(c\) => c\.id === galleryCache\.selectedId\)/,
  );
  // Play an older clip, delete another, and the player must not jump to the newest.
  assert.doesNotMatch(
    audioPageSource,
    /galleryCache\.clips = page\.audio/,
  );
});

test("a superseded refresh still reports the clips its own fetch saw", () => {
  // Otherwise a generation whose clip really persisted was told it was not saved.
  assert.match(
    audioPageSource,
    /if \(generation !== galleryRefreshGeneration\.current\) return page\.audio;/,
  );
  assert.match(
    audioPageSource,
    /const refreshed = await refreshGallery\(\);[\s\S]*persistedClipForGeneration\(\s*generated\.clip_id,\s*refreshed,/,
  );
});

test("keeping the scrollback keeps the deeper pagination cursor", () => {
  // A clip carries no mtime, so adopting the page-0 cursor made loadMore re-walk loaded pages.
  assert.match(
    audioPageSource,
    /if \(!stitched\) \{[\s\S]*galleryCache\.hasMore = page\.has_more;[\s\S]*galleryCache\.nextCursor =/,
  );
  assert.match(audioPageSource, /setHasMore\(galleryCache\.hasMore\);/);
});

test("a cache with nothing in common with the page is dropped, not stitched", () => {
  // Another client can write a full page between two refreshes; stitching would render the
  // gap as contiguous and the preserved cursor could never fetch it.
  const merged = mergeGalleryPage(
    [{ id: "z" }, { id: "y" }],
    [{ id: "c" }, { id: "b" }],
  );
  assert.deepEqual(merged, { clips: [{ id: "z" }, { id: "y" }], stitched: false });
});

test("the recorder is gated on the same capability check the composer uses", () => {
  // Safari ships no MediaRecorder, and an http LAN origin (-H 0.0.0.0) is not a
  // secure context, so navigator.mediaDevices is undefined there. Without this
  // gate Record is enabled and its only outcome is "Could not access the
  // microphone", which blames the wrong thing.
  assert.match(
    audioPageSource,
    /StudioModelDictationAdapter\.isSupported\(\)/,
    "the audio page must reuse the dictation capability check",
  );
  assert.match(
    audioPageSource,
    /disabled=\{\s*!recordingSupported/,
    "Record must be disabled when recording is unsupported",
  );
  // File upload stays available, so transcription still works on those hosts.
  assert.match(audioPageSource, /accept="audio\/\*"/);
});

test("a recording is stopped at the sidecar's duration and size limits", () => {
  // Without a cap the page buffered an over-long recording in memory and uploaded it only
  // for the backend to refuse it. Mirrors _MAX_AUDIO_SECONDS and the b64 upload ceiling.
  assert.match(audioPageSource, /const RECORDING_MAX_SECONDS = 30 \* 60;/);
  assert.match(audioPageSource, /const RECORDING_MAX_BYTES = /);
  assert.match(audioPageSource, /const RECORDING_MAX_BYTES = 25 \* 1024 \* 1024;/);
  assert.match(
    audioPageSource,
    /if \(recordedBytes \+ event\.data\.size > RECORDING_MAX_BYTES\) \{\s*stopAtLimit\("size"\);/,
  );
  assert.match(
    audioPageSource,
    /window\.setTimeout\(\s*\(\) => stopAtLimit\("duration"\),\s*maxSeconds \* 1000,/,
  );
  // Uncompressed WAV on the PCM capture path (#9543) reaches the byte cap well
  // before the 30 minute one, so the duration enforced is the lower of the two.
  // Without this the recorder ran to 30 minutes and the upload was refused,
  // losing audio the user had already recorded.
  assert.match(
    audioPageSource,
    /const maxSeconds =\s*recorder instanceof PcmRecorder\s*\?\s*Math\.min\(\s*RECORDING_MAX_SECONDS,\s*recorder\.secondsWithin\(RECORDING_MAX_BYTES\),\s*\)\s*:\s*RECORDING_MAX_SECONDS;/,
  );
  assert.match(audioPageSource, /window\.clearTimeout\(durationTimer\);/);
});

test("the trained-model list applies the native-aware macOS policy", () => {
  assert.match(
    audioPageSource,
    /!isMac \|\|\s*trainedTtsCheckpointIsRunnableOnMac\(\s*lora\.audio_type,\s*lora\.export_type,?\s*\)/,
  );
});

test("the transcript download revokes its URL only after the click is consumed", () => {
  // Immediate revocation raced browsers that resolve a synthetic download navigation
  // asynchronously, leaving the action with no file.
  assert.match(
    audioPageSource,
    /anchor\.download = `\$\{\(transcribedName[\s\S]*?anchor\.click\(\);[\s\S]*?window\.setTimeout\(\(\) => URL\.revokeObjectURL\(url\), 0\);/,
  );
});

test("a complete first page drops cached rows the server no longer holds", () => {
  // has_more=false means the page IS everything on the server, so a cached clip below it
  // was deleted by another client or pruned by the size cap. Stitching it back rendered a
  // row that stayed on screen across every refresh and failed to play.
  const merged = mergeGalleryPage(
    [{ id: "c" }, { id: "b" }],
    [{ id: "c" }, { id: "b" }, { id: "a" }],
    undefined,
    false,
  );
  assert.deepEqual(merged, { clips: [{ id: "c" }, { id: "b" }], stitched: false });

  // With more on the server the scrollback is still real and is kept.
  const stitched = mergeGalleryPage(
    [{ id: "c" }, { id: "b" }],
    [{ id: "c" }, { id: "b" }, { id: "a" }],
    undefined,
    true,
  );
  assert.deepEqual(stitched.clips, [{ id: "c" }, { id: "b" }, { id: "a" }]);
  assert.equal(stitched.stitched, true);
});

test("the refresh passes the page's completeness into the merge", () => {
  assert.match(
    audioPageSource,
    /mergeGalleryPage\(\s*page\.audio,\s*galleryCache\.clips,\s*removedId,\s*page\.has_more,/,
  );
});

test("generating waits for the transcribe release the mode switch started", () => {
  // Switching straight from Transcribe to Speak with a speech model already resident needs
  // no load, so the load path's gate never runs and Generate could allocate beside the
  // dictation model, which OOMs a device that fits either one alone.
  assert.match(
    audioPageSource,
    /const handleGenerate = useCallback\(async \(\) => \{[\s\S]{0,900}?const releaseInFlight = pendingTranscribeRelease\.current;[\s\S]{0,200}?if \(releaseInFlight && !\(await releaseInFlight\)\) \{[\s\S]{0,80}?setMode\("transcribe"\);/,
  );
});

test("a failed transcribe release puts the page back in Transcribe", () => {
  // Otherwise the pill reads Speak while the sidecar still holds its model, and nothing
  // on screen offers the Eject that would retry the unload.
  assert.match(
    audioPageSource,
    /void release\.then\(\(released\) => \{[\s\S]{0,600}?if \(!released && modeRef\.current === "speak"\) setMode\("transcribe"\);/,
  );
});

test("a gguf selection served by Transformers still reads as resident", () => {
  // Without whisper-server the backend serves and loads the equivalent Transformers
  // model, so residency for the pick lives in that block. Reading only the gguf block
  // returned nothing on the refresh that completes the load (preserveSelected is true
  // there) and the Transcribe controls stayed disabled until the page was revisited.
  const status = {
    transformers: { loaded_model: "small", available: true },
    gguf: { loaded_model: null, available: false },
  };
  assert.deepEqual(resolveSttResidency(status, "gguf", true), {
    model: "small",
    engine: "gguf",
  });
  assert.equal(resolveSttLoadedModel(status, "gguf", true), "small");

  // whisper.cpp present: its own block still answers, and an empty one is still empty.
  const live = {
    transformers: { loaded_model: "small", available: true },
    gguf: { loaded_model: "base", available: true },
  };
  assert.deepEqual(resolveSttResidency(live, "gguf", true), {
    model: "base",
    engine: "gguf",
  });
  assert.equal(resolveSttResidency({ gguf: { loaded_model: null } }, "gguf", true), null);
});

test("generation is claimed before the transcribe release is awaited", () => {
  // The button only disables on `busy`, so awaiting first let several clicks through and
  // each resumed into its own generateAudio while generateAbort tracked only the last.
  assert.match(
    audioPageSource,
    /if \(busyRef\.current\) return;\s*busyRef\.current = "generating";\s*setBusy\("generating"\);\s*const releaseInFlight = pendingTranscribeRelease\.current;\s*if \(releaseInFlight/,
  );
  // And a release that failed hands the slot back rather than wedging the button.
  assert.match(
    audioPageSource,
    /if \(releaseInFlight && !\(await releaseInFlight\)\) \{\s*busyRef\.current = null;\s*setBusy\(null\);\s*setMode\("transcribe"\);/,
  );
});
