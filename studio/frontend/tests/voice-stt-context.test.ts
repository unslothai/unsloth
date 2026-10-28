// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

import type { transcribeAudioInput } from "../src/features/audio/api.ts";
import { sourceRefOf } from "../src/features/audio/audio-run-request.ts";
import type { useReferenceTranscribe } from "../src/features/audio/hooks/use-reference-transcribe.ts";
import {
  sttEngineForRepoId,
  sttSidecarKeyFor,
} from "../src/features/audio/stt-artifacts.ts";
import type { transcribeSourceWithProgress } from "../src/features/audio/transcribe-api.ts";
import { readTranscriptStream } from "../src/features/audio/transcript-stream.ts";
import { sttRequestError } from "../src/features/chat/adapters/stt-errors.ts";
import type {
  loadSttModel,
  transcribeAudioBlob,
} from "../src/features/chat/adapters/studio-model-dictation-adapter.ts";
import { completeChatAudioUpload } from "../src/features/chat/utils/chat-audio-upload.ts";

import {
  DEFAULT_STT_CONTEXT_LENGTH,
  STT_CONTEXT_LENGTHS,
  normalizeSttContextLength,
  sttContextBody,
} from "../src/features/settings/stores/stt-context.ts";
import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
const { store: storage } = installLocalStorageFake();
const { useVoiceSettingsStore } = await import(
  "../src/features/settings/stores/voice-settings-store.ts"
);

test("old or invalid preferences use Auto; valid fixed limits survive", () => {
  assert.equal(DEFAULT_STT_CONTEXT_LENGTH, 0);
  for (const value of [
    undefined,
    null,
    "4096",
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -1,
    1,
    1023,
    65537,
    4096.5,
  ]) {
    assert.equal(normalizeSttContextLength(value), 0);
  }
  for (const value of [0, ...STT_CONTEXT_LENGTHS, 5000]) {
    assert.equal(normalizeSttContextLength(value), value);
  }
});

test("only curated Qwen3-ASR on mtmd receives a context preference", () => {
  for (const model of ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]) {
    assert.deepEqual(sttContextBody(model, "mtmd", 0), { context_length: 0 });
    assert.deepEqual(sttContextBody(model, "mtmd", 4096), {
      context_length: 4096,
    });
    for (const engine of ["audiocpp", "gguf", "transformers", ""]) {
      assert.deepEqual(sttContextBody(model, engine, 4096), {});
    }
  }
  assert.deepEqual(sttContextBody("small", "mtmd", 4096), {});
  assert.deepEqual(sttContextBody("owner/custom-model", "mtmd", 4096), {});
});

test("existing saves default to Auto and a fixed choice persists across hydration", async () => {
  storage.set(
    "unsloth_voice_settings",
    JSON.stringify({
      version: 1,
      state: { sttModel: "qwen3-asr-0.6b", sttDevice: "cpu" },
    }),
  );
  await useVoiceSettingsStore.persist.rehydrate();
  assert.equal(useVoiceSettingsStore.getState().sttContextLength, 0);
  assert.equal(useVoiceSettingsStore.getState().sttDevice, "cpu");

  useVoiceSettingsStore.getState().setSttContextLength(8192);
  const saved = storage.get("unsloth_voice_settings");
  assert.ok(saved);
  assert.equal(JSON.parse(saved).state.sttContextLength, 8192);
  await useVoiceSettingsStore.persist.rehydrate();
  assert.equal(useVoiceSettingsStore.getState().sttContextLength, 8192);
  useVoiceSettingsStore.getState().setSttContextLength(0);
  await useVoiceSettingsStore.persist.rehydrate();
  assert.equal(useVoiceSettingsStore.getState().sttContextLength, 0);
});

// Run the actual request builders with a fake transport, avoiding the browser runtime
// imported by the full dictation adapter. This checks the emitted requests, including Auto=0.
function requestFunction<T>(
  path: string,
  name: string,
  next: string,
  scope: Record<string, unknown>,
): T {
  const source = readSrc(path);
  const start = source.search(
    new RegExp(`export (?:async )?function ${name}\\(`),
  );
  const end = source.indexOf(next, start);
  assert.ok(start >= 0 && end > start);
  const fn = source.slice(start, end).replace(/^export /, "");
  return new Function(
    ...Object.keys(scope),
    `${stripTypeScriptTypes(fn)}\nreturn ${name};`,
  )(...Object.values(scope)) as T;
}

test("loads, microphone/upload audio, sources and reference clips send the saved context", async () => {
  const requests: { url: string; init: RequestInit }[] = [];
  const scope = {
    useVoiceSettingsStore,
    sttContextBody,
    authFetch: async (url: string, init: RequestInit) => {
      requests.push({ url, init });
      return new Response(JSON.stringify({ text: "hello" }));
    },
    sttEngineFor: (model: string) => (model === "small" ? "gguf" : "mtmd"),
    resolveModelDictationLanguage: () => undefined,
    sttModelVariant: () => null,
    sttVariantBody: () => ({}),
    queueSttLifecycle: (fn: () => Promise<void>) => fn(),
    withModelLoadNotice: (
      _kind: string,
      _model: string,
      fn: () => Promise<void>,
    ) => fn(),
    JSON_HEADERS: { "Content-Type": "application/json" },
    readTranscriptStream: async () => ({ text: "hello" }),
    parseAudioJson: async (response: Response) => response.json(),
    transcribeUrl: () => "/api/inference/audio/inputs/input-1/transcribe",
  };
  const adapter = "features/chat/adapters/studio-model-dictation-adapter.ts";
  const load = requestFunction<typeof loadSttModel>(
    adapter,
    "loadSttModel",
    "export async function startSttDownload",
    scope,
  );
  const raw = requestFunction<typeof transcribeAudioBlob>(
    adapter,
    "transcribeAudioBlob",
    "export interface SttDownloadStatus",
    scope,
  );
  const source = requestFunction<typeof transcribeSourceWithProgress>(
    "features/audio/transcribe-api.ts",
    "transcribeSourceWithProgress",
    "export function fetchSttCapabilities",
    scope,
  );
  const reference = requestFunction<typeof transcribeAudioInput>(
    "features/audio/api.ts",
    "transcribeAudioInput",
    "export interface AudioRunResponse",
    scope,
  );
  const { useReferenceTranscribe: referenceHook } = loadWithStubs<{
    useReferenceTranscribe: typeof useReferenceTranscribe;
  }>(
    new URL(
      "../src/features/audio/hooks/use-reference-transcribe.ts",
      import.meta.url,
    ),
    {
      "@/features/chat/adapters/studio-model-dictation-adapter": {
        sttEngineFor: scope.sttEngineFor,
      },
      "@/features/settings": {
        useVoiceSettingsStore,
        withSttVariant: (model: string) => model,
      },
      "@/features/settings/stores/stt-context": { sttContextBody },
      react: {
        useCallback: (callback: unknown) => callback,
        useEffect: () => undefined,
        useRef: (current: unknown) => ({ current }),
        useState: (value: unknown) => [value, () => undefined],
      },
      "../api": { transcribeAudioInput: reference },
      "../audio-run-request": { sourceRefOf },
      "../catalog": { sttEngineForRepoId, sttSidecarKeyFor },
    },
  );

  for (const contextLength of [0, 4096]) {
    useVoiceSettingsStore.getState().setSttContextLength(contextLength);
    for (const [model, engine] of [
      ["qwen3-asr-0.6b", "mtmd"],
      ["qwen3-asr-1.7b", "mtmd"],
      ["small", "gguf"],
    ] as const) {
      requests.length = 0;
      await load(model, engine);
      await raw(new Blob(["audio"]), { model, engine });
      const sourceOptions = {
        model,
        engine,
        device: "auto",
        language: "",
        timestamps: false,
        speakers: false,
        signal: new AbortController().signal,
      };
      await source({ input_id: "input-1" }, "clip", sourceOptions, () => {});
      useVoiceSettingsStore.getState().setSttModel(model);
      const hook = referenceHook({ sttRepo: null, onText: () => undefined });
      await hook.transcribe({
        kind: "input",
        id: "input-1",
        name: "clip",
        durationS: 1,
      });
      const expected = engine === "mtmd" ? contextLength : undefined;
      for (const index of [0, 2, 3]) {
        assert.equal(
          JSON.parse(String(requests[index].init.body)).context_length,
          expected,
        );
      }
      assert.equal(
        new URL(requests[1].url, "http://localhost").searchParams.get(
          "context_length",
        ),
        expected === undefined ? null : String(expected),
      );
      await source(
        { input_id: "input-1" },
        "clip",
        {
          ...sourceOptions,
          providerId: "external-provider",
        },
        () => {},
      );
      assert.equal(
        JSON.parse(String(requests[4].init.body)).context_length,
        undefined,
      );
    }
  }
});

test("context changes spare active transcriptions and scope the unload to the selected model", () => {
  const source = readSrc("features/settings/tabs/voice-tab.tsx");
  const start = source.indexOf(
    "const next = normalizeSttContextLength(Number(value))",
  );
  const end = source.indexOf("setSttContextLength(next)", start);
  assert.ok(start >= 0 && end > start);
  assert.match(
    source.slice(start, end),
    /unloadSttModel\(sttEngineFor\(sttModel\), sttModel, \{\s*wait: false,/,
  );
});

test("an incomplete raw transcription preserves its error without committing partial text", async () => {
  const detail =
    "Transcription stopped at a token limit and is incomplete. Choose Auto or a larger voice context.";
  const raw = requestFunction<typeof transcribeAudioBlob>(
    "features/chat/adapters/studio-model-dictation-adapter.ts",
    "transcribeAudioBlob",
    "export interface SttDownloadStatus",
    {
      useVoiceSettingsStore,
      sttContextBody,
      resolveModelDictationLanguage: () => undefined,
      sttModelVariant: () => null,
      sttRequestError,
      sttErrorDetail: async (response: Response) =>
        (await response.json()).detail,
      authFetch: () =>
        Promise.resolve(
          new Response(JSON.stringify({ detail, text: "partial words" }), {
            status: 422,
          }),
        ),
    },
  );
  const fence = { generation: 1, owner: "thread-a", authSessionEpoch: 1 };
  const committed: string[] = [];
  await assert.rejects(
    completeChatAudioUpload({
      started: fence,
      current: () => fence,
      signal: new AbortController().signal,
      blocked: () => false,
      transcribe: () =>
        raw(new Blob(["audio"]), { model: "qwen3-asr-0.6b", engine: "mtmd" }),
      commit: (text) => committed.push(text),
    }),
    (error) => error instanceof Error && error.message === detail,
  );
  assert.deepEqual(committed, []);
});

test("a context error after streamed progress never becomes a completed transcript", async () => {
  const message = "Transcription stopped at a token limit and is incomplete.";
  const events = [
    { type: "progress", text: "partial words" },
    { type: "error", message },
  ]
    .map((event) => JSON.stringify(event))
    .join("\n");
  const updates: string[] = [];
  const response = new Response(events);
  assert.ok(response.body);
  await assert.rejects(
    readTranscriptStream(response.body, (progress) =>
      updates.push(progress.text),
    ),
    (error) => error instanceof Error && error.message === message,
  );
  assert.deepEqual(updates, ["partial words"]);
});
