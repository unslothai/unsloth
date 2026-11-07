// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

test("an unload sent while the detector is still starting waits for that load", async () => {
  const calls: string[] = [];
  let finishLoad: () => void = () => {};
  const vad = loadWithStubs<typeof import("../src/features/chat/voice/vad-client.ts")>(
    new URL("../src/features/chat/voice/vad-client.ts", import.meta.url),
    {
      "@/features/audio/audio-cpp-catalog": { AUDIO_CPP_REPO: "unsloth/audio" },
      "@/features/auth": {
        authFetch: (url: string) => {
          calls.push(url);
          if (url.endsWith("/vad/load")) {
            return new Promise((resolve) => {
              finishLoad = () =>
                resolve(new Response(JSON.stringify({ loaded_model: "pulsevad" })));
            });
          }
          return Promise.resolve(new Response("{}"));
        },
      },
    },
  );
  const load = vad.loadVadModel("pulsevad");
  const unload = vad.unloadVadModel();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(calls, ["/api/inference/audio/vad/load"]);
  finishLoad();
  assert.equal(await load, "pulsevad");
  await unload;
  assert.deepEqual(calls, ["/api/inference/audio/vad/load", "/api/inference/audio/vad/unload"]);
});

test("an unload waits for every load in flight, not only the latest", async () => {
  const calls: string[] = [];
  const finishers: (() => void)[] = [];
  const vad = loadWithStubs<typeof import("../src/features/chat/voice/vad-client.ts")>(
    new URL("../src/features/chat/voice/vad-client.ts", import.meta.url),
    {
      "@/features/audio/audio-cpp-catalog": { AUDIO_CPP_REPO: "unsloth/audio" },
      "@/features/auth": {
        authFetch: (url: string) => {
          calls.push(url.split("/").pop() ?? url);
          if (url.endsWith("/vad/load")) {
            return new Promise((resolve) => {
              finishers.push(() =>
                resolve(new Response(JSON.stringify({ loaded_model: "pulsevad" }))),
              );
            });
          }
          return Promise.resolve(new Response("{}"));
        },
      },
    },
  );
  const first = vad.loadVadModel("pulsevad");
  const second = vad.loadVadModel("pulsevad");
  finishers[1]();
  await second;
  const unload = vad.unloadVadModel();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(calls, ["load", "load"]);
  finishers[0]();
  await first;
  await unload;
  assert.deepEqual(calls, ["load", "load", "unload"]);
});
