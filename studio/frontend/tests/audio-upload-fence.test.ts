// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// A peer tab has written the fence and published the next account's token.
const sent: string[] = [];
class FakeXhr {
  upload = { onprogress: null as unknown };
  status = 201;
  response = { id: "in1" };
  responseType = "";
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  open(_method: string, url: string) {
    sent.push(url);
  }
  setRequestHeader() {}
  send() {
    this.onload?.();
  }
  abort() {}
}
(globalThis as Record<string, unknown>).XMLHttpRequest = FakeXhr;

const { uploadAudioInput } = loadWithStubs<{
  uploadAudioInput: (
    blob: Blob,
    name: string,
    options?: { onProgress?: (fraction: number | null) => void },
  ) => Promise<{ id: string }>;
}>(new URL("../src/features/audio/api.ts", import.meta.url), {
  "@/features/auth": {
    authFetch: async () => {
      throw new Error(
        "Another tab is switching accounts; this tab will reload.",
      );
    },
    getAuthToken: () => "next-account-token",
  },
  "@/lib/account-transition": { accountTransitionPending: () => true },
  "@/lib/api-base": { apiUrl: (path: string) => path },
  "@/lib/format-fastapi-error": {
    formatApiErrorBody: () => null,
    readFastApiError: async () => "error",
  },
  "./audio-run-request": {
    buildAudioRunBody: () => ({}),
    buildVoiceCreateBody: () => ({}),
    transcribeUrl: () => "/t",
  },
});

test("a reference upload with progress waits out another tab's account switch", async () => {
  await assert.rejects(
    uploadAudioInput(new Blob(["x"], { type: "audio/wav" }), "a.wav", {
      onProgress: () => {},
    }),
    /switching accounts/,
  );
  assert.deepEqual(sent, []);
});
