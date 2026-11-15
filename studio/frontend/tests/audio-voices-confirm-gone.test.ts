// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Voice = { id: string };
let server: Voice[] = [];
let gate: Promise<void> = Promise.resolve();
let deleteGate: Promise<void> = Promise.resolve();
let offline = false;

const { useAudioVoicesStore } = loadWithStubs<
  typeof import("../src/features/audio/stores/audio-voices-store.ts")
>(new URL("../src/features/audio/stores/audio-voices-store.ts", import.meta.url), {
  zustand,
  "../api": {
    listVoices: async () => {
      if (offline) throw new Error("offline");
      const snapshot = [...server];
      await gate;
      return snapshot;
    },
    createVoice: async () => ({ id: "x" }),
    updateVoice: async () => ({ id: "x" }),
    deleteVoice: async (id: string) => {
      await deleteGate;
      server = server.filter((voice) => voice.id !== id);
    },
  },
});

const store = () => useAudioVoicesStore.getState();

test("a list fetched before a delete lands does not bring the voice back", async () => {
  server = [{ id: "v-del" }];
  await store().refresh();
  let finishDelete!: () => void;
  deleteGate = new Promise((resolve) => (finishDelete = resolve));
  const removal = store().remove("v-del");
  await store().refresh();
  finishDelete();
  await removal;
  deleteGate = Promise.resolve();
  assert.deepEqual(store().voices.map((v) => v.id), []);
});

test("a voice saved in another window after our list loaded is not treated as deleted", async () => {
  server = [];
  await store().refresh();
  server = [{ id: "v-new" }];
  assert.equal(await store().confirmGone("v-new"), false);
  assert.deepEqual(store().voices.map((v) => v.id), ["v-new"]);
});

test("a voice that is really gone is confirmed gone", async () => {
  server = [];
  assert.equal(await store().confirmGone("v-new"), true);
});

test("a list request already in flight is not trusted for a newer voice", async () => {
  server = [];
  let release!: () => void;
  gate = new Promise((resolve) => (release = resolve));
  const stale = store().refresh();
  server = [{ id: "v-late" }];
  const check = store().confirmGone("v-late");
  gate = Promise.resolve();
  release();
  await stale;
  assert.equal(await check, false);
});

test("two checks waiting on the same old list both see a fresh one", async () => {
  server = [];
  let release!: () => void;
  gate = new Promise((resolve) => (release = resolve));
  const stale = store().refresh();
  server = [{ id: "v-late" }];
  const first = store().confirmGone("v-late");
  const second = store().confirmGone("v-late");
  gate = Promise.resolve();
  release();
  await stale;
  assert.deepEqual([await first, await second], [false, false]);
});

test("a failed list keeps the selection", async () => {
  offline = true;
  assert.equal(await store().confirmGone("anything"), false);
  offline = false;
});
