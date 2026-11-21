// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test, { beforeEach } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

// Run the hook outside a renderer: state is inert and effects fire inline.
const reactStub = `data:text/javascript,${encodeURIComponent(
  "export const useState=(i)=>[typeof i==='function'?i():i,()=>{}];" +
    "export const useEffect=(f)=>{f()};" +
    "export const useSyncExternalStore=(_s,g)=>g();" +
    "export const useRef=(v)=>({current:v});" +
    "export const useMemo=(f)=>f();export const useCallback=(f)=>f;" +
    "export const useDebugValue=()=>{};" +
    "export default {useState,useEffect,useSyncExternalStore,useRef,useMemo,useCallback,useDebugValue};",
)}`;
registerHooks({
  resolve: (specifier, context, next) =>
    specifier === "react" || specifier.startsWith("use-sync-external-store")
      ? { url: reactStub, shortCircuit: true }
      : next(specifier, context),
});

const requests: string[] = [];
globalThis.fetch = (async (input: string) => {
  requests.push(input);
  const user = /\/api\/users\/user-owner\//.test(input);
  return user
    ? new Response(JSON.stringify({ avatarUrl: "https://cdn/user.png" }))
    : new Response("{}", { status: 404 });
}) as typeof fetch;

const {
  _resetAvatarCacheForTest,
  readCachedHfOwnerAvatar,
  useHfOwnerAvatar,
} = await import("../src/features/hub/lib/hf-owner-avatar.ts");
const { getHfEndpoint } = await import("../src/lib/hf-endpoint.ts");

const STORAGE_KEY = "unsloth:hf-owner-avatars:v1";

beforeEach(() => {
  localStorage.removeItem(STORAGE_KEY);
  _resetAvatarCacheForTest();
});

test("readCachedHfOwnerAvatar hydrates persisted URLs across cache resets", () => {
  const endpoint = getHfEndpoint();
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify([
      {
        key: `${endpoint}::atomicchat`,
        url: "https://cdn-avatars.huggingface.co/v1/production/uploads/atomicchat.png",
        expiresAt: Date.now() + 60_000,
      },
    ]),
  );
  _resetAvatarCacheForTest();

  assert.equal(
    readCachedHfOwnerAvatar("AtomicChat"),
    "https://cdn-avatars.huggingface.co/v1/production/uploads/atomicchat.png",
  );
  assert.equal(
    readCachedHfOwnerAvatar("atomicchat"),
    "https://cdn-avatars.huggingface.co/v1/production/uploads/atomicchat.png",
  );
  assert.equal(readCachedHfOwnerAvatar("unknown-org"), null);
});

test("corrupt localStorage entries are ignored without throwing", () => {
  localStorage.setItem(STORAGE_KEY, "{not valid json");
  _resetAvatarCacheForTest();
  assert.equal(readCachedHfOwnerAvatar("AtomicChat"), null);

  localStorage.setItem(STORAGE_KEY, JSON.stringify([{ bad: true }]));
  _resetAvatarCacheForTest();
  assert.equal(readCachedHfOwnerAvatar("AtomicChat"), null);
});

test("hits and 404s both survive a relaunch, so neither is asked for again", async () => {
  const settle = () => new Promise((resolve) => setTimeout(resolve, 400));
  requests.length = 0;
  useHfOwnerAvatar("user-owner", true);
  useHfOwnerAvatar("gone-owner", true);
  await settle();
  // Org endpoint first, then the user endpoint, for each owner.
  assert.equal(requests.length, 4);
  assert.equal(readCachedHfOwnerAvatar("user-owner"), "https://cdn/user.png");

  requests.length = 0;
  _resetAvatarCacheForTest();
  useHfOwnerAvatar("user-owner", true);
  useHfOwnerAvatar("gone-owner", true);
  await settle();
  assert.deepEqual(requests, []);
});
