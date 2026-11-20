// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { beforeEach } from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const {
  _resetAvatarCacheForTest,
  readCachedHfOwnerAvatar,
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
