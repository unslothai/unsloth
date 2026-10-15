// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every export throws: tests use `remoteId: undefined`, so any backend call is a failure.

const unexpected = (name) => () => {
  throw new Error(`${name} must not be called when remoteId is undefined`);
};

export const listChatMessages = unexpected("listChatMessages");
export const ensureStoredChatThread = unexpected("ensureStoredChatThread");
export const syncStoredChatMessages = unexpected("syncStoredChatMessages");
