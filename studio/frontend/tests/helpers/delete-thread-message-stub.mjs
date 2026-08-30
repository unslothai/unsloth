// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for the two backend modules delete-thread-message imports. Importing the real
// chat-api pulls the auth flow and the login page in behind it, which is a lot of app to load
// for a function whose interesting behaviour is repository surgery.
//
// Every export here throws. The cases that use it pass `remoteId: undefined`, which is the
// branch that never touches the backend, so a throw is the assertion: if the delete path ever
// starts calling one of these without a remote id, the test fails loudly instead of quietly
// exercising a different code path than the one it claims to cover.

const unexpected = (name) => () => {
  throw new Error(`${name} must not be called when remoteId is undefined`);
};

export const listChatMessages = unexpected("listChatMessages");
export const ensureStoredChatThread = unexpected("ensureStoredChatThread");
export const isThreadIncognito = () => false;
export const syncStoredChatMessages = unexpected("syncStoredChatMessages");
