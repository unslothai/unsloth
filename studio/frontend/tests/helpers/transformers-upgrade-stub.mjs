// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for src/features/transformers-upgrade/index.ts.
//
// The real barrel re-exports the consent dialog (.tsx) and node
// --experimental-strip-types cannot parse JSX, so importing the training-side gate here
// would pull in the whole React tree. Same cut as export-api-stub.mjs.
// `checkTransformersUpgrade` answers from `checkResult` (or throws it), the consent
// answer comes from `consentResult`, and every call is recorded on `calls`.

export const calls = [];

export const state = {
  checkResult: {
    upgrade: null,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
    installBreaksExactResume: false,
  },
  consentResult: true,
  installRan: false,
  serverUnloadedChat: false,
};

export function resetStub() {
  calls.length = 0;
  state.checkResult = {
    upgrade: null,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
    installBreaksExactResume: false,
  };
  state.consentResult = true;
  state.installRan = false;
  state.serverUnloadedChat = false;
}

export async function checkTransformersUpgrade(modelName, hfToken, options) {
  calls.push({
    name: "checkTransformersUpgrade",
    args: [modelName, hfToken, options],
  });
  if (state.checkResult instanceof Error) {
    throw state.checkResult;
  }
  return state.checkResult;
}

export async function confirmTransformersUpgradeIfNeeded(args) {
  calls.push({ name: "confirmTransformersUpgradeIfNeeded", args: [args] });
  return state.consentResult;
}

export async function installLatestTransformers() {
  throw new Error("installLatestTransformers is not exercised by this stub");
}

export const useTransformersUpgradeDialogStore = {
  getState() {
    return {
      installRan: state.installRan,
      consumeServerUnloadedChat() {
        const value = state.serverUnloadedChat;
        state.serverUnloadedChat = false;
        return value;
      },
    };
  },
};
