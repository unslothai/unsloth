// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for features/transformers-upgrade: the real barrel re-exports a .tsx dialog
// that node --experimental-strip-types cannot parse.

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

// Same rule as lib/upgrade-dialog-actions.ts: the PyPI release, else transformers main.
export function upgradeInstallVersion(upgrade) {
  if (upgrade?.supported_in_pypi && upgrade?.pypi_version) {
    return upgrade.pypi_version;
  }
  if (upgrade?.supported_in_main && upgrade?.main_version) {
    return upgrade.main_version;
  }
  return null;
}
