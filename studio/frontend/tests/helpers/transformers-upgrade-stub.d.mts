// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Types for transformers-upgrade-stub.mjs so a .ts test can drive it without `any`.
// Same shape as export-api-stub.d.mts next to it.

/** One argument object as the stub records it. Loose on purpose: a test asserts
 *  that a key is absent as well as that one is present, so every field is optional
 *  and reading an unknown one is a compile error rather than `any`. */
export interface RecordedArgs {
  modelName?: unknown;
  upgrade?: unknown;
  trustRemoteCodeFallback?: unknown;
  forceCancelActive?: unknown;
}

export declare const calls: { name: string; args: RecordedArgs[] }[];

export declare const state: {
  /** What `checkTransformersUpgrade` returns, or an Error it throws instead. */
  checkResult:
    | {
        upgrade: unknown;
        requiresTrustRemoteCode: boolean;
        latestTierActive: boolean;
        forces16Bit: boolean;
      }
    | Error;
  /** What the consent dialog resolves to. */
  consentResult: boolean;
  /** Whether the dialog actually ran an install, which is what forces 16-bit. */
  installRan: boolean;
  /** Whether the install unloaded the chat model, read once and cleared. */
  serverUnloadedChat: boolean;
};

export declare function resetStub(): void;

export declare function checkTransformersUpgrade(
  modelName: string,
  hfToken?: string | null,
): Promise<unknown>;

export declare function confirmTransformersUpgradeIfNeeded(
  args: unknown,
): Promise<boolean>;

export declare function installLatestTransformers(): Promise<never>;

export declare const useTransformersUpgradeDialogStore: {
  getState(): {
    installRan: boolean;
    consumeServerUnloadedChat(): boolean;
  };
};
