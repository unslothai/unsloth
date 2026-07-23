// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { installLatestTransformers } from "../api/transformers-upgrade-api";
import type { TransformersUpgradeInfo, TransformersUpgradePhase } from "../types";

type Resolver = (installed: boolean) => void;

// One in-flight consent; a new request resolves any prior pending one as declined.
let pendingResolver: Resolver | null = null;
let pendingOwner: unknown;
let cancelAfterInstall = false;

interface TransformersUpgradeDialogStore {
  open: boolean;
  modelName: string | null;
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  errorMessage: string | null;
  /** Model ships custom code; without a PyPI install the load may fall back to trust_remote_code. */
  trustRemoteCodeFallback: boolean;
  /** True once this consent's install completed. The install unloads the previous
   *  model before swapping, so the caller must treat it as already unloaded; the
   *  custom-code fallback resolves true without installing and leaves it loaded. */
  installRan: boolean;
  /** True when the server unloaded the active chat model during this consent,
   *  including a swap that failed AFTER the unload: callers must then treat
   *  their previous model as gone and roll back on any later cancel. */
  serverUnloadedChat: boolean;
  /** Read-and-clear serverUnloadedChat: each waiter consumes the signal once,
   *  so a superseding consent can neither erase it before the old waiter reads
   *  it nor leak it into an unrelated later load. */
  consumeServerUnloadedChat: () => boolean;
  /** Open the dialog for a paused load; resolves true on install success or custom-code fallback. */
  requestConsent: (
    modelName: string,
    upgrade: TransformersUpgradeInfo,
    options?: { trustRemoteCodeFallback?: boolean; owner?: unknown },
  ) => Promise<boolean>;
  /** Accept/Retry: run the install; on success resolve(true) and close. */
  install: () => Promise<void>;
  /** Decline immediately, or wait for an in-flight install to settle first. */
  cancelPending: (owner?: unknown) => void;
  resolve: (installed: boolean) => void;
}

export const useTransformersUpgradeDialogStore =
  create<TransformersUpgradeDialogStore>()((set, get) => ({
    open: false,
    modelName: null,
    upgrade: null,
    phase: "consent",
    errorMessage: null,
    trustRemoteCodeFallback: false,
    installRan: false,
    serverUnloadedChat: false,
    requestConsent: (modelName, upgrade, options) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        pendingOwner = options?.owner;
        cancelAfterInstall = false;
        set({
          open: true,
          modelName,
          upgrade,
          phase: "consent",
          errorMessage: null,
          trustRemoteCodeFallback: Boolean(options?.trustRemoteCodeFallback),
          installRan: false,
        });
      }),
    consumeServerUnloadedChat: () => {
      const value = get().serverUnloadedChat;
      if (value) set({ serverUnloadedChat: false });
      return value;
    },
    install: async () => {
      const { upgrade, phase } = get();
      const version = upgrade?.pypi_version;
      if (!version || phase === "installing") return;
      const requestResolver = pendingResolver;
      set({ phase: "installing", errorMessage: null });
      let result: Awaited<ReturnType<typeof installLatestTransformers>>;
      try {
        result = await installLatestTransformers(version);
        // Latch the server-side unload IMMEDIATELY, before any resolver-identity
        // guard: even a superseded consent's install may have unloaded the chat
        // model, and the signal must survive for whichever load consumes it next.
        if (result.model_unloaded) {
          set({ serverUnloadedChat: true });
        }
      } catch (error) {
        if (pendingResolver === requestResolver && cancelAfterInstall) {
          get().resolve(false);
          return;
        }
        // Ignore the failure if a newer request superseded this consent.
        if (pendingResolver === requestResolver) {
          set({
            phase: "error",
            errorMessage:
              error instanceof Error && error.message
                ? error.message
                : "Failed to install transformers.",
          });
        }
        return;
      }
      if (pendingResolver === requestResolver) {
        if (cancelAfterInstall) {
          get().resolve(false);
          return;
        }
        if (result.success) {
          // serverUnloadedChat was latched above (and is never reset here): a
          // retry after a failed-after-unload attempt reports false because the
          // model is already gone, and a superseded install may have set it too.
          set({ installRan: true });
          get().resolve(true);
          return;
        }
        // Structured failure: the swap failed but may have already unloaded the
        // chat model; record that so a later cancel still rolls the caller back.
        // A version mismatch also carries the superseding release, so Retry
        // re-requests a version that can actually succeed.
        const { upgrade } = get();
        set({
          phase: "error",
          errorMessage: result.message || "Failed to install transformers.",
          serverUnloadedChat:
            get().serverUnloadedChat || Boolean(result.model_unloaded),
          ...(result.latest_version && upgrade
            ? { upgrade: { ...upgrade, pypi_version: result.latest_version } }
            : {}),
        });
      }
    },
    cancelPending: (owner) => {
      if (!pendingResolver) return;
      if (owner !== undefined && pendingOwner !== owner) return;
      if (get().phase === "installing") {
        cancelAfterInstall = true;
        return;
      }
      get().resolve(false);
    },
    resolve: (installed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      pendingOwner = undefined;
      cancelAfterInstall = false;
      set({
        open: false,
        modelName: null,
        upgrade: null,
        phase: "consent",
        errorMessage: null,
        trustRemoteCodeFallback: false,
      });
      resolver?.(installed);
    },
  }));
