// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { installLatestTransformers } from "../api/transformers-upgrade-api";
import type {
  TransformersUpgradeInfo,
  TransformersUpgradePhase,
} from "../types";

type Resolver = (installed: boolean) => void;

// One in-flight consent; a new request resolves any prior pending one as declined.
let pendingResolver: Resolver | null = null;

interface TransformersUpgradeDialogStore {
  open: boolean;
  modelName: string | null;
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  errorMessage: string | null;
  trustRemoteCodeFallback: boolean;
  /** Without the user's "stop N chats" answer the install 409s and Retry can never succeed. */
  forceCancelActive: boolean;
  /** The install unloads the previous model; the custom-code fallback leaves it loaded. */
  installRan: boolean;
  /** Completed installs this session. The sidecar overlay changes every later answer, so caches
   * key on this. Survives `resolve`. */
  sidecarGeneration: number;
  /** Includes a swap that failed after the unload: callers must roll back on a later cancel. */
  serverUnloadedChat: boolean;
  /** Read-and-clear so each waiter consumes the signal exactly once. */
  consumeServerUnloadedChat: () => boolean;
  requestConsent: (
    modelName: string,
    upgrade: TransformersUpgradeInfo,
    options?: {
      trustRemoteCodeFallback?: boolean;
      forceCancelActive?: boolean;
    },
  ) => Promise<boolean>;
  install: () => Promise<void>;
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
    forceCancelActive: false,
    installRan: false,
    sidecarGeneration: 0,
    serverUnloadedChat: false,
    requestConsent: (modelName, upgrade, options) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({
          open: true,
          modelName,
          upgrade,
          phase: "consent",
          errorMessage: null,
          trustRemoteCodeFallback: Boolean(options?.trustRemoteCodeFallback),
          forceCancelActive: Boolean(options?.forceCancelActive),
          installRan: false,
        });
      }),
    consumeServerUnloadedChat: () => {
      const value = get().serverUnloadedChat;
      if (value) set({ serverUnloadedChat: false });
      return value;
    },
    install: async () => {
      const { upgrade, phase, forceCancelActive } = get();
      const version = upgrade?.pypi_version;
      if (!version || phase === "installing") return;
      const requestResolver = pendingResolver;
      set({ phase: "installing", errorMessage: null });
      let result: Awaited<ReturnType<typeof installLatestTransformers>>;
      try {
        result = await installLatestTransformers(version, forceCancelActive);
        // Latch before the resolver-identity guard: a superseded install may still have unloaded chat.
        if (result.model_unloaded) {
          set({ serverUnloadedChat: true });
        }
      } catch (error) {
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
        if (result.success) {
          // serverUnloadedChat is never reset here: a retry after unload reports false.
          set({
            installRan: true,
            sidecarGeneration: get().sidecarGeneration + 1,
          });
          get().resolve(true);
          return;
        }
        // The swap may have unloaded chat before failing; a version mismatch also names a release Retry
        // can use.
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
    resolve: (installed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({
        open: false,
        modelName: null,
        upgrade: null,
        phase: "consent",
        errorMessage: null,
        trustRemoteCodeFallback: false,
        forceCancelActive: false,
      });
      resolver?.(installed);
    },
  }));
