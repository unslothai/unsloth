// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { installLatestTransformers } from "../api/transformers-upgrade-api";
import type { TransformersUpgradeInfo, TransformersUpgradePhase } from "../types";

type Resolver = (installed: boolean) => void;

// One in-flight consent at a time; a new request resolves any prior pending one as
// declined so its promise never leaks (mirrors the remote-code consent store).
let pendingResolver: Resolver | null = null;

interface TransformersUpgradeDialogStore {
  open: boolean;
  modelName: string | null;
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  errorMessage: string | null;
  /** The model also declares custom (auto_map) code, so when no PyPI install is
   *  possible the load can still continue into the trust_remote_code consent
   *  flow as a last resort instead of hard-aborting. */
  trustRemoteCodeFallback: boolean;
  /** Open the dialog for a paused load; resolves true after a successful install
   *  (or, with no installable release, after the user continues into the
   *  custom-code fallback when one exists). */
  requestConsent: (
    modelName: string,
    upgrade: TransformersUpgradeInfo,
    options?: { trustRemoteCodeFallback?: boolean },
  ) => Promise<boolean>;
  /** Accept/Retry: run the sidecar install; on success resolve(true) and close. */
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
        });
      }),
    install: async () => {
      const { upgrade, phase } = get();
      const version = upgrade?.pypi_version;
      if (!version || phase === "installing") return;
      const requestResolver = pendingResolver;
      set({ phase: "installing", errorMessage: null });
      try {
        await installLatestTransformers(version);
      } catch (error) {
        // Only surface the failure if this consent is still the active one (a
        // newer request supersedes the dialog state).
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
        get().resolve(true);
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
      });
      resolver?.(installed);
    },
  }));
