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
  /** Open the dialog for a paused load; resolves true only after a successful install. */
  requestConsent: (
    modelName: string,
    upgrade: TransformersUpgradeInfo,
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
    requestConsent: (modelName, upgrade) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({
          open: true,
          modelName,
          upgrade,
          phase: "consent",
          errorMessage: null,
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
      });
      resolver?.(installed);
    },
  }));
