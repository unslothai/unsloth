// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { create } from "zustand";
import { installLatestTransformers } from "../api/transformers-upgrade-api";
import {
  canAutoInstallUpgrade,
  upgradeInstallVersion,
} from "../lib/upgrade-dialog-actions";
import type {
  TransformersUpgradeInfo,
  TransformersUpgradePhase,
} from "../types";

type Resolver = (installed: boolean) => void;

// One in-flight consent; a new request resolves any prior pending one as declined.
let pendingResolver: Resolver | null = null;

const AUTO_INSTALL_TOAST_ID = "transformers-auto-install";

interface TransformersUpgradeDialogStore {
  open: boolean;
  modelName: string | null;
  upgrade: TransformersUpgradeInfo | null;
  phase: TransformersUpgradePhase;
  errorMessage: string | null;
  /** Why an automatic install fell back to asking (scan not clean, offline, ...). */
  autoInstallNotice: string | null;
  /** The package scan flagged the release: the dialog offers no install. */
  installBlocked: boolean;
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
  /** `auto` runs with the dialog closed; anything short of success opens it. */
  install: (options?: { auto?: boolean }) => Promise<void>;
  resolve: (installed: boolean) => void;
}

export const useTransformersUpgradeDialogStore =
  create<TransformersUpgradeDialogStore>()((set, get) => ({
    open: false,
    modelName: null,
    upgrade: null,
    phase: "consent",
    errorMessage: null,
    autoInstallNotice: null,
    installBlocked: false,
    trustRemoteCodeFallback: false,
    forceCancelActive: false,
    installRan: false,
    sidecarGeneration: 0,
    serverUnloadedChat: false,
    requestConsent: (modelName, upgrade, options) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        const trustRemoteCodeFallback = Boolean(
          options?.trustRemoteCodeFallback,
        );
        const auto = canAutoInstallUpgrade(upgrade, trustRemoteCodeFallback);
        set({
          // A scanned PyPI release installs behind a notice; the dialog opens only if it must.
          open: !auto,
          modelName,
          upgrade,
          phase: "consent",
          errorMessage: null,
          autoInstallNotice: null,
          installBlocked: false,
          trustRemoteCodeFallback,
          forceCancelActive: Boolean(options?.forceCancelActive),
          installRan: false,
        });
        if (auto) void get().install({ auto: true });
      }),
    consumeServerUnloadedChat: () => {
      const value = get().serverUnloadedChat;
      if (value) set({ serverUnloadedChat: false });
      return value;
    },
    install: async (options) => {
      const auto = Boolean(options?.auto);
      const { upgrade, phase, forceCancelActive, installBlocked } = get();
      const version = upgradeInstallVersion(upgrade);
      if (!version || phase === "installing" || installBlocked) return;
      const requestResolver = pendingResolver;
      set({ phase: "installing", errorMessage: null });
      if (auto) {
        toast.loading(`Updating transformers to ${version}`, {
          id: AUTO_INSTALL_TOAST_ID,
          description:
            "This model needs a newer transformers. Installing the PyPI release after a package scan.",
        });
      }
      const current = () => pendingResolver === requestResolver;
      // Anything short of success shows the dialog, with the reason.
      const fail = (state: Partial<TransformersUpgradeDialogStore>) => {
        if (auto) toast.dismiss(AUTO_INSTALL_TOAST_ID);
        if (current()) set({ open: true, phase: "error", ...state });
      };
      let result: Awaited<ReturnType<typeof installLatestTransformers>>;
      try {
        result = await installLatestTransformers(
          version,
          forceCancelActive,
          auto,
        );
        // Latch before the resolver-identity guard: a superseded install may still have unloaded chat.
        if (result.model_unloaded) {
          set({ serverUnloadedChat: true });
        }
      } catch (error) {
        fail({
          errorMessage:
            error instanceof Error && error.message
              ? error.message
              : "Failed to install transformers.",
        });
        return;
      }
      if (result.success) {
        if (auto) {
          toast.success(`Updated transformers to ${version}`, {
            id: AUTO_INSTALL_TOAST_ID,
            description: undefined,
          });
        }
        if (current()) {
          // serverUnloadedChat is never reset here: a retry after unload reports false.
          set({
            installRan: true,
            sidecarGeneration: get().sidecarGeneration + 1,
          });
          get().resolve(true);
        }
        return;
      }
      if (result.requires_consent) {
        // The scan was not clean (or could not run): ask, as before this was automatic.
        if (auto) toast.dismiss(AUTO_INSTALL_TOAST_ID);
        if (current()) {
          set({
            open: true,
            phase: "consent",
            autoInstallNotice:
              result.message ||
              "Automatic install skipped: the release could not be scanned.",
          });
        }
        return;
      }
      // The swap may have unloaded chat before failing; a version mismatch also names a release Retry
      // can use.
      const latest = get().upgrade;
      fail({
        errorMessage: result.message || "Failed to install transformers.",
        serverUnloadedChat:
          get().serverUnloadedChat || Boolean(result.model_unloaded),
        installBlocked: result.scan_status === "flagged",
        ...(latest && (result.latest_version || result.latest_main_version)
          ? {
              upgrade: {
                ...latest,
                pypi_version: result.latest_version ?? latest.pypi_version,
                main_version: result.latest_main_version ?? latest.main_version,
              },
            }
          : {}),
      });
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
        autoInstallNotice: null,
        installBlocked: false,
        trustRemoteCodeFallback: false,
        forceCancelActive: false,
      });
      resolver?.(installed);
    },
  }));
