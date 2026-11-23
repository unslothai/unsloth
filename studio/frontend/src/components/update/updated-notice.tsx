// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { loadDesktopAppVersion } from "@/features/settings/desktop-app-version";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { useEffect } from "react";
import { toast } from "sonner";
import { compareVersions, sameVersion } from "./release-meta";
import { releases } from "./releases";
import { displayVersion } from "./update-offer-store";
import { useWhatsNew } from "./whats-new-store";

const SEEN_KEY = "unsloth_whats_new_seen_version";
const TARGET_KEY = "unsloth_whats_new_update_target";
// Lets an end-to-end run stand in for the desktop app version.
const E2E_VERSION_GLOBAL = "__unslothE2EAppVersion";

function storage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Called when the user starts an update, so the next launch can tell if it applied. */
export function rememberUpdateTarget(version: string): void {
  // Best effort: a full or read-only store must not block the update itself.
  try {
    storage()?.setItem(TARGET_KEY, version);
  } catch {
    // The post-update notice is skipped for this update.
  }
}

function appVersion(): Promise<string | null> {
  const e2e = (window as unknown as Record<string, unknown>)[
    E2E_VERSION_GLOBAL
  ];
  if (typeof e2e === "string") {
    return Promise.resolve(e2e);
  }
  // Web installs report PyPI versions, which the bundled release tags cannot be matched to.
  return isTauri ? loadDesktopAppVersion() : Promise.resolve(null);
}

type Outcome =
  | { kind: "none" }
  | { kind: "not-applied" }
  | { kind: "updated"; since: string };

/** Reads and advances the stored versions for this launch. */
function settle(store: Storage, version: string): Outcome {
  const seen = store.getItem(SEEN_KEY);
  const target = store.getItem(TARGET_KEY);
  store.setItem(SEEN_KEY, version);
  store.removeItem(TARGET_KEY);
  if (target && (compareVersions(version, target) ?? 0) < 0) {
    return { kind: "not-applied" };
  }
  if (seen && (compareVersions(version, seen) ?? 0) > 0) {
    return { kind: "updated", since: seen };
  }
  return { kind: "none" };
}

/** After an update: "Updated to x" with a link to the catch-up, or a note that it did not apply. */
export function UpdatedNotice() {
  const t = useT();
  useEffect(() => {
    let canceled = false;
    appVersion().then((version) => {
      const store = storage();
      if (canceled || !version || !store) {
        return;
      }
      const outcome = settle(store, version);
      if (outcome.kind === "not-applied") {
        toast.warning(t("updateNotes.notUpdated"), {
          description: t("updateNotes.notUpdatedBody", {
            version: displayVersion(version),
          }),
        });
        return;
      }
      if (outcome.kind === "none") {
        return;
      }
      const seen = outcome.since;
      const bundled = releases.some((r) => sameVersion(r.version, version));
      toast.success(
        t("updateNotes.updatedTo", { version: displayVersion(version) }),
        bundled
          ? {
              action: {
                label: t("updateNotes.seeWhatsNew"),
                onClick: () =>
                  useWhatsNew.getState().show({ version, since: seen }),
              },
            }
          : undefined,
      );
    });
    return () => {
      canceled = true;
    };
  }, [t]);
  return null;
}
