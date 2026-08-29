// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useTauriUpdateController } from "@/hooks/tauri-update-context";
import { useT } from "@/i18n";
import { restartPlan } from "@/lib/update-preparation";
import type { ReactElement } from "react";
import { SettingsRow } from "./settings-row";

function formatDesktopVersion(version: string): string {
  return version.startsWith("v") ? version : `v${version}`;
}

/** Explains the automatic check, so the button does not read as the only path. */
export function DesktopUpdateNote(): ReactElement | null {
  const t = useT();
  const update = useTauriUpdateController();
  if (!update) return null;
  return (
    <p className="pb-1 text-xs text-muted-foreground leading-relaxed">
      {t("settings.about.update.desktopManaged")}
    </p>
  );
}

// Lives in General, under the version rows it acts on: that is the tab the
// settings dialog opens on. Desktop-only, outside Tauri there is no controller
// in context and the row renders nothing.
export function DesktopUpdateControl(): ReactElement | null {
  const t = useT();
  const update = useTauriUpdateController();
  if (!update) return null;

  const checking = update.status === "checking";
  const preparing = update.status === "preparing";
  const ready = update.status === "ready";
  const fastRestart = ready && restartPlan(update.preparation) === "fast";
  // A running install owns the update screen; no second "Update now".
  const inFlight =
    update.status === "updating-backend" ||
    update.status === "downloading" ||
    update.status === "installing";
  const busy = checking || inFlight || preparing;
  const available = update.info !== null && !checking;
  const checkFailed = update.checkError !== null && !available;

  let label: string;
  let description: string;
  if (checking) {
    label = t("settings.about.update.desktopChecking");
    description = t("settings.about.update.desktopCheckingDescription");
  } else if (available && update.info) {
    label = t("settings.about.update.desktopAvailable", {
      version: formatDesktopVersion(update.info.version),
    });
    description = update.isExternalServer
      ? t("settings.about.update.desktopExternalServer")
      : update.updatePolicyMode === "manual_linux_package"
        ? t("settings.about.update.desktopManualInstall")
        : fastRestart
          ? t("settings.about.update.desktopReadyToRestartDescription")
          : ready
            ? t("settings.about.update.desktopReadyToInstallDescription")
            : preparing
              ? t("settings.about.update.desktopPreparingDescription")
              : t("settings.about.update.desktopAvailableDescription");
  } else if (checkFailed) {
    label = t("settings.about.update.desktopCheckFailed");
    // Keep the raw reason: failures can come from the network, HTTP response,
    // release manifest, or updater itself.
    description = update.checkError ?? label;
  } else if (update.hasChecked) {
    label = t("settings.about.update.desktopCurrent");
    description = t("settings.about.update.desktopCurrentDescription");
  } else {
    label = t("settings.about.update.desktopReady");
    description = t("settings.about.update.desktopReadyDescription");
  }

  const action = available
    ? update.updatePolicyMode === "manual_linux_package"
      ? t("settings.about.update.openReleasePage")
      : fastRestart
        ? t("settings.about.update.restartToUpdate")
        : ready
          ? t("settings.about.update.finishUpdate")
          : preparing
            ? t("settings.about.update.preparing")
            : t("settings.about.update.updateNow")
    : checkFailed
      ? t("settings.about.update.retryCheck")
      : update.hasChecked
        ? t("settings.about.update.checkAgain")
        : t("settings.about.update.checkForUpdates");

  return (
    <SettingsRow
      label={label}
      description={<span aria-live="polite">{description}</span>}
    >
      {available && update.isExternalServer ? null : (
        <Button
          size="sm"
          variant={available ? "default" : "outline"}
          disabled={busy}
          aria-busy={busy}
          onClick={() => {
            if (available) {
              void update.installUpdate();
            } else {
              void update.checkForUpdate();
            }
          }}
        >
          {checking ? t("settings.about.update.checking") : action}
        </Button>
      )}
    </SettingsRow>
  );
}
