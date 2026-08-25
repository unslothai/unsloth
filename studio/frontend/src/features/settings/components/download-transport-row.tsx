// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type TransportMode,
  useHttpPartialsResumable,
  useTransportMode,
} from "@/features/hub";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import {
  type DownloadTransportSettings,
  loadDownloadTransportSettings,
  subscribeDownloadTransportSettings,
} from "../api/download-transport";
import { SettingsRow } from "./settings-row";

const OPTIONS: {
  value: TransportMode;
  labelKey: TranslationKey;
  hintKey: TranslationKey;
}[] = [
  {
    value: "http",
    labelKey: "settings.general.downloads.https",
    hintKey: "settings.general.downloads.httpsHint",
  },
  {
    value: "xet",
    labelKey: "settings.general.downloads.xet",
    hintKey: "settings.general.downloads.xetHint",
  },
  {
    value: "auto",
    labelKey: "settings.general.downloads.auto",
    hintKey: "settings.general.downloads.autoHint",
  },
];

export function DownloadTransportRow() {
  const t = useT();
  const [mode, setMode] = useTransportMode();
  // huggingface_hub 1.18 made the HTTP writer process-unique, so an interrupted transfer is
  // refetched from zero unless the resumable writer is restored. Saying HTTPS "resumes where
  // it stopped" there is exactly backwards for someone picking it to keep their progress.
  const partialsResumable = useHttpPartialsResumable();
  const [settings, setSettings] = useState<DownloadTransportSettings | null>(
    null,
  );

  useEffect(() => {
    // The subscription is the only teardown: an in-flight load resolves into a cached value and
    // the unsubscribed setter is never called again.
    const unsubscribe = subscribeDownloadTransportSettings(setSettings);
    loadDownloadTransportSettings()
      .then(setSettings)
      .catch(() => undefined);
    return unsubscribe;
  }, []);

  const xetUnavailable = settings?.xetAvailable === false;
  // The localized string first. The backend has exactly one reason for Xet being
  // unavailable and it is English prose, so preferring it made the translated key
  // unreachable and showed English to every other locale. A future backend reason we
  // do not have a translation for still gets through.
  const serverReason = settings?.xetUnavailableReason;
  const xetReason =
    !serverReason || /hf_xet is not installed/i.test(serverReason)
      ? t("settings.general.downloads.xetMissing")
      : serverReason;

  // Xet is selected but cannot run here: say so, rather than leaving a selected
  // option that silently downloads over HTTPS.
  const status =
    xetUnavailable && mode === "xet"
      ? xetReason
      : mode === "auto" && settings
        ? settings.autoReason
          ? t("settings.general.downloads.autoCurrentlyReason", {
              transport:
                settings.autoResolvesTo === "xet"
                  ? t("settings.general.downloads.xet")
                  : t("settings.general.downloads.https"),
              reason: settings.autoReason,
            })
          : t("settings.general.downloads.autoCurrently", {
              transport:
                settings.autoResolvesTo === "xet"
                  ? t("settings.general.downloads.xet")
                  : t("settings.general.downloads.https"),
            })
        : null;

  return (
    <SettingsRow
      alignTop={true}
      label={t("settings.general.downloads.transport")}
      description={t(
        partialsResumable
          ? "settings.general.downloads.transportDescription"
          : "settings.general.downloads.transportDescriptionNoResume",
      )}
      hint={t("settings.general.downloads.transportHint")}
    >
      <div className="flex flex-col items-end gap-1.5">
        <div
          role="radiogroup"
          aria-label={t("settings.general.downloads.transport")}
          className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
        >
          {OPTIONS.map((opt) => {
            const active = mode === opt.value;
            const disabled = opt.value === "xet" && xetUnavailable;
            return (
              <Tooltip key={opt.value}>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={active}
                    aria-disabled={disabled || undefined}
                    onClick={() => {
                      if (!disabled) setMode(opt.value);
                    }}
                    className={cn(
                      "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors",
                      disabled
                        ? "cursor-not-allowed text-muted-foreground/45"
                        : active
                          ? "hub-tab-toggle-pill text-foreground"
                          : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <span className="relative z-10">{t(opt.labelKey)}</span>
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  side="bottom"
                  sideOffset={6}
                  className="max-w-[260px]"
                >
                  {disabled
                    ? xetReason
                    : t(
                        opt.value === "http" && !partialsResumable
                          ? "settings.general.downloads.httpsHintNoResume"
                          : opt.hintKey,
                      )}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
        {status ? (
          <span className="max-w-[280px] text-right text-xs text-muted-foreground">
            {status}
          </span>
        ) : null}
      </div>
    </SettingsRow>
  );
}
