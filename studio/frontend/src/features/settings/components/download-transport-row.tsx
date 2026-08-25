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
  // Distinct from "settings is null": a load that FAILED leaves us with no capability, and
  // refusing Xet for good on a flaky settings route would be worse than the race below.
  const [capabilityPending, setCapabilityPending] = useState(true);

  useEffect(() => {
    // The subscription is the only teardown: an in-flight load resolves into a cached value and
    // the unsubscribed setter is never called again.
    const unsubscribe = subscribeDownloadTransportSettings((next) => {
      setSettings(next);
      setCapabilityPending(false);
    });
    loadDownloadTransportSettings()
      .then(setSettings)
      .catch(() => undefined)
      .finally(() => setCapabilityPending(false));
    return unsubscribe;
  }, []);

  // Unknown counts as unavailable WHILE the first load is in flight. Xet used to be clickable
  // in that window on a machine without hf_xet, and the click stored a Xet preference locally
  // and install-wide that every later download then silently ignored.
  const xetUnavailable = capabilityPending || settings?.xetAvailable === false;
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
  // Only once we KNOW: while the capability is still pending the option is disabled, but
  // claiming hf_xet is missing would be asserting something we have not been told yet.
  const status =
    !capabilityPending && xetUnavailable && mode === "xet"
      ? xetReason
      : mode === "auto" && settings
        ? t("settings.general.downloads.autoCurrently", {
            transport:
              settings.autoResolvesTo === "xet"
                ? t("settings.general.downloads.xet")
                : t("settings.general.downloads.https"),
          })
        : null;

  // The backend's own words for WHY, on their own line rather than interpolated into the
  // sentence above. It comes from the Xet health check as free-form English, and there is
  // no translation to give it, so folding it into a translated sentence produced half a
  // sentence in the reader's language and half in English.
  // Only under Auto, which is the only branch above that has a reason to explain.
  const statusReason = mode === "auto" ? (settings?.autoReason ?? null) : null;

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
                  {disabled && !capabilityPending
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
        {statusReason ? (
          <span
            lang="en"
            className="max-w-[280px] text-right text-xs text-muted-foreground/70"
          >
            {statusReason}
          </span>
        ) : null}
      </div>
    </SettingsRow>
  );
}
