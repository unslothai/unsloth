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
  // huggingface_hub 1.18 refetches an interrupted transfer from zero, so promising HTTPS
  // "resumes where it stopped" there is backwards.
  const partialsResumable = useHttpPartialsResumable();
  const [settings, setSettings] = useState<DownloadTransportSettings | null>(
    null,
  );
  // Distinct from "settings is null": a FAILED load must not disable Xet for good.
  const [capabilityPending, setCapabilityPending] = useState(true);

  useEffect(() => {
    // The only teardown: an in-flight load resolves into the cache, not into this setter.
    const unsubscribe = subscribeDownloadTransportSettings((next) => {
      setSettings(next);
      setCapabilityPending(false);
    });
    // Refreshed on mount: the cache can hold a mode another browser changed, or a stale
    // Auto verdict.
    loadDownloadTransportSettings({ refresh: true })
      .then(setSettings)
      .catch(() => undefined)
      .finally(() => setCapabilityPending(false));
    return unsubscribe;
  }, []);

  // Unknown counts as unavailable while the first load is in flight: clicking Xet there
  // stored a preference every later download silently ignored.
  const xetUnavailable = capabilityPending || settings?.xetAvailable === false;
  // The localized string first: the backend's one reason is English prose, and preferring it
  // showed English to every other locale. An unknown future reason still gets through.
  const serverReason = settings?.xetUnavailableReason;
  const xetReason =
    !serverReason || /hf_xet is not installed/i.test(serverReason)
      ? t("settings.general.downloads.xetMissing")
      : serverReason;

  // Xet selected but unable to run here: say so rather than silently using HTTPS. Only once
  // known, since while pending we have not been told hf_xet is missing.
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

  // The health check's own words, on their own line: free-form English, so folding it into a
  // translated sentence produced half a sentence in each language. Auto is the only branch
  // with a reason to explain.
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
                    // Indexed for settings search, so the result has somewhere to scroll to.
                    data-settings-label={t(opt.labelKey)}
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
