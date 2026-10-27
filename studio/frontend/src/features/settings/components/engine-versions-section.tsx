// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useIsAccountOwner } from "@/features/auth";
import { resyncInferenceStatusAfterServerModelChange } from "@/features/chat";
import { useHardwareInfo } from "@/hooks/use-hardware-info";
import {
  type ComponentOffer,
  llamaUpdateOffered,
  useLlamaUpdateCheck,
} from "@/hooks/use-llama-update-check";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

type EngineRow = {
  key: "llama" | "whisper" | "audio";
  labelKey: TranslationKey;
  installed: string | null;
  offer: ComponentOffer | null;
};

export function EngineVersionsSection() {
  const t = useT();
  const hw = useHardwareInfo();
  const isOwner = useIsAccountOwner();
  const { status, applying, apply, refresh } = useLlamaUpdateCheck({
    onReloadRequired: resyncInferenceStatusAfterServerModelChange,
    passive: true,
  });
  const [checking, setChecking] = useState(false);
  const [checkFailed, setCheckFailed] = useState(false);

  // The binary's own build string names the variant the release tag leaves out.
  const rows: EngineRow[] = [
    {
      key: "llama",
      labelKey: "settings.about.llamaCppVersion",
      installed: hw.llamaCpp ?? status?.llama.installed_tag ?? null,
      offer: status?.llama ?? null,
    },
    {
      key: "whisper",
      labelKey: "settings.about.whisperCppVersion",
      installed: status?.whisper?.installed_tag ?? null,
      offer: status?.whisper ?? null,
    },
    {
      key: "audio",
      labelKey: "settings.about.audioCppVersion",
      installed: hw.audioCpp ?? status?.audio?.installed_tag ?? null,
      offer: status?.audio ?? null,
    },
  ];
  // whisper.cpp only reports once the backend knows about it; an empty row would read as missing.
  const visibleRows = rows.filter(
    (row) => row.key !== "whisper" || row.offer != null,
  );
  const job = status?.job;
  const offered = status != null && llamaUpdateOffered(status);
  // `supported` is llama.cpp's alone: an audio.cpp update still installs beside a self-managed llama.cpp.
  const unmanaged = status?.supported === false && !offered;

  async function handleCheck() {
    setChecking(true);
    const next = await refresh();
    setChecking(false);
    setCheckFailed(next == null);
  }

  async function handleUpdate() {
    const result = await apply();
    if (result.ok) {
      toast.success(t("settings.about.engines.updated"));
    } else {
      toast.error(
        t("settings.about.engines.updateFailed", {
          error: result.error ?? "",
        }),
      );
    }
  }

  const summary = applying
    ? job?.message || t("settings.about.engines.updating")
    : checking
      ? t("settings.about.engines.checking")
      : checkFailed
        ? t("settings.about.engines.checkFailed")
        : unmanaged
          ? t("settings.about.engines.unsupported")
          : offered
            ? t("settings.about.engines.updatesAvailable")
            : status
              ? t("settings.about.engines.upToDate")
              : t("settings.about.engines.checking");

  return (
    <SettingsSection
      title={t("settings.about.engines.title")}
      description={t("settings.about.engines.description")}
    >
      {visibleRows.map((row) => {
        const latest =
          row.offer?.update_available && row.offer.latest_tag
            ? row.offer.latest_tag
            : null;
        return (
          <SettingsRow
            key={row.key}
            label={t(row.labelKey)}
            below={
              latest ? (
                <span
                  className="text-xs font-medium text-primary"
                  data-testid={`engine-${row.key}-available`}
                >
                  {t("settings.about.engines.available", { version: latest })}
                </span>
              ) : null
            }
          >
            <code
              className="font-mono text-xs text-muted-foreground"
              data-testid={`engine-${row.key}-version`}
            >
              {row.installed ?? t("settings.about.engines.notInstalled")}
            </code>
          </SettingsRow>
        );
      })}

      <SettingsRow
        label={t("settings.about.engines.updatesLabel")}
        description={summary}
      >
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={checking || applying}
            onClick={() => void handleCheck()}
            data-testid="engine-check-updates"
          >
            {t("settings.about.engines.checkForUpdates")}
          </Button>
          {/* POST /api/llama/update is owner-only; a managed account can read versions, not install. */}
          {isOwner ? (
            <Button
              size="sm"
              disabled={!offered || applying || checking}
              onClick={() => void handleUpdate()}
              data-testid="engine-update"
            >
              {applying
                ? t("settings.about.engines.updating")
                : t("settings.about.engines.update")}
            </Button>
          ) : null}
        </div>
      </SettingsRow>

      {applying ? (
        <div className="pb-3" data-testid="engine-update-progress">
          <Progress
            value={Math.round((job?.progress ?? 0) * 100)}
            aria-label={t("settings.about.engines.updating")}
            className="h-1.5 w-full rounded-full bg-muted"
          />
        </div>
      ) : null}
    </SettingsSection>
  );
}
