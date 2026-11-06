// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { useEffect, useRef, useState } from "react";
import {
  loadDownloadSpeedLimitSettings,
  updateDownloadSpeedLimitSettings,
} from "../api/download-speed-limit";
import { SettingsRow } from "./settings-row";

function asDraft(mbps: number | null) {
  return mbps === null ? "" : String(mbps);
}

export function DownloadSpeedLimitRow() {
  const t = useT();
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const edited = useRef(false);

  useEffect(() => {
    let cancelled = false;
    loadDownloadSpeedLimitSettings()
      .then((settings) => {
        if (!(cancelled || edited.current)) {
          setDraft(asDraft(settings.mbps));
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const save = async () => {
    const trimmed = draft.trim();
    const mbps = trimmed === "" ? null : Number(trimmed);
    if (mbps !== null && (!Number.isFinite(mbps) || mbps < 0)) {
      setError(t("settings.general.downloads.speedLimitInvalid"));
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      const settings = await updateDownloadSpeedLimitSettings(
        mbps === 0 ? null : mbps,
      );
      setDraft(asDraft(settings.mbps));
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.downloads.speedLimitInvalid"),
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <SettingsRow
      label={t("settings.general.downloads.speedLimit")}
      description={t("settings.general.downloads.speedLimitDescription")}
      below={
        error ? (
          <span className="max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-destructive">
            {error}
          </span>
        ) : null
      }
    >
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5">
          <Input
            type="number"
            min={0}
            step={1}
            value={draft}
            placeholder={t("settings.general.downloads.speedLimitNoLimit")}
            aria-label={t("settings.general.downloads.speedLimit")}
            onChange={(event) => {
              edited.current = true;
              setDraft(event.target.value);
            }}
            className="h-control w-24"
          />
          <span className="text-xs font-medium text-muted-foreground">
            Mbit/s
          </span>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={isSaving}
          onClick={() => void save()}
        >
          {isSaving ? t("common.saving") : t("common.save")}
        </Button>
      </div>
    </SettingsRow>
  );
}
