// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { usePlatformStore } from "@/config/env";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type DeveloperModeSettings,
  loadDeveloperModeSettings,
  updateDeveloperModeSettings,
} from "../api/developer-mode";
import {
  videoCpuRamCheck,
  videoCpuToggleNeedsConfirm,
} from "../lib/video-cpu-warning";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/** Owner-only. Developer mode's one switch: experimental video generation on a CPU-only host.
 *  Enabling asks first, with the host's RAM against the minimum; disabling restores the GPU gate. */
export function DeveloperModeSection() {
  const t = useT();
  const [settings, setSettings] = useState<DeveloperModeSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadDeveloperModeSettings()
      .then((next) => {
        if (cancelled) return;
        setSettings(next);
        setError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof Error
            ? err.message
            : t("settings.general.developerMode.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const save = async (videoCpu: boolean) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateDeveloperModeSettings(videoCpu);
      setSettings(next);
      // Both gates are cached: the sidebar's Video row reads the /api/health verdict and the Video
      // page /api/system/hardware. Update the one and re-read the other, so neither needs a reload.
      usePlatformStore.setState({ videoCpuOptIn: next.videoCpu });
      void refreshHardwareInfo();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("settings.general.developerMode.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const onToggle = (next: boolean) => {
    if (videoCpuToggleNeedsConfirm(next)) {
      setConfirmOpen(true);
      return;
    }
    void save(false);
  };

  const ram = settings
    ? videoCpuRamCheck(settings.hostRamBytes, settings.minRamBytes)
    : null;
  const envForced = settings?.videoCpuEnvOverride ?? null;

  let note: string | null = null;
  if (error) note = error;
  else if (envForced !== null)
    note = t(
      envForced
        ? "settings.general.developerMode.forcedOnByEnv"
        : "settings.general.developerMode.forcedOffByEnv",
    );
  else if (settings && !settings.videoCpuApplicable)
    note = t("settings.general.developerMode.notApplicable");

  return (
    <>
      <SettingsSection title={t("settings.general.developerMode.sectionTitle")}>
        <SettingsRow
          label={t("settings.general.developerMode.videoCpu")}
          description={
            <>
              {t("settings.general.developerMode.videoCpuDescription")}
              {note ? (
                <span
                  data-testid="developer-mode-note"
                  className={cn(
                    "mt-1.5 block",
                    error ? "text-destructive" : "text-muted-foreground/80",
                  )}
                >
                  {note}
                </span>
              ) : null}
            </>
          }
        >
          <Switch
            data-testid="developer-mode-video-cpu"
            checked={settings?.videoCpu ?? false}
            disabled={!settings || saving || envForced !== null}
            onCheckedChange={onToggle}
          />
        </SettingsRow>
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md" data-testid="developer-mode-confirm">
          <DialogHeader>
            <DialogTitle>
              {t("settings.general.developerMode.confirmTitle")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.general.developerMode.confirmWarning")}
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-2 text-sm text-muted-foreground">
            {ram?.status === "below" ? (
              <p
                data-testid="developer-mode-ram-warning"
                className="text-amber-600 dark:text-amber-400"
              >
                {t("settings.general.developerMode.ramBelowMinimum", {
                  host: String(ram.hostGib),
                  min: String(ram.minGib),
                })}
              </p>
            ) : ram?.status === "ok" ? (
              <p>
                {t("settings.general.developerMode.ramOk", {
                  host: String(ram.hostGib),
                  min: String(ram.minGib),
                })}
              </p>
            ) : ram ? (
              <p>
                {t("settings.general.developerMode.ramUnknown", {
                  min: String(ram.minGib),
                })}
              </p>
            ) : null}
            <p>{t("settings.general.developerMode.performanceEstimate")}</p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              disabled={saving}
              onClick={() => {
                setConfirmOpen(false);
                void save(true);
              }}
              variant="destructive"
            >
              {t("settings.general.developerMode.confirmAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
