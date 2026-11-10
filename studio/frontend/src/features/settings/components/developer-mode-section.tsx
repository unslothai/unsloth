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
import { bootstrapPersistedCredentials } from "@/features/credentials/bootstrap";
import { refreshHardwareInfo } from "@/hooks/use-hardware-info";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  type DeveloperModePatch,
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

  const save = async (patch: DeveloperModePatch) => {
    setSaving(true);
    setError(null);
    try {
      const next = await updateDeveloperModeSettings(patch);
      setSettings(next);
      if (patch.videoCpu !== undefined) {
        // Sidebar reads /api/health, Video page /api/system/hardware: update one, re-read the other.
        usePlatformStore.setState({ videoCpuOptIn: next.videoCpu });
        void refreshHardwareInfo();
      }
      if (patch.claudeAgent !== undefined) void bootstrapPersistedCredentials();
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
    void save({ videoCpu: false });
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
  const claudeAgentForced = settings?.claudeAgentEnvOverride ?? null;

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
        <SettingsRow
          label={t("settings.general.developerMode.claudeAgent")}
          description={
            <>
              {t("settings.general.developerMode.claudeAgentDescription")}
              {claudeAgentForced !== null ? (
                <span
                  data-testid="developer-mode-claude-agent-note"
                  className="mt-1.5 block text-muted-foreground/80"
                >
                  {t(
                    claudeAgentForced
                      ? "settings.general.developerMode.claudeAgentForcedOnByEnv"
                      : "settings.general.developerMode.claudeAgentForcedOffByEnv",
                  )}
                </span>
              ) : null}
            </>
          }
        >
          <Switch
            data-testid="developer-mode-claude-agent"
            aria-label={t("settings.general.developerMode.claudeAgent")}
            checked={settings?.claudeAgent ?? false}
            disabled={!settings || saving || claudeAgentForced !== null}
            onCheckedChange={(next) => void save({ claudeAgent: next })}
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
                void save({ videoCpu: true });
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
