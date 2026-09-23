// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  DOWNLOAD_KIND,
  downloadManager,
  formatBytes,
  jobKeyOf,
  scopedVariant,
  useDownloadManagerStore,
} from "@/features/hub";
import { type TranslationKey, translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { TaskDone01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import {
  type SystemOneDevice,
  type SystemOneDownloadPlan,
  type SystemOneSettings,
  loadSystemOneSettings,
  resolveSystemOneDownload,
  unloadSystemOneModel,
  updateSystemOneSettings,
} from "../api/systemone";
import { SettingsRow } from "./settings-row";

/** One download slot for every Laya checkpoint, so switching models while one downloads reports busy rather than racing. */
const DOWNLOAD_SCOPE = "systemone";
const POLL_MS = 5000;
const RECOMMENDED_MODEL = "laya-multilingual";
const MODEL_LABELS: Record<string, TranslationKey> = {
  "laya-multilingual": "settings.apiKeys.decisionApi.modelMultilingual",
  "laya-english": "settings.apiKeys.decisionApi.modelEnglish",
  "laya-typed-decisions": "settings.apiKeys.decisionApi.modelTypedDecisions",
};
const ENV_DISABLE = "UNSLOTH_SYSTEMONE_DISABLE";
const ENV_MODEL = "UNSLOTH_SYSTEMONE_MODEL";
const ENV_DEVICE = "UNSLOTH_SYSTEMONE_DEVICE";

function deviceLabel(device: string | null): string {
  return device && device !== "cpu" ? "GPU" : "CPU";
}

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

export function DecisionApiSection(): ReactElement | null {
  const t = useT();
  const [settings, setSettings] = useState<SystemOneSettings | null>(null);
  const [planState, setPlanState] = useState<{
    model: string;
    plan: SystemOneDownloadPlan;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enabled = settings?.enabled ?? false;
  const model = settings?.model ?? null;
  // Keyed by model so a switch shows "Checking" until its own plan lands, never the previous model's.
  const plan = planState && planState.model === model ? planState.plan : null;

  useEffect(() => {
    let live = true;
    loadSystemOneSettings().then(
      (next) => live && setSettings(next),
      (err) =>
        live &&
        setError(
          errorMessage(err) ??
            translate("settings.apiKeys.decisionApi.loadError"),
        ),
    );
    return () => {
      live = false;
    };
  }, []);

  // Residency changes on API traffic this dialog never sees, so poll while the feature is on and the tab is visible.
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      loadSystemOneSettings().then(setSettings, () => undefined);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [enabled]);

  const jobKey =
    plan?.repo && !plan.cached && plan.files.length > 0
      ? jobKeyOf(DOWNLOAD_KIND.MODEL, plan.repo, scopedVariant(DOWNLOAD_SCOPE))
      : null;
  const jobState = useDownloadManagerStore((s) =>
    jobKey ? (s.jobs[jobKey]?.state ?? null) : null,
  );
  const downloading = jobState === "running" || jobState === "cancelling";
  const downloadDone = jobState === "complete";

  useEffect(() => {
    if (!enabled || !model) return;
    let live = true;
    resolveSystemOneDownload().then(
      (next) => live && setPlanState({ model, plan: next }),
      (err) => live && setError(errorMessage(err)),
    );
    return () => {
      live = false;
    };
  }, [enabled, model, downloadDone]);

  const modelLabel = (name: string) =>
    MODEL_LABELS[name] ? t(MODEL_LABELS[name]) : name;

  const startDownload = async (next: SystemOneDownloadPlan) => {
    if (!next.repo || next.cached || next.files.length === 0) return;
    try {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: next.repo,
        variant: scopedVariant(DOWNLOAD_SCOPE),
        scopeId: DOWNLOAD_SCOPE,
        files: next.files,
        inventoryKind: "model",
        expectedBytes: next.sizeBytes,
      });
      // The download manager shows its own panel and progress, so a start needs no toast of its own.
      if (outcome === "started") return;
      if (outcome === "conflict" || outcome === "busy") {
        toast.info(t("settings.apiKeys.decisionApi.downloadBusy"));
      } else {
        toast.error(t("settings.apiKeys.decisionApi.downloadFailed"));
      }
    } catch (err) {
      toast.error(t("settings.apiKeys.decisionApi.downloadFailed"), {
        description: errorMessage(err) ?? undefined,
      });
    }
  };

  const apply = async (
    patch: { enabled?: boolean; model?: string; device?: SystemOneDevice },
    downloadAfter: boolean,
  ) => {
    setBusy(true);
    setError(null);
    try {
      const next = await updateSystemOneSettings(patch);
      setSettings(next);
      // Download on the switch, not on the first request: a first API call should not sit behind a 700 MB transfer.
      if (next.enabled && downloadAfter) {
        const nextPlan = await resolveSystemOneDownload();
        setPlanState({ model: next.model, plan: nextPlan });
        await startDownload(nextPlan);
      }
    } catch (err) {
      setError(
        errorMessage(err) ?? t("settings.apiKeys.decisionApi.saveFailed"),
      );
    } finally {
      setBusy(false);
    }
  };

  const unload = async () => {
    setBusy(true);
    setError(null);
    try {
      setSettings(await unloadSystemOneModel());
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const header = (
    <>
      <div className="flex items-start gap-3 bg-muted/30 p-4">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
          <HugeiconsIcon
            icon={TaskDone01Icon}
            className="size-4 text-foreground"
          />
        </div>
        <div className="flex min-w-0 flex-col gap-0.5">
          <h2 className="text-base font-semibold font-heading text-foreground">
            {t("settings.apiKeys.decisionApi.title")}
          </h2>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("settings.apiKeys.decisionApi.description")}
          </p>
        </div>
      </div>

      {error ? (
        <p className="border-t border-border/60 px-4 py-2.5 text-xs leading-snug text-destructive">
          {error}
        </p>
      ) : null}
    </>
  );

  // A failed first load still shows the section, so the owner sees why its controls are missing.
  if (!settings) {
    return error ? (
      <section
        data-settings-label={t("settings.apiKeys.decisionApi.title")}
        className="overflow-hidden rounded-lg border border-border/70"
      >
        {header}
      </section>
    ) : null;
  }

  const current = settings.models.find((m) => m.name === settings.model);
  const knownModel = current !== undefined;
  const sizeBytes = plan?.sizeBytes || current?.downloadBytes || 0;

  let tone: "pending" | "ready" | "error" | null = null;
  let status = "";
  let action: "download" | "unload" | null = null;
  if (settings.error) {
    tone = "error";
    status = settings.error;
  } else if (settings.installing) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.installing");
  } else if (downloading) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.downloading");
  } else if (settings.loadingModel) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.loading");
  } else if (settings.loadedModel === settings.model) {
    tone = "ready";
    status = t("settings.apiKeys.decisionApi.loadedOn", {
      device: deviceLabel(settings.loadedDevice),
    });
    action = "unload";
  } else if (plan === null) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.checking");
  } else if (!plan.cached && plan.error) {
    tone = "error";
    status = plan.error;
  } else if (!plan.cached) {
    status = t("settings.apiKeys.decisionApi.notDownloaded", {
      size: formatBytes(sizeBytes),
    });
    action = "download";
  } else {
    tone = "ready";
    status = t("settings.apiKeys.decisionApi.downloaded");
  }

  return (
    <section
      data-settings-label={t("settings.apiKeys.decisionApi.title")}
      className="overflow-hidden rounded-lg border border-border/70"
    >
      {header}

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label={t("settings.apiKeys.decisionApi.enable")}
          description={
            settings.enabledLocked
              ? t("settings.apiKeys.decisionApi.lockedByEnv", {
                  name: ENV_DISABLE,
                })
              : t("settings.apiKeys.decisionApi.enableDescription")
          }
          alignTop={true}
        >
          <Switch
            checked={enabled}
            disabled={busy || settings.enabledLocked}
            onCheckedChange={(on) => void apply({ enabled: on }, on)}
            aria-label={t("settings.apiKeys.decisionApi.enable")}
          />
        </SettingsRow>

        <SettingsRow
          label={t("settings.apiKeys.decisionApi.model")}
          hint={
            settings.modelLocked
              ? t("settings.apiKeys.decisionApi.lockedByEnv", {
                  name: ENV_MODEL,
                })
              : undefined
          }
          description={
            enabled && status ? (
              <span
                className={cn(
                  "flex min-w-0 items-center gap-2",
                  tone === "error" && "text-destructive",
                )}
              >
                {tone ? (
                  <span
                    className={cn(
                      "size-1.5 shrink-0 rounded-full",
                      tone === "pending"
                        ? "animate-pulse bg-current"
                        : tone === "ready"
                          ? "bg-emerald-500"
                          : "bg-destructive",
                    )}
                  />
                ) : null}
                <span className="truncate" title={status}>
                  {status}
                </span>
              </span>
            ) : undefined
          }
          className="max-[420px]:flex-col max-[420px]:items-stretch max-[420px]:gap-3"
        >
          <div className="flex items-center gap-2 max-[420px]:w-full">
            {action === "download" ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 shrink-0 px-2.5 text-xs"
                disabled={busy || downloading}
                onClick={() => plan && void startDownload(plan)}
              >
                {downloading ? <Spinner className="mr-1.5" /> : null}
                {t("settings.apiKeys.decisionApi.download")}
              </Button>
            ) : action === "unload" ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 shrink-0 px-2.5 text-xs"
                disabled={busy}
                onClick={() => void unload()}
              >
                {t("settings.apiKeys.decisionApi.unload")}
              </Button>
            ) : null}
            {knownModel ? (
              <Select
                value={settings.model}
                disabled={busy || settings.modelLocked}
                onValueChange={(name) => void apply({ model: name }, true)}
              >
                <SelectTrigger
                  className="w-48 max-[420px]:flex-1"
                  aria-label={t("settings.apiKeys.decisionApi.model")}
                >
                  <SelectValue>{modelLabel(settings.model)}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {settings.models.map((option) => (
                    <SelectItem key={option.name} value={option.name}>
                      <span className="flex items-center gap-2">
                        {modelLabel(option.name)}
                        <span className="text-ui-10 tabular-nums text-muted-foreground">
                          {formatBytes(option.downloadBytes)}
                        </span>
                        {option.name === RECOMMENDED_MODEL ? (
                          <span className="rounded-full bg-emerald-500/12 px-1.5 py-px text-ui-9 font-medium text-emerald-600 dark:text-emerald-400">
                            {t("settings.apiKeys.decisionApi.recommended")}
                          </span>
                        ) : null}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <span className="font-mono text-xs text-foreground">
                {settings.model}
              </span>
            )}
          </div>
        </SettingsRow>

        <SettingsRow
          label={t("settings.apiKeys.decisionApi.device")}
          description={
            settings.deviceLocked
              ? t("settings.apiKeys.decisionApi.lockedByEnv", {
                  name: ENV_DEVICE,
                })
              : t("settings.apiKeys.decisionApi.deviceDescription")
          }
          alignTop={true}
        >
          <Select
            value={settings.device}
            disabled={busy || settings.deviceLocked}
            onValueChange={(device) =>
              void apply({ device: device as SystemOneDevice }, false)
            }
          >
            <SelectTrigger
              className="w-36"
              aria-label={t("settings.apiKeys.decisionApi.device")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cpu">
                {t("settings.apiKeys.decisionApi.deviceCpu")}
              </SelectItem>
              <SelectItem value="gpu" disabled={!settings.gpuAvailable}>
                {t("settings.apiKeys.decisionApi.deviceGpu")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
      </div>
    </section>
  );
}
