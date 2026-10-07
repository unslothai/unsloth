// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
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
import { translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { TaskDone01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import {
  type SystemOneBackend,
  type SystemOneConnection,
  type SystemOneDevice,
  type SystemOneDownloadPlan,
  type SystemOneSettings,
  loadSystemOneConnections,
  loadSystemOneSettings,
  resolveSystemOneDownload,
  unloadSystemOneModel,
  updateSystemOneSettings,
  validateSystemOneSettings,
} from "../api/systemone";
import {
  DECISION_MODEL_LABELS,
  isClefDecisionModel,
} from "../lib/decision-model-labels";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";

const DOWNLOAD_SCOPE = "systemone";
const POLL_MS = 5000;
const RECOMMENDED_MODEL = "laya-multilingual";
const ENV_DISABLE = "UNSLOTH_SYSTEMONE_DISABLE";
const ENV_MODEL = "UNSLOTH_SYSTEMONE_MODEL";
const ENV_DEVICE = "UNSLOTH_SYSTEMONE_DEVICE";

function isClefModel(name: string | undefined): boolean {
  return [
    "clef",
    "clef-flash",
    "Cloudflare/clef",
    "Cloudflare/clef-flash",
  ].includes(name ?? "");
}

function deviceLabel(device: string | null): string {
  return device && device !== "cpu" ? "GPU" : "CPU";
}

function errorMessage(error: unknown): string | null {
  return error instanceof Error ? error.message : null;
}

export function DecisionApiSection(): ReactElement | null {
  const t = useT();
  const [settings, setSettings] = useState<SystemOneSettings | null>(null);
  const [connections, setConnections] = useState<SystemOneConnection[] | null>(
    null,
  );
  const [planState, setPlanState] = useState<{
    model: string;
    backend: SystemOneBackend;
    plan: SystemOneDownloadPlan;
  } | null>(null);
  const [confirm, setConfirm] = useState<{
    plan: SystemOneDownloadPlan;
    patch: Parameters<typeof updateSystemOneSettings>[0];
    model: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);

  const enabled = settings?.enabled ?? false;
  const model = settings?.model ?? null;
  const backend = settings?.backend ?? "auto";
  const plan =
    planState?.model === model && planState?.backend === backend
      ? planState.plan
      : null;

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
    loadSystemOneConnections().then(
      (next) => live && setConnections(next),
      () => live && setConnections([]),
    );
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (scrollTarget !== "api-keys-decision-api" || !settings) return;
    const frame = window.requestAnimationFrame(() => {
      sectionRef.current?.scrollIntoView({
        block: "start",
        behavior: "smooth",
      });
      useSettingsDialogStore
        .getState()
        .consumeScrollTarget("api-keys-decision-api");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [scrollTarget, settings]);

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
    resolveSystemOneDownload(model, backend).then(
      (next) => live && setPlanState({ model, backend, plan: next }),
      (err) => live && setError(errorMessage(err)),
    );
    return () => {
      live = false;
    };
  }, [enabled, model, backend, downloadDone]);

  const modelLabel = (name: string) => {
    const connection = connections?.find((c) => c.name === name);
    if (connection) return `${connection.provider} · ${connection.model}`;
    const option = settings?.models.find((m) => m.name === name);
    if (option?.kind === "fine_tune" && option.label) return option.label;
    return DECISION_MODEL_LABELS[name] ? t(DECISION_MODEL_LABELS[name]) : name;
  };

  const resyncSettingsAfterError = async (message: string) => {
    try {
      setSettings(await loadSystemOneSettings());
    } catch (refreshError) {
      console.warn(
        "Couldn't refresh Decision API settings after a rejected change.",
        refreshError,
      );
    }
    setError(message);
  };

  const startDownload = async (next: SystemOneDownloadPlan) => {
    if (!next.repo || next.cached || next.files.length === 0) return false;
    const downloadKey = jobKeyOf(
      DOWNLOAD_KIND.MODEL,
      next.repo,
      scopedVariant(DOWNLOAD_SCOPE),
    );
    try {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId: next.repo,
        revision: next.revision ?? undefined,
        variant: scopedVariant(DOWNLOAD_SCOPE),
        scopeId: DOWNLOAD_SCOPE,
        files: next.files,
        inventoryKind: "model",
        expectedBytes: next.sizeBytes,
      });
      if (outcome === "started") {
        const acceptedState =
          useDownloadManagerStore.getState().jobs[downloadKey]?.state;
        if (acceptedState === "running" || acceptedState === "complete") {
          return true;
        }
      }
      if (outcome === "conflict" || outcome === "busy") {
        toast.info(t("settings.apiKeys.decisionApi.downloadBusy"));
      } else if (outcome === "started") {
        toast.info(t("settings.apiKeys.decisionApi.downloadBusy"));
      } else {
        toast.error(t("settings.apiKeys.decisionApi.downloadFailed"));
      }
    } catch (err) {
      toast.error(t("settings.apiKeys.decisionApi.downloadFailed"), {
        description: errorMessage(err) ?? undefined,
      });
    }
    return false;
  };

  const apply = async (
    patch: Parameters<typeof updateSystemOneSettings>[0],
    downloadAfter: boolean,
  ) => {
    setBusy(true);
    setError(null);
    try {
      const nextEnabled = patch.enabled ?? settings?.enabled;
      const nextModel = patch.model ?? settings?.model;
      const nextBackend = patch.backend ?? backend;
      const settingsPatch =
        downloadAfter && settings
          ? {
              ...patch,
              expectedEnabled: settings.enabled,
              expectedModel: settings.model,
              expectedBackend: settings.backend,
            }
          : patch;
      let resolvedPlan: typeof planState = null;
      if (nextEnabled && nextModel && downloadAfter) {
        const nextPlan = await resolveSystemOneDownload(nextModel, nextBackend);
        if (!nextPlan.cached) {
          if (nextPlan.error || !nextPlan.repo || nextPlan.files.length === 0) {
            throw new Error(
              nextPlan.error ??
                t("settings.apiKeys.decisionApi.downloadFailed"),
            );
          }
          setConfirm({
            plan: nextPlan,
            patch: settingsPatch,
            model: nextModel,
          });
          return;
        }
        resolvedPlan = {
          model: nextModel,
          backend: nextBackend,
          plan: nextPlan,
        };
      }
      const next = await updateSystemOneSettings(settingsPatch);
      setSettings(next);
      if (resolvedPlan?.model === next.model) setPlanState(resolvedPlan);
    } catch (err) {
      await resyncSettingsAfterError(
        errorMessage(err) ?? t("settings.apiKeys.decisionApi.saveFailed"),
      );
    } finally {
      setBusy(false);
    }
  };

  const acceptDownload = async (accepted: NonNullable<typeof confirm>) => {
    setBusy(true);
    setError(null);
    try {
      await validateSystemOneSettings(accepted.patch);
      if (!(await startDownload(accepted.plan))) return;
      const next = await updateSystemOneSettings(accepted.patch);
      setSettings(next);
      if (next.model === accepted.model) {
        setPlanState({
          model: accepted.model,
          backend: next.backend,
          plan: accepted.plan,
        });
      }
    } catch (err) {
      await resyncSettingsAfterError(
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

  const isClef = isClefModel(settings?.model);
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
          <h2 className="settings-heading text-base font-semibold font-heading">
            {t("settings.apiKeys.decisionApi.title")}
          </h2>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t(
              isClef
                ? "settings.apiKeys.decisionApi.descriptionClef"
                : "settings.apiKeys.decisionApi.description",
            )}
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
  const isRemote = settings.model.startsWith("connection:");
  const remote = connections?.find((c) => c.name === settings.model);
  const knownModel = current !== undefined || isRemote;
  const longLabel = isRemote || current?.kind === "fine_tune";
  const connectionGroups = [
    ...new Set(connections?.map((c) => c.providerId)),
  ].map((id) => connections?.filter((c) => c.providerId === id) ?? []);
  const sizeBytes = plan?.sizeBytes || current?.downloadBytes || 0;

  let tone: "pending" | "ready" | "error" | null = null;
  let status = "";
  let action: "download" | "unload" | null = null;
  if (remote) {
    status = t("settings.apiKeys.decisionApi.sendsTo", {
      provider: remote.provider,
    });
  } else if (isRemote) {
    tone = connections ? "error" : "pending";
    status = connections
      ? t("settings.apiKeys.decisionApi.connectionMissing")
      : t("settings.apiKeys.decisionApi.checking");
  } else if (settings.error) {
    tone = "error";
    status = settings.error;
  } else if (settings.installing) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.installing");
  } else if (downloading) {
    tone = "pending";
    status = t("settings.apiKeys.decisionApi.downloading");
  } else if (settings.loadingModel === settings.model) {
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
    status =
      current?.kind === "fine_tune"
        ? t("settings.apiKeys.decisionApi.ready")
        : t("settings.apiKeys.decisionApi.downloaded");
  }

  return (
    <section
      ref={sectionRef}
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
              : isRemote
                ? t("settings.apiKeys.decisionApi.enableRemoteDescription")
                : t("settings.apiKeys.decisionApi.enableDescription")
          }
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
            (enabled || isRemote) && status ? (
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
                  className={cn(
                    longLabel ? "w-64" : "w-48",
                    "max-[420px]:flex-1",
                  )}
                  aria-label={t("settings.apiKeys.decisionApi.model")}
                  title={longLabel ? modelLabel(settings.model) : undefined}
                >
                  <SelectValue className="min-w-0">
                    <span className="truncate">
                      {modelLabel(settings.model)}
                    </span>
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>
                      {t("settings.apiKeys.decisionApi.thisMachine")}
                    </SelectLabel>
                    {settings.models.map((option) => (
                      <SelectItem
                        key={option.name}
                        value={option.name}
                        disabled={!option.available}
                        title={option.unavailableReason ?? undefined}
                      >
                        <span className="flex items-center gap-2">
                          {modelLabel(option.name)}
                          {option.kind === "fine_tune" ? null : (
                            <span className="text-ui-10 tabular-nums text-muted-foreground">
                              {formatBytes(option.downloadBytes)}
                            </span>
                          )}
                          {option.name === RECOMMENDED_MODEL ? (
                            <span className="rounded-full bg-emerald-500/12 px-1.5 py-px text-ui-9 font-medium text-emerald-600 dark:text-emerald-400">
                              {t("settings.apiKeys.decisionApi.recommended")}
                            </span>
                          ) : null}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                  {connectionGroups.map((group) => (
                    <SelectGroup key={group[0].providerId}>
                      <SelectLabel>{group[0].provider}</SelectLabel>
                      {group.map((option) => (
                        <SelectItem key={option.name} value={option.name}>
                          {option.model}
                        </SelectItem>
                      ))}
                    </SelectGroup>
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

        {current ? (
          <p className="pb-3 text-xs text-muted-foreground leading-relaxed">
            {current.description}
          </p>
        ) : null}

        {isClef ? (
          <>
            <SettingsRow
              label={t("settings.apiKeys.decisionApi.backend")}
              description={t("settings.apiKeys.decisionApi.backendDescription")}
            >
              <Select
                value={backend}
                disabled={busy}
                onValueChange={(value) =>
                  void apply({ backend: value as SystemOneBackend }, true)
                }
              >
                <SelectTrigger
                  className="w-36"
                  aria-label={t("settings.apiKeys.decisionApi.backend")}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">
                    {t("settings.apiKeys.decisionApi.backendAuto")}
                  </SelectItem>
                  <SelectItem value="llama.cpp">llama.cpp</SelectItem>
                  <SelectItem value="pytorch">PyTorch</SelectItem>
                </SelectContent>
              </Select>
            </SettingsRow>
            <p
              className="pb-3 text-xs text-muted-foreground leading-relaxed"
              data-decision-backend
            >
              {t("settings.apiKeys.decisionApi.backendStatus", {
                backend:
                  settings.loadedModel === settings.model
                    ? (settings.loadedBackend ?? "—")
                    : (settings.effectiveBackend ?? "—"),
              })}
              {settings.fallbackReason ? ` · ${settings.fallbackReason}` : ""}{" "}
              {t(
                settings.inputModalities.includes("image")
                  ? "settings.apiKeys.decisionApi.mediaImages"
                  : "settings.apiKeys.decisionApi.mediaText",
              )}
            </p>
          </>
        ) : null}

        {isRemote ? null : (
          <SettingsRow
            label={t("settings.apiKeys.decisionApi.device")}
            description={
              settings.deviceLocked
                ? t("settings.apiKeys.decisionApi.lockedByEnv", {
                    name: ENV_DEVICE,
                  })
                : t(
                    isClef
                      ? "settings.apiKeys.decisionApi.clefDeviceDescription"
                      : "settings.apiKeys.decisionApi.deviceDescription",
                  )
            }
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
        )}

        {connections?.length === 0 && !settings.modelLocked ? (
          <p className="pb-3 text-xs text-muted-foreground">
            {t("settings.apiKeys.decisionApi.addConnection")}{" "}
            <button
              type="button"
              className="font-medium text-foreground underline underline-offset-2 hover:text-primary"
              onClick={() =>
                useSettingsDialogStore.getState().setActiveTab("connections")
              }
            >
              {t("settings.apiKeys.decisionApi.openConnections")}
            </button>
          </p>
        ) : null}
      </div>

      <AlertDialog
        open={confirm !== null}
        onOpenChange={(open) => {
          if (!open && confirm) {
            setConfirm(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia>
              <HugeiconsIcon icon={TaskDone01Icon} strokeWidth={1.75} />
            </AlertDialogMedia>
            <AlertDialogTitle>
              {t(
                isClefDecisionModel(confirm?.model ?? settings.model)
                  ? "settings.apiKeys.decisionApi.downloadConfirmTitleModel"
                  : "settings.apiKeys.decisionApi.downloadConfirmTitle",
                { model: modelLabel(confirm?.model ?? settings.model) },
              )}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.apiKeys.decisionApi.downloadConfirmBody", {
                size: formatBytes(confirm?.plan.sizeBytes || sizeBytes),
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                const accepted = confirm;
                setConfirm(null);
                if (accepted) void acceptDownload(accepted);
              }}
            >
              {t("settings.apiKeys.decisionApi.download")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
