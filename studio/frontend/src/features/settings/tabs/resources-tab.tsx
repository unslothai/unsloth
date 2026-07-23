// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { FolderBrowser } from "@/features/model-picker";
import {
  openModelsDir,
  pickHuggingFaceCacheDir,
} from "@/features/native-intents";
import { useSystemInfo, type GpuDevice } from "@/hooks/use-system";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useT } from "@/i18n";
import { useEffect, useMemo, useState } from "react";
import {
  type HuggingFaceCacheSettings,
  loadHuggingFaceCacheSettings,
  updateHuggingFaceCacheSettings,
} from "../api/hugging-face-cache";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { useMonitorOverlayStore } from "../stores/monitor-overlay-store";
import { CopyIcon, FolderOpenIcon, LayersIcon } from "lucide-react";

const POLL_MS = 3000;

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clampPercent(value: number | null | undefined): number {
  if (!isFiniteNumber(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function usageIndicatorClass(percent: number): string {
  if (percent >= 90) return "bg-destructive";
  if (percent >= 70) return "bg-amber-500";
  return "bg-control-accent";
}

function usageTextClass(percent: number): string {
  if (percent >= 90) return "text-destructive";
  if (percent >= 70) return "text-amber-600 dark:text-amber-400";
  return "text-primary";
}

function formatGb(value: number | null | undefined): string {
  const safe = isFiniteNumber(value) ? Math.max(0, value) : 0;
  const digits = safe >= 10 ? 1 : 2;
  return `${safe.toFixed(digits)} GB`;
}

function formatBytes(value: number | null): string | null {
  if (value === null || !Number.isFinite(value)) return null;
  const gib = value / 1024 ** 3;
  return `${gib >= 10 ? gib.toFixed(1) : gib.toFixed(2)} GiB`;
}

// RAM/VRAM come from the backend in binary units (bytes / 1024**3), matching
// nvidia-smi and PyTorch, so label those readouts GiB. Disk stays on formatGb
// because the backend reports disk in decimal GB (bytes / 1e9).
function formatGiB(value: number | null | undefined): string {
  const safe = isFiniteNumber(value) ? Math.max(0, value) : 0;
  const digits = safe >= 10 ? 1 : 2;
  return `${safe.toFixed(digits)} GiB`;
}

function formatMb(value: number | null | undefined): string {
  const safe = isFiniteNumber(value) ? Math.max(0, value) : 0;
  return `${Math.round(safe).toLocaleString()} MB`;
}

function formatPercent(value: number | null | undefined): string {
  return `${Math.round(clampPercent(value))}%`;
}

function formatFrequency(mhz: number | null | undefined): string | null {
  if (!isFiniteNumber(mhz) || mhz <= 0) return null;
  if (mhz >= 1000) return `${(mhz / 1000).toFixed(2)} GHz`;
  return `${Math.round(mhz)} MHz`;
}

function formatUptime(seconds: number | null | undefined): string {
  if (!isFiniteNumber(seconds) || seconds <= 0) return "0m";
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `${days}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  return `${Math.max(1, minutes)}m`;
}

function MetricTile({
  label,
  value,
  detail,
  percent,
}: {
  label: string;
  value: string;
  detail: string;
  // null = usage unknown (e.g. Windows ROCm perf counter): show a dash and
  // empty bar rather than a fabricated 0%.
  percent: number | null;
}) {
  const percentKnown = isFiniteNumber(percent);
  const safePercent = clampPercent(percent);
  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-md border border-border/60 bg-muted/20 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="truncate text-ui-11 font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {label}
        </span>
        <span
          className={cn(
            "shrink-0 font-mono text-xs tabular-nums",
            percentKnown ? usageTextClass(safePercent) : "text-muted-foreground",
          )}
        >
          {percentKnown ? formatPercent(safePercent) : "--"}
        </span>
      </div>
      <div className="min-w-0">
        <div className="truncate font-mono text-sm tabular-nums text-foreground">
          {value}
        </div>
        <div className="mt-0.5 truncate text-xs text-muted-foreground">
          {detail}
        </div>
      </div>
      <Progress
        value={percentKnown ? safePercent : 0}
        aria-label={label}
        className="h-1.5 rounded-full bg-muted"
        indicatorClassName={usageIndicatorClass(safePercent)}
      />
    </div>
  );
}

function InfoRow({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-4 py-2.5">
      <span className="min-w-0 truncate text-sm font-medium text-foreground">
        {label}
      </span>
      <span
        title={detail ?? value}
        className="min-w-0 max-w-[60%] truncate text-right font-mono text-xs tabular-nums text-muted-foreground"
      >
        {detail ? `${value} (${detail})` : value}
      </span>
    </div>
  );
}

function deviceOrdinal(device: GpuDevice): number | undefined {
  return device.visible_ordinal ?? device.index;
}

export function ResourcesTab() {
  const t = useT();
  const [liveUpdates, setLiveUpdates] = useState(true);
  const { isOpen, setIsOpen } = useMonitorOverlayStore();
  const systemInfo = useSystemInfo({
    enabled: liveUpdates,
    pollMs: liveUpdates ? POLL_MS : undefined,
  });
  const [hfCache, setHfCache] = useState<HuggingFaceCacheSettings | null>(null);
  const [hfCacheLoaded, setHfCacheLoaded] = useState(false);
  const [cacheBrowserOpen, setCacheBrowserOpen] = useState(false);
  const [cacheSaving, setCacheSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadHuggingFaceCacheSettings()
      .then((settings) => {
        if (cancelled) return;
        setHfCache(settings);
        setHfCacheLoaded(true);
      })
      .catch(() => {
        if (cancelled) return;
        setHfCacheLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const metrics = useMemo(() => {
    const devices = systemInfo.gpu?.devices ?? [];
    const ramTotal = systemInfo.memory?.total_gb ?? 0;
    const ramAvailable = systemInfo.memory?.available_gb ?? 0;
    const ramUsed = Math.max(0, ramTotal - ramAvailable);
    const diskTotal = systemInfo.disk?.total_gb ?? 0;
    const diskFree = systemInfo.disk?.free_gb ?? 0;
    const diskUsed = Math.max(0, diskTotal - diskFree);
    const vramTotal = devices.reduce(
      (sum, device) => sum + (device.memory_total_gb ?? 0),
      0,
    );
    // null usage = unknown (e.g. Windows ROCm perf counter): treating it as 0
    // fabricates a 0-used total, so the aggregate is unknown if any device is.
    const vramUsageKnown =
      devices.length > 0 &&
      devices.every((device) => isFiniteNumber(device.vram_used_gb));
    const vramUsed = vramUsageKnown
      ? devices.reduce((sum, device) => sum + (device.vram_used_gb ?? 0), 0)
      : null;
    const vramFree = vramUsageKnown
      ? devices.reduce(
          (sum, device) =>
            sum +
            (device.vram_free_gb ??
              Math.max(
                0,
                (device.memory_total_gb ?? 0) - (device.vram_used_gb ?? 0),
              )),
          0,
        )
      : null;
    const vramPercent =
      vramUsageKnown && isFiniteNumber(vramUsed) && vramTotal > 0
        ? (vramUsed / vramTotal) * 100
        : 0;

    return {
      devices,
      ramTotal,
      ramUsed,
      diskTotal,
      diskFree,
      diskUsed,
      vramTotal,
      vramUsed,
      vramFree,
      vramPercent,
      vramUsageKnown,
    };
  }, [systemInfo]);

  const handleCacheFolder = async () => {
    if (!hfCache) return;
    if (isTauri) {
      try {
        await openModelsDir(hfCache.cacheHome);
      } catch (error) {
        toast.error(t("settings.resources.storage.openError"), {
          description: error instanceof Error ? error.message : undefined,
        });
      }
      return;
    }
    if (await copyToClipboard(hfCache.cacheHome)) {
      toast.success(t("settings.resources.storage.copied"));
    } else {
      toast.error(t("settings.resources.storage.copyError"));
    }
  };

  const saveCacheFolder = async (path: string | null) => {
    setCacheSaving(true);
    try {
      const settings = await updateHuggingFaceCacheSettings(path);
      setHfCache(settings);
      toast.success(t("settings.resources.storage.cacheSaved"));
    } catch (error) {
      toast.error(t("settings.resources.storage.cacheSaveError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setCacheSaving(false);
    }
  };

  const changeCacheFolder = async () => {
    if (!isTauri) {
      setCacheBrowserOpen(true);
      return;
    }
    try {
      const path = await pickHuggingFaceCacheDir();
      if (path) await saveCacheFolder(path);
    } catch (error) {
      toast.error(t("settings.resources.storage.cachePickerError"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const cpuCoresLabel =
    systemInfo.cpu?.logical_count && systemInfo.cpu?.physical_count
      ? t("settings.resources.liveMonitor.cpuCores", {
          logical: systemInfo.cpu.logical_count,
          physical: systemInfo.cpu.physical_count,
        })
      : t("settings.resources.environment.unknown");
  const cpuFrequencyLabel = formatFrequency(systemInfo.cpu?.frequency_mhz);
  const hasGpu =
    (systemInfo.gpu?.available ?? false) && metrics.devices.length > 0;
  const backendLabel = (
    systemInfo.gpu?.backend ?? systemInfo.device_backend ?? "cpu"
  ).toUpperCase();
  const modelsFolderPath = hfCache
    ? hfCache.cacheHome
    : hfCacheLoaded
      ? t("settings.resources.environment.unknown")
      : t("common.loading");
  const cacheLocationDetail = hfCache
    ? hfCache.source === "environment"
      ? t("settings.resources.storage.environmentManaged", {
          variable: hfCache.environmentVariable ?? "HF_HOME",
        })
      : [
          t("settings.resources.storage.futureDownloads"),
          hfCache.freeBytes !== null
            ? t("settings.resources.storage.locationFree", {
                free: formatBytes(hfCache.freeBytes) ?? "",
              })
            : null,
        ]
          .filter(Boolean)
          .join(" · ")
    : null;
  const unknownLabel = t("settings.resources.environment.unknown");

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <h1 className="text-xl font-semibold font-heading">
            {t("settings.resources.title")}
          </h1>
          <p className="text-xs text-muted-foreground">
            {t("settings.resources.description")}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant={isOpen ? "secondary" : "outline"}
            size="sm"
            className="gap-1.5 h-8 text-xs rounded-full px-3"
            onClick={() => setIsOpen(!isOpen)}
          >
            <LayersIcon className="size-3.5" />
            {isOpen
              ? t("settings.resources.disableOverlay")
              : t("settings.resources.floatingWindow")}
          </Button>

          <div className="flex shrink-0 items-center gap-2 rounded-full border border-border/60 px-2.5 py-1.5 text-xs font-medium text-foreground">
            <span>{t("settings.resources.liveUpdates")}</span>
            <Switch
              aria-label={t("settings.resources.liveUpdates")}
              checked={liveUpdates}
              onCheckedChange={setLiveUpdates}
            />
          </div>
        </div>
      </header>

      <SettingsSection title={t("settings.resources.liveMonitor.title")}>
        <div className="grid gap-2 py-3 sm:grid-cols-2">
          <MetricTile
            label={t("settings.resources.liveMonitor.cpu")}
            value={cpuFrequencyLabel ?? cpuCoresLabel}
            detail={
              cpuFrequencyLabel
                ? cpuCoresLabel
                : t("settings.resources.liveMonitor.currentLoad")
            }
            percent={systemInfo.cpu?.usage_percent ?? 0}
          />
          <MetricTile
            label={t("settings.resources.liveMonitor.ram")}
            value={`${formatGiB(metrics.ramUsed)} / ${formatGiB(metrics.ramTotal)}`}
            detail={t("settings.resources.liveMonitor.free", {
              value: formatGiB(systemInfo.memory?.available_gb),
            })}
            percent={systemInfo.memory?.percent_used ?? 0}
          />
          <MetricTile
            label={t("settings.resources.liveMonitor.disk")}
            value={`${formatGb(metrics.diskUsed)} / ${formatGb(metrics.diskTotal)}`}
            detail={t("settings.resources.liveMonitor.free", {
              value: formatGb(metrics.diskFree),
            })}
            percent={systemInfo.disk?.percent_used ?? 0}
          />
          <MetricTile
            label={t("settings.resources.liveMonitor.vram")}
            value={
              hasGpu
                ? metrics.vramUsageKnown
                  ? `${formatGiB(metrics.vramUsed)} / ${formatGiB(metrics.vramTotal)}`
                  : `${unknownLabel} / ${formatGiB(metrics.vramTotal)}`
                : t("settings.resources.liveMonitor.noGpu")
            }
            detail={
              hasGpu
                ? metrics.vramUsageKnown
                  ? t("settings.resources.liveMonitor.free", {
                      value: formatGiB(metrics.vramFree),
                    })
                  : unknownLabel
                : backendLabel
            }
            percent={metrics.vramUsageKnown ? metrics.vramPercent : null}
          />
        </div>
      </SettingsSection>

      <SettingsSection title={t("settings.resources.gpu.title")}>
        {hasGpu ? (
          metrics.devices.map((device, index) => {
            const ordinal = deviceOrdinal(device);
            // Preserve null (unknown, e.g. Windows ROCm perf counter); coercing
            // to 0 would render a fabricated 0 used / full free.
            const total = device.memory_total_gb ?? null;
            const used = device.vram_used_gb ?? null;
            const free =
              device.vram_free_gb ??
              (isFiniteNumber(total) && isFiniteNumber(used)
                ? Math.max(0, total - used)
                : null);
            const percent =
              device.vram_utilization_pct ??
              (isFiniteNumber(total) && total > 0 && isFiniteNumber(used)
                ? (used / total) * 100
                : null);
            const safePercent = clampPercent(percent);
            const usedText = isFiniteNumber(used)
              ? formatGiB(used)
              : unknownLabel;
            const freeText = isFiniteNumber(free)
              ? formatGiB(free)
              : unknownLabel;
            const totalText = isFiniteNumber(total)
              ? formatGiB(total)
              : unknownLabel;
            const percentText = isFiniteNumber(percent)
              ? formatPercent(safePercent)
              : unknownLabel;
            return (
              <div
                key={`${device.index ?? index}-${device.name ?? "gpu"}`}
                className="flex min-w-0 flex-col gap-2 py-3"
              >
                <div className="flex min-w-0 items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-foreground">
                      {device.name ??
                        t("settings.resources.gpu.unknownDevice")}
                    </div>
                    <div className="mt-0.5 truncate text-xs text-muted-foreground">
                      {ordinal === undefined
                        ? backendLabel
                        : `${t("settings.resources.gpu.deviceWithIndex", {
                            index: ordinal,
                          })}, ${backendLabel}`}
                    </div>
                  </div>
                  <div className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                    <span>
                      {percentText}{" "}
                      {t("settings.resources.gpu.vramUtilization")}
                    </span>
                  </div>
                </div>
                <div className="grid gap-1 text-xs text-muted-foreground sm:grid-cols-3 sm:gap-2">
                  <span className="min-w-0 truncate font-mono tabular-nums">
                    {t("settings.resources.gpu.used", {
                      value: usedText,
                    })}
                  </span>
                  <span className="min-w-0 truncate font-mono tabular-nums sm:text-center">
                    {t("settings.resources.gpu.free", {
                      value: freeText,
                    })}
                  </span>
                  <span className="min-w-0 truncate font-mono tabular-nums sm:text-right">
                    {t("settings.resources.gpu.total", {
                      value: totalText,
                    })}
                  </span>
                </div>
                <Progress
                  value={safePercent}
                  aria-label={device.name ?? "GPU"}
                  className="h-1.5 rounded-full bg-muted"
                  indicatorClassName={usageIndicatorClass(safePercent)}
                />
              </div>
            );
          })
        ) : (
          <div className="py-3 text-sm text-muted-foreground">
            {t("settings.resources.gpu.noGpu")}
          </div>
        )}
      </SettingsSection>

      <SettingsSection title={t("settings.resources.storage.title")}>
        <InfoRow
          label={t("settings.resources.storage.systemDisk")}
          value={t("settings.resources.storage.diskUsage", {
            used: formatGb(metrics.diskUsed),
            total: formatGb(metrics.diskTotal),
          })}
          detail={t("settings.resources.storage.diskFree", {
            free: formatGb(metrics.diskFree),
          })}
        />
        <SettingsRow
          label={t("settings.resources.storage.modelsFolder")}
          description={t("settings.resources.storage.modelsFolderDescription")}
          className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
        >
          <div className="grid w-[392px] min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1.5 max-[840px]:w-full">
            <div className="relative min-w-0">
              <Input
                readOnly
                aria-label={t("settings.resources.storage.modelsFolder")}
                value={modelsFolderPath}
                title={hfCache?.cacheHome}
                className="h-8 w-full pr-7 font-mono text-xs"
              />
              <button
                type="button"
                disabled={!hfCache}
                onClick={() => void handleCacheFolder()}
                aria-label={
                  isTauri
                    ? t("settings.resources.storage.openAction")
                    : t("settings.resources.storage.copyAction")
                }
                title={
                  isTauri
                    ? t("settings.resources.storage.openAction")
                    : t("settings.resources.storage.copyAction")
                }
                className="absolute right-1.5 top-1/2 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
              >
                {isTauri ? (
                  <FolderOpenIcon className="size-3.5" />
                ) : (
                  <CopyIcon className="size-3.5" />
                )}
              </button>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-8"
              disabled={!hfCache?.editable || cacheSaving}
              onClick={() => void changeCacheFolder()}
            >
              {t("settings.resources.storage.changeAction")}
            </Button>
            {cacheLocationDetail || hfCache?.isCustom ? (
              <div className="col-span-2 flex min-w-0 items-center justify-between gap-2 pl-3.5 pr-1 text-xs text-muted-foreground">
                {cacheLocationDetail ? (
                  <span
                    title={cacheLocationDetail}
                    className="min-w-0 truncate"
                  >
                    {cacheLocationDetail}
                  </span>
                ) : null}
                {hfCache?.isCustom ? (
                  <Button
                    variant="link"
                    size="xs"
                    className="h-auto px-0 text-xs"
                    disabled={cacheSaving}
                    onClick={() => void saveCacheFolder(null)}
                  >
                    {t("settings.resources.storage.resetAction")}
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>
        </SettingsRow>
      </SettingsSection>

      <FolderBrowser
        open={!isTauri && cacheBrowserOpen}
        onOpenChange={setCacheBrowserOpen}
        onSelect={(path) => void saveCacheFolder(path)}
        initialPath={hfCache?.cacheHome}
        title={t("settings.resources.storage.chooseTitle")}
        confirmLabel={t("settings.resources.storage.chooseAction")}
        showModelHints={false}
      />

      <SettingsSection title={t("settings.resources.environment.title")}>
        <InfoRow
          label={t("settings.resources.environment.backend")}
          value={backendLabel}
        />
        <InfoRow
          label={t("settings.resources.environment.python")}
          value={systemInfo.python_version}
        />
        <InfoRow
          label={t("settings.resources.environment.torch")}
          value={
            systemInfo.ml_packages.torch ??
            t("settings.resources.environment.notInstalled")
          }
        />
        <InfoRow
          label={t("settings.resources.environment.transformers")}
          value={
            systemInfo.ml_packages.transformers ??
            t("settings.resources.environment.notInstalled")
          }
        />
        <InfoRow
          label={t("settings.resources.environment.uptime")}
          value={formatUptime(systemInfo.uptime_seconds)}
        />
        <InfoRow
          label={t("settings.resources.environment.processMemory")}
          value={formatMb(systemInfo.memory?.process_used_mb)}
        />
      </SettingsSection>
    </div>
  );
}
