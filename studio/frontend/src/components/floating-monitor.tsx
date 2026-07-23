// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useMonitorOverlayStore } from "@/features/settings";
import { useSystemInfo } from "@/hooks/use-system";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { CpuIcon, GripVerticalIcon, XIcon } from "lucide-react";
import { AnimatePresence, motion, useDragControls } from "motion/react";
import { type PointerEvent, useMemo, useState } from "react";

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function usageIndicatorClass(percent: number): string {
  if (percent >= 90) {
    return "bg-destructive";
  }
  if (percent >= 70) {
    return "bg-amber-500";
  }
  return "bg-control-accent";
}

function usageTextClass(percent: number): string {
  if (percent >= 90) {
    return "text-destructive";
  }
  if (percent >= 70) {
    return "text-amber-600 dark:text-amber-400";
  }
  return "text-primary";
}

function formatGiB(value: number): string {
  // RAM/VRAM come from the backend in binary units (bytes / 1024**3), matching
  // nvidia-smi and PyTorch, so label the readout GiB rather than GB.
  const digits = value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} GiB`;
}

export function FloatingMonitor() {
  const t = useT();
  const { isOpen, setIsOpen } = useMonitorOverlayStore();
  const systemInfo = useSystemInfo({ enabled: isOpen, pollMs: 5000 });

  const [constraintsElement, setConstraintsElement] =
    useState<HTMLDivElement | null>(null);
  const constraintsRef = useMemo(
    () => ({ current: constraintsElement }),
    [constraintsElement],
  );
  const dragControls = useDragControls();

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    dragControls.start(event);
  }

  const ramTotal = systemInfo.memory?.total_gb ?? 0;
  const ramAvailable = systemInfo.memory?.available_gb ?? 0;
  const ramUsed = Math.max(0, ramTotal - ramAvailable);
  const ramPercent = clampPercent(systemInfo.memory?.percent_used ?? 0);

  const devices = systemInfo.gpu?.devices ?? [];
  const vramTotal = devices.reduce(
    (sum, device) => sum + (device.memory_total_gb ?? 0),
    0,
  );
  // null usage = unknown (e.g. Windows ROCm perf counter): treating it as 0
  // fabricates a 0-used readout, so the aggregate is unknown if any device is.
  const vramUsageKnown =
    devices.length > 0 &&
    devices.every((device) => Number.isFinite(device.vram_used_gb));
  const vramUsed = vramUsageKnown
    ? devices.reduce((sum, device) => sum + (device.vram_used_gb ?? 0), 0)
    : 0;
  const vramPercent = clampPercent(
    vramUsageKnown && vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0,
  );
  const unknownLabel = t("settings.resources.environment.unknown");

  const hasGpu = (systemInfo.gpu?.available ?? false) && devices.length > 0;

  return (
    <AnimatePresence>
      {isOpen && (
        <div
          ref={setConstraintsElement}
          className="fixed inset-0 z-50 pointer-events-none"
        >
          <motion.div
            drag={true}
            dragControls={dragControls}
            dragListener={false}
            dragConstraints={constraintsRef}
            dragElastic={0}
            dragMomentum={false}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="settings-surface fixed bottom-4 right-4 w-64 max-w-[calc(100vw-32px)] resize overflow-hidden rounded-xl border border-border/70 p-3 shadow-border ring-0 backdrop-blur-sm pointer-events-auto cursor-default select-none"
          >
            <div className="mb-2 flex items-center justify-between gap-2 border-b border-border/60 pb-2">
              <div className="flex min-w-0 flex-1 items-center gap-1.5 truncate text-xs font-semibold text-foreground">
                <CpuIcon className="size-3.5 shrink-0 text-primary" />
                <span className="truncate">
                  {t("settings.resources.liveMonitor.title")}
                </span>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <div
                  onPointerDown={startDrag}
                  className="touch-none cursor-grab rounded-md px-1 text-muted-foreground/60 transition-colors hover:bg-muted/60 hover:text-muted-foreground active:cursor-grabbing"
                >
                  <GripVerticalIcon className="size-3.5" />
                </div>

                <Button
                  size="icon-xs"
                  variant="ghost"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => setIsOpen(false)}
                  title={t("common.close")}
                  aria-label={t("common.close")}
                >
                  <XIcon className="size-3" />
                </Button>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-3 overflow-hidden"
            >
              <div className="space-y-1">
                <div className="flex justify-between text-[0.6875rem] font-medium font-mono">
                  <span>{t("settings.resources.liveMonitor.ram")}</span>
                  <span
                    className={cn("tabular-nums", usageTextClass(ramPercent))}
                  >
                    {Math.round(ramPercent)}%
                  </span>
                </div>
                <div className="text-xs text-muted-foreground font-mono tabular-nums">
                  {formatGiB(ramUsed)} / {formatGiB(ramTotal)}
                </div>
                <Progress
                  value={ramPercent}
                  className="mt-1 h-1.5 rounded-full bg-muted"
                  indicatorClassName={usageIndicatorClass(ramPercent)}
                />
              </div>

              {hasGpu && (
                <div className="space-y-1">
                  <div className="flex justify-between text-[0.6875rem] font-medium font-mono">
                    <span className="truncate flex-1 pr-2">
                      {t("settings.resources.liveMonitor.vram")}{" "}
                      {devices.length > 1
                        ? `(${devices.length} GPUs)`
                        : `(${devices[0].name ?? "GPU"})`}
                    </span>
                    <span
                      className={cn(
                        "shrink-0 tabular-nums",
                        vramUsageKnown
                          ? usageTextClass(vramPercent)
                          : "text-muted-foreground",
                      )}
                    >
                      {vramUsageKnown ? `${Math.round(vramPercent)}%` : "--"}
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground font-mono tabular-nums">
                    {vramUsageKnown ? formatGiB(vramUsed) : unknownLabel} /{" "}
                    {formatGiB(vramTotal)}
                  </div>
                  <Progress
                    value={vramUsageKnown ? vramPercent : 0}
                    className="mt-1 h-1.5 rounded-full bg-muted"
                    indicatorClassName={usageIndicatorClass(vramPercent)}
                  />
                </div>
              )}
            </motion.div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
