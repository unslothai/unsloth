// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSystemInfo } from "@/hooks/use-system";
import { useMonitorOverlayStore } from "@/features/settings/stores/monitor-overlay-store";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { motion } from "motion/react";
import { CpuIcon, XIcon, GripVerticalIcon } from "lucide-react";
import { useRef } from "react";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function usageIndicatorClass(percent: number): string {
  if (percent >= 90) return "bg-destructive";
  if (percent >= 70) return "bg-amber-500";
  return "bg-primary";
}

function formatGb(value: number): string {
  const digits = value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} GB`;
}

export function FloatingMonitor() {
  const t = useT();
  const { isOpen, setIsOpen } = useMonitorOverlayStore();
  const systemInfo = useSystemInfo({ enabled: isOpen, pollMs: 5000 });

  const constraintsRef = useRef<HTMLDivElement>(null);

  if (!isOpen) return null;

  const ramTotal = systemInfo.memory.total_gb ?? 0;
  const ramAvailable = systemInfo.memory.available_gb ?? 0;
  const ramUsed = Math.max(0, ramTotal - ramAvailable);
  const ramPercent = clampPercent(systemInfo.memory.percent_used ?? 0);

  const devices = systemInfo.gpu.devices ?? [];
  const vramTotal = devices.reduce(
    (sum, device) => sum + (device.memory_total_gb ?? 0),
    0,
  );
  const vramUsed = devices.reduce(
    (sum, device) => sum + (device.vram_used_gb ?? 0),
    0,
  );
  const vramPercent = clampPercent(vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0);

  const hasGpu = (systemInfo.gpu.available ?? false) && devices.length > 0;

  return (
    <div ref={constraintsRef} className="fixed inset-0 z-50 pointer-events-none">
      <motion.div
        layout
        drag
        dragConstraints={constraintsRef}
        dragElastic={0.1}
        dragMomentum={false}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="fixed resize bottom-4 right-4 pointer-events-auto rounded-xl border border-border bg-background p-3 shadow-2xl w-60 cursor-default select-none overflow-hidden"
      >
        <div className="flex items-center justify-between border-b pb-2 mb-2 gap-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground truncate flex-1">
            <CpuIcon className="size-3.5 text-primary shrink-0" />
            <span className="truncate">Resource Monitor</span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <div className="text-muted-foreground/50 cursor-grab hover:text-muted-foreground active:cursor-grabbing px-1">
              <GripVerticalIcon className="size-3.5" />
            </div>

            <Button size="icon" variant="ghost" className="size-5 rounded text-destructive hover:bg-destructive/10" onClick={() => setIsOpen(false)} title={t("common.close")}>
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
            <div className="flex justify-between text-[11px] font-medium font-mono">
              <span>RAM</span>
              <span className={cn("tabular-nums", ramPercent)}>
                {Math.round(ramPercent)}%
              </span>
            </div>
            <div className="text-xs text-muted-foreground font-mono tabular-nums">
              {formatGb(ramUsed)} / {formatGb(ramTotal)}
            </div>
            <Progress 
              value={ramPercent} 
              className="h-1.5 rounded-full bg-muted mt-1" 
              indicatorClassName={usageIndicatorClass(ramPercent)}
            />
          </div>

          {hasGpu && (
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-medium font-mono">
                <span className="truncate flex-1 pr-2">
                  VRAM {devices.length > 1 ? `(${devices.length} GPUs)` : `(${devices[0].name ?? "GPU"})`}
                </span>
                <span className={cn("shrink-0 tabular-nums", vramPercent)}>
                  {Math.round(vramPercent)}%
                </span>
              </div>
              <div className="text-xs text-muted-foreground font-mono tabular-nums">
                {formatGb(vramUsed)} / {formatGb(vramTotal)}
              </div>
              <Progress 
                value={vramPercent} 
                className="h-1.5 rounded-full bg-muted mt-1" 
                indicatorClassName={usageIndicatorClass(vramPercent)}
              />
            </div>
          )}
        </motion.div>
      </motion.div>
    </div>
  );
}