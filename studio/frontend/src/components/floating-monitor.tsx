// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useFloatingPanelLayout } from "@/hooks/use-floating-panel-layout";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { FIND_PORTAL_ATTRIBUTE } from "@/features/find-in-page/lib/find-attributes";

import { useMonitorOverlayStore } from "@/features/settings";
import { gpuMemoryDisplay } from "@/hooks/gpu-memory-display";
import { gpuMemoryTotalsGb, resolveGpuVramUsedGb } from "@/hooks/gpu-vram";
import { useChatSettingsWidth } from "@/hooks/use-chat-settings-width";
import { useIsMobile } from "@/hooks/use-mobile";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useSidebarWidth } from "@/hooks/use-sidebar-width";
import { aggregateGpuMemoryTotalGb, useSystemInfo } from "@/hooks/use-system";
import { useT } from "@/i18n";
import {
  useFloatingPanelOrderStore,
  useFloatingPanelZIndex,
} from "@/lib/floating-panel-order";
import { cn } from "@/lib/utils";
import { useRouterState } from "@tanstack/react-router";
import { CpuIcon, GripVerticalIcon, XIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import {
  FLOATING_MONITOR_WIDTH,
  floatingMonitorConstraintStyle,
  getFloatingMonitorLayout,
} from "./floating-monitor-layout";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";

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

interface FloatingMonitorPanelProps {
  dockedBesideRunSettings: boolean;
  onClose: () => void;
  settingsWidth: number;
  onRenderedWidth: (width: number) => void;
  suppressed: boolean;
  systemInfo: ReturnType<typeof useSystemInfo>;
  /** The live `--ui-space-scale`; the resize handle's clearance follows it. */
  uiSpaceScale: number;
}
function FloatingMonitorPanel({
  dockedBesideRunSettings,
  onClose,
  onRenderedWidth,
  settingsWidth,
  suppressed,
  systemInfo,
  uiSpaceScale,
}: FloatingMonitorPanelProps) {
  const t = useT();
  const [constraintsElement, setConstraintsElement] =
    useState<HTMLDivElement | null>(null);
  const {
    monitorRef,
    scrollRef,
    contentRef,
    layout,
    startDrag,
    updateDrag,
    finishDrag,
  } = useFloatingPanelLayout(
    constraintsElement,
    dockedBesideRunSettings || suppressed,
    suppressed,
  );

  // offsetWidth, not the bounding rect: the panel animates in from scale 0.94, and
  // a rect read through that transform is short of the width it settles at.
  useEffect(() => {
    const monitor = monitorRef.current;
    if (!monitor) {
      return;
    }
    const measure = () => onRenderedWidth(monitor.offsetWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(monitor);
    return () => observer.disconnect();
  }, [monitorRef, onRenderedWidth]);

  const zIndex = useFloatingPanelZIndex("resource-monitor");
  const raisePanel = useFloatingPanelOrderStore((state) => state.raise);

  // Opening the monitor puts it in front of the API monitor panel; touching
  // either afterwards brings that one forward instead.
  useEffect(() => {
    raisePanel("resource-monitor");
  }, [raisePanel]);

  const ramTotal = systemInfo.memory?.total_gb ?? 0;
  const ramAvailable = systemInfo.memory?.available_gb ?? 0;
  const ramUsed = Math.max(0, ramTotal - ramAvailable);
  const ramPercent = clampPercent(systemInfo.memory?.percent_used ?? 0);

  const displayedGpu = systemInfo.gpu?.available
    ? systemInfo.gpu
    : (systemInfo.inference_gpu ?? systemInfo.gpu);
  const separateInferenceGpu =
    systemInfo.gpu?.available &&
    systemInfo.inference_gpu &&
    systemInfo.inference_gpu.backend !== systemInfo.gpu.backend
      ? systemInfo.inference_gpu
      : null;
  const memoryDisplay = gpuMemoryDisplay(displayedGpu);
  const inferenceDisplay = gpuMemoryDisplay(separateInferenceGpu);
  const inferenceVramTotal = separateInferenceGpu
    ? aggregateGpuMemoryTotalGb(inferenceDisplay.usageDevices)
    : 0;
  const devices = memoryDisplay.usageDevices;
  const memoryTotals = gpuMemoryTotalsGb(devices);
  const vramTotal = memoryTotals.total;
  const hasSharedPool = memoryTotals.shared > 0;
  // null usage = unknown (e.g. Windows ROCm perf counter); 0 would fabricate a
  // readout. The host figure can still be known when no device's is (#7452).
  const resolvedVramUsed = resolveGpuVramUsedGb(memoryDisplay.usageGpu);
  const vramUsageKnown = resolvedVramUsed !== null;
  const vramUsed = resolvedVramUsed ?? 0;
  const vramPercent = clampPercent(
    vramUsageKnown && vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0,
  );
  const unknownLabel = t("settings.resources.environment.unknown");

  const hasGpu =
    (displayedGpu?.available ?? false) &&
    (displayedGpu?.devices.length ?? 0) > 0;

  // The container sits on the floating panel layer, above the bottom-right overlay stack. The stack
  // is anchored to that same corner and does not move for this monitor, so the two can overlap. The
  // stack is passive status; this is a window being dragged, resized and closed, so it wins. Still
  // below the startup screen and tooltips. See lib/z-layers. The API monitor panel shares this
  // layer rather than sitting under it, and whichever of the two the user touched last is the one
  // in front.
  return (
    <div
      ref={setConstraintsElement}
      // The panel stays mounted while suppressed: the settings sheet is a
      // temporary overlay, and unmounting would throw away the position and the
      // browser-owned resize dimensions the user set. `invisible` keeps the box
      // (and the observers watching it) while taking it off the screen.
      aria-hidden={suppressed || undefined}
      className={cn(
        "pointer-events-none fixed inset-y-4 left-4",
        suppressed && "invisible",
        dockedBesideRunSettings ? undefined : "right-4",
      )}
      style={floatingMonitorConstraintStyle({
        zIndex,
        dockedBesideRunSettings,
        settingsWidth,
        uiSpaceScale,
      })}
    >
      <motion.div
        {...{ [FIND_PORTAL_ATTRIBUTE]: "" }}
        ref={monitorRef}
        onPointerDownCapture={() => raisePanel("resource-monitor")}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className={cn(
          "settings-surface bg-background pointer-events-auto absolute flex max-h-full w-64 max-w-full cursor-default select-none flex-col overflow-hidden rounded-xl border border-border/70 p-3 shadow-border ring-0 backdrop-blur-sm",
          layout ? "top-0 left-0 resize" : "right-0 bottom-0",
        )}
        data-testid="floating-monitor"
        style={
          layout
            ? {
                left: layout.left,
                top: layout.top,
                minWidth: Math.min(layout.minWidth, layout.maxWidth),
                minHeight: Math.min(layout.minHeight, layout.maxHeight),
                maxWidth: layout.maxWidth,
                maxHeight: layout.maxHeight,
              }
            : undefined
        }
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
              data-testid="floating-monitor-drag-handle"
              onPointerDown={startDrag}
              onPointerMove={updateDrag}
              onPointerUp={finishDrag}
              onPointerCancel={finishDrag}
              onLostPointerCapture={finishDrag}
              className="touch-none cursor-grab rounded-md px-1 text-muted-foreground/60 transition-colors hover:bg-muted/60 hover:text-muted-foreground active:cursor-grabbing"
            >
              <GripVerticalIcon className="size-3.5" />
            </div>

            <Button
              size="icon-xs"
              variant="ghost"
              className="text-muted-foreground hover:text-foreground"
              onClick={onClose}
              title={t("common.close")}
              aria-label={t("common.close")}
            >
              <XIcon className="size-3" />
            </Button>
          </div>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div
            ref={contentRef}
            data-testid="floating-monitor-content"
            className="space-y-3"
          >
            <div className="space-y-1">
              <div className="flex justify-between text-ui-11 font-medium font-mono">
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

            {hasGpu && devices.length > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-ui-11 font-medium font-mono">
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
                  {hasSharedPool
                    ? t("settings.resources.environment.vramWithShared", {
                        vram: formatGiB(memoryTotals.dedicated),
                        shared: formatGiB(memoryTotals.shared),
                      })
                    : formatGiB(vramTotal)}
                </div>
                <Progress
                  value={vramUsageKnown ? vramPercent : 0}
                  className="mt-1 h-1.5 rounded-full bg-muted"
                  indicatorClassName={usageIndicatorClass(vramPercent)}
                />
              </div>
            )}
            {hasGpu && memoryDisplay.sharedDevices.length > 0 && (
              <div className="space-y-1 text-xs">
                <div className="font-medium">
                  {t("settings.resources.gpu.sharedWithSystemRam")}
                </div>
                <div className="font-mono text-muted-foreground">
                  {t("settings.resources.gpu.estimatedAvailable", {
                    value:
                      memoryDisplay.sharedAvailableGb === null
                        ? unknownLabel
                        : formatGiB(memoryDisplay.sharedAvailableGb),
                  })}
                </div>
              </div>
            )}
            {separateInferenceGpu && (
              <div className="space-y-1 border-t border-border/60 pt-2 text-ui-11">
                <span className="block font-medium text-muted-foreground">
                  {t("settings.resources.gpu.ggufInference")}
                </span>
                <span className="block font-mono text-foreground">
                  {separateInferenceGpu.backend?.toUpperCase() ?? "GPU"}
                  {separateInferenceGpu.available ? (
                    inferenceVramTotal ? (
                      <span className="block">
                        {t("settings.resources.gpu.vramUtilization")}:{" "}
                        {formatGiB(inferenceVramTotal)}
                      </span>
                    ) : (
                      ""
                    )
                  ) : (
                    ` · ${t("settings.resources.gpu.unavailable")}`
                  )}
                  {separateInferenceGpu.available &&
                    inferenceDisplay.sharedDevices.length > 0 && (
                      <span className="block normal-case">
                        {t("settings.resources.gpu.sharedEstimatedAvailable", {
                          value:
                            inferenceDisplay.sharedAvailableGb === null
                              ? unknownLabel
                              : formatGiB(inferenceDisplay.sharedAvailableGb),
                        })}
                      </span>
                    )}
                </span>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export function FloatingMonitor() {
  const uiSpaceScale = useUiSpaceScale();
  const { isOpen, setIsOpen } = useMonitorOverlayStore();
  const settingsPanelOpen = useChatRuntimeStore((s) => s.settingsPanelOpen);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isMobile = useIsMobile();
  // The panel's own rendered width: it is user-resizable from 248 to 560 px, so the
  // docked offset has to come from the same value the panel paints at.
  const { width: committedSettingsWidth } = useChatSettingsWidth();
  const { pinned } = useSidebarPin();
  const { width: committedSidebarWidth } = useSidebarWidth();
  const isChatRoute = pathname === "/chat";

  // Dragging the panel's edge paints `--chat-settings-width` straight onto the
  // <aside> and commits to the store only on pointer up, so the store trails the
  // panel by a whole drag. The monitor has to clear what is on screen, so it
  // follows the painted width and falls back to the committed one.
  const [paintedSettingsWidth, setPaintedSettingsWidth] = useState(0);
  useEffect(() => {
    if (!(isChatRoute && settingsPanelOpen)) {
      setPaintedSettingsWidth(0);
      return;
    }
    const panel = document.querySelector('[data-slot="chat-settings-panel"]');
    if (!panel) {
      setPaintedSettingsWidth(0);
      return;
    }
    const measure = () => {
      if (panel.isConnected) {
        setPaintedSettingsWidth(panel.getBoundingClientRect().width);
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(panel);
    return () => observer.disconnect();
  }, [isChatRoute, settingsPanelOpen, isMobile]);

  // Same lag as the settings panel: the sidebar paints per frame and commits
  // on release, so a docked monitor could sit under a grown sidebar.
  const [paintedSidebarWidth, setPaintedSidebarWidth] = useState(0);
  // Unpinning still holds the collapsed icon rail as a column on the web shell;
  // `collapseToZero` is desktop-app only, and that rail is the one unpinned
  // state that takes the monitor's room.
  const [sidebarHoldsRail, setSidebarHoldsRail] = useState(false);
  useEffect(() => {
    const sidebar = document.querySelector('[data-slot="sidebar"]');
    if (!sidebar) {
      setPaintedSidebarWidth(0);
      setSidebarHoldsRail(false);
      return;
    }
    const measure = () => {
      if (sidebar.isConnected) {
        setPaintedSidebarWidth(sidebar.getBoundingClientRect().width);
        setSidebarHoldsRail(
          sidebar.getAttribute("data-collapsible") === "icon",
        );
      }
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(sidebar);
    return () => observer.disconnect();
  }, [isMobile]);

  // `useIsMobile` only notifies when the 768 px breakpoint is crossed, and the
  // two width stores stop notifying at their maxima, so the capacity decision
  // needs its own subscription or a plain resize never re-evaluates it.
  const viewportWidth = useSyncExternalStore(
    (onChange) => {
      window.addEventListener("resize", onChange);
      return () => window.removeEventListener("resize", onChange);
    },
    () => window.innerWidth,
    () => 0,
  );

  const settingsWidth =
    paintedSettingsWidth > 0 ? paintedSettingsWidth : committedSettingsWidth;
  const pinnedSidebarWidth =
    paintedSidebarWidth > 0 ? paintedSidebarWidth : committedSidebarWidth;
  // A pinned sidebar holds its column; an unpinned one overlays the content, but
  // the web shell still paints its collapsed icon rail as a column, so reserve
  // that measured rail while it really is one.
  const unpinnedSidebarWidth = sidebarHoldsRail ? paintedSidebarWidth : 0;
  const sidebarWidth = pinned ? pinnedSidebarWidth : unpinnedSidebarWidth;
  // The panel is natively resizable, so what it renders is what docking has to
  // reserve. Before the first measure the constant is the floor.
  const [monitorWidth, setMonitorWidth] = useState(FLOATING_MONITOR_WIDTH);
  const { visible, suppressed, dockedBesideRunSettings } =
    getFloatingMonitorLayout({
      isOpen,
      isMobile,
      isChatRoute,
      settingsPanelOpen,
      settingsWidth,
      sidebarWidth,
      viewportWidth,
      monitorWidth,
      uiSpaceScale,
    });
  const systemInfo = useSystemInfo({ enabled: visible, pollMs: 5000 });
  const [panelKey, setPanelKey] = useState(0);
  const wasOpenRef = useRef(isOpen);

  // Each visible panel owns native inline resize state. Advance the key on
  // close so reopening during the exit animation still mounts fresh geometry.
  // Docking and the mobile yield are not closes, so they do not advance it.
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      setPanelKey((current) => current + 1);
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  return (
    <AnimatePresence>
      {(visible || suppressed) && (
        <FloatingMonitorPanel
          key={panelKey}
          // The mobile sheet covers the panel rather than closing it, so the
          // panel is kept mounted and invisible: the position and the size the
          // user set survive the sheet being opened and closed.
          suppressed={suppressed}
          dockedBesideRunSettings={dockedBesideRunSettings}
          settingsWidth={settingsWidth}
          uiSpaceScale={uiSpaceScale}
          onRenderedWidth={setMonitorWidth}
          systemInfo={systemInfo}
          onClose={() => setIsOpen(false)}
        />
      )}
    </AnimatePresence>
  );
}
