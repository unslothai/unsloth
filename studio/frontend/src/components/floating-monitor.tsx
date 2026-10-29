// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  FLOATING_PANEL_POSITION_KEYS,
  type ResizeEdges,
} from "@/hooks/floating-panel-position";
import { useFloatingPanelLayout } from "@/hooks/use-floating-panel-layout";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { FIND_PORTAL_ATTRIBUTE } from "@/features/find-in-page/lib/find-attributes";

import { useMonitorOverlayStore } from "@/features/settings";
import { monitorRefreshMs } from "./monitor-refresh";
import { vramCards } from "./monitor-cards";
import {
  formatGiB,
  MeterRow,
  MemoryShareBar,
  UsageBar,
  usageIndicatorClass,
  usageTextClass,
} from "./monitor-memory";
import {
  memorySegments,
  memoryShare,
  modelBreakdown,
  modelShares,
  unslothPoolGb,
} from "./monitor-share";
import type { LoadedModelEntry } from "@/features/loaded-models";
import { useCornerCardFrameStore } from "@/features/settings/stores/monitor-frame-store";
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
import { CpuIcon, GripVerticalIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import {
  formatPower,
  formatTemperature,
  gpuTelemetryRows,
  selectedDetailMetrics,
  temperatureLevel,
} from "./monitor-details";
import {
  FLOATING_MONITOR_WIDTH,
  floatingMonitorConstraintStyle,
  getFloatingMonitorLayout,
} from "./floating-monitor-layout";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

const LEVEL_INDICATOR_CLASS = {
  ok: "bg-control-accent",
  warm: "bg-amber-500",
  hot: "bg-destructive",
} as const;

const LEVEL_TEXT_CLASS = {
  ok: "text-primary",
  warm: "text-amber-600 dark:text-amber-400",
  hot: "text-destructive",
} as const;

function DetailBar({
  label,
  value,
  percent,
  indicatorClassName,
  textClassName,
}: {
  label: string;
  value: string;
  percent: number;
  indicatorClassName: string;
  textClassName: string;
}) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between gap-2 text-xs text-muted-foreground">
        <span className="truncate">{label}</span>
        <span className={cn("shrink-0 font-mono tabular-nums", textClassName)}>
          {value}
        </span>
      </div>
      <Progress
        value={clampPercent(percent)}
        className="h-1 rounded-full bg-muted"
        indicatorClassName={indicatorClassName}
      />
    </div>
  );
}

const MonitorLoadedModels = lazy(() => import("./monitor-loaded-models"));

const BACKGROUND_DRAG_EXCLUDED =
  'button, a, input, select, textarea, [role="button"], [data-testid="floating-monitor-resize"]';

const RESIZE_HANDLES: { edge: string; edges: ResizeEdges; className: string }[] = [
  { edge: "w", edges: { x: -1, y: 0 }, className: "inset-y-0 left-0 w-1.5 cursor-ew-resize" },
  { edge: "e", edges: { x: 1, y: 0 }, className: "inset-y-0 right-0 w-1.5 cursor-ew-resize" },
  { edge: "nw", edges: { x: -1, y: -1 }, className: "left-0 top-0 z-20 size-4 cursor-nwse-resize" },
  { edge: "se", edges: { x: 1, y: 1 }, className: "bottom-0 right-0 z-20 size-5 cursor-nwse-resize" },
  { edge: "ne", edges: { x: 1, y: -1 }, className: "right-0 top-0 z-20 size-4 cursor-nesw-resize" },
  { edge: "sw", edges: { x: -1, y: 1 }, className: "bottom-0 left-0 z-20 size-5 cursor-nesw-resize" },
];

const MONITOR_RESIZE = {
  minWidth: 232,
  maxWidth: 520,
  storageKey: FLOATING_PANEL_POSITION_KEYS.liveMonitorWidth,
};

interface FloatingMonitorPanelProps {
  dockedBesideRunSettings: boolean;
  onClose: () => void;
  settingsWidth: number;
  onRenderedWidth: (width: number) => void;
  suppressed: boolean;
  systemInfo: ReturnType<typeof useSystemInfo>;
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
  const cornerCard = useCornerCardFrameStore((state) => state.frame);
  const {
    monitorRef,
    scrollRef,
    contentRef,
    layout,
    startDrag,
    updateDrag,
    finishDrag,
    resetPosition,
    nudge,
    width,
    startResize,
    updateResize,
    finishResize,
    resetWidth,
  } = useFloatingPanelLayout(
    constraintsElement,
    dockedBesideRunSettings || suppressed,
    suppressed,
    undefined,
    FLOATING_PANEL_POSITION_KEYS.liveMonitor,
    cornerCard,
    MONITOR_RESIZE,
  );

  // offsetWidth, not the bounding rect: a drag's transform skews the rect.
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

  useEffect(() => {
    raisePanel("resource-monitor");
  }, [raisePanel]);

  const ramTotal = systemInfo.memory?.total_gb ?? 0;
  const ramAvailable = systemInfo.memory?.available_gb ?? 0;
  const ramUsed = Math.max(0, ramTotal - ramAvailable);
  const ramPercent = clampPercent(systemInfo.memory?.percent_used ?? 0);
  // The placeholder's zeros are not a reading: loading or a failed first request shows "--".
  const ramKnown = systemInfo.status === "ready";
  const stale = systemInfo.stale === true;

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
  const hasSharedPool = memoryTotals.shared > 0;
  // No dedicated pool (Apple unified memory): the shared total alone is the capacity.
  const unifiedPool = hasSharedPool && memoryTotals.dedicated === 0;
  // null usage = unknown (e.g. Windows ROCm perf counter); 0 would fabricate a
  // readout. The host figure can still be known when no device's is (#7452).
  const resolvedVramUsed = resolveGpuVramUsedGb(memoryDisplay.usageGpu);
  const vramUsageKnown = resolvedVramUsed !== null;
  const vramUsed = resolvedVramUsed ?? 0;
  const unknownLabel = t("settings.resources.environment.unknown");

  const showDetails = useMonitorOverlayStore((state) => state.showDetails);
  const storedDetailMetrics = useMonitorOverlayStore(
    (state) => state.detailMetrics,
  );
  const detailMetrics = selectedDetailMetrics(storedDetailMetrics);
  const cpuPercent = ramKnown
    ? clampPercent(systemInfo.cpu?.usage_percent ?? 0)
    : null;
  const telemetryRows =
    showDetails && displayedGpu?.available
      ? gpuTelemetryRows(displayedGpu.devices, detailMetrics)
      : [];

  const hasGpu =
    (displayedGpu?.available ?? false) &&
    (displayedGpu?.devices.length ?? 0) > 0;
  // Apple unified memory: RAM and "VRAM" are one pool, so two bars contradict each other.
  const unifiedRow =
    hasGpu &&
    devices.length > 0 &&
    unifiedPool &&
    memoryDisplay.sharedDevices.length === 0;
  const discreteVram = hasGpu && devices.length > 0 && !unifiedRow;
  const vram = vramCards(devices, memoryDisplay.usageGpu);
  const cardsUsedKnown = vram.usedGb !== null;
  const cardsUsed = vram.usedGb ?? 0;
  const cardsPercent = clampPercent(
    cardsUsedKnown && vram.totalGb > 0 ? (cardsUsed / vram.totalGb) * 100 : 0,
  );
  const cardLabel =
    vram.cards.length === 1 ? (vram.cards[0]?.name ?? undefined) : undefined;

  const showShare = useMonitorOverlayStore((state) => state.showUnslothShare);
  const showBreakdown = useMonitorOverlayStore(
    (state) => state.showModelBreakdown,
  );
  const wantsModels = showShare || showBreakdown;
  const [loadedModels, setLoadedModels] = useState<LoadedModelEntry[]>([]);
  const models = wantsModels ? modelShares(loadedModels) : [];
  const modelsGb = models.reduce((sum, model) => sum + model.gb, 0);
  // Unsloth Studio's server plus its children (llama-server, workers), resident set.
  const processMb =
    systemInfo.memory?.process_tree_used_mb ?? systemInfo.memory?.process_used_mb;
  const unslothRamGb = unslothPoolGb(
    ramKnown && processMb !== undefined ? processMb / 1024 : null,
    modelsGb,
    !discreteVram,
  );
  // The model split lives in the pool that holds the models: Memory/RAM, or VRAM on a discrete GPU.
  const breakdown = showBreakdown
    ? modelBreakdown(discreteVram ? null : unslothRamGb, models, !discreteVram)
    : null;
  const ramShare =
    showShare || (breakdown && !discreteVram)
      ? memoryShare(ramUsed, ramTotal, unslothRamGb)
      : null;
  const ramSegments = ramShare
    ? memorySegments(ramShare, discreteVram ? null : breakdown)
    : null;
  // On a discrete GPU the models are what Unsloth holds in VRAM.
  const vramShare =
    wantsModels && discreteVram && cardsUsedKnown
      ? memoryShare(cardsUsed, vram.totalGb, modelsGb)
      : null;
  const vramSegments = vramShare ? memorySegments(vramShare, breakdown, false) : null;
  const ramAmount = ramKnown
    ? `${formatGiB(ramUsed)} / ${formatGiB(ramTotal)}`
    : "-- / --";

  // The container sits on the floating panel layer, above the bottom-right overlay stack. The stack
  // is anchored to that same corner and does not move for this monitor; at its default corner the
  // monitor rests above the Loaded models card instead (useCornerCardFrameStore), and anywhere it
  // overlaps it wins, since this is a window being dragged, resized and closed. Still below the
  // startup screen and tooltips. See lib/z-layers. The API monitor panel shares this layer rather
  // than sitting under it, and whichever of the two the user touched last is the one in front.
  return (
    <div
      ref={setConstraintsElement}
      // Stays mounted while suppressed so the user's position and resize dimensions survive.
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
        // The whole card is a grip; controls, resize edges and the grip itself keep their own pointer.
        onPointerDown={(event) => {
          if (
            event.target instanceof Element &&
            event.target.closest(BACKGROUND_DRAG_EXCLUDED)
          ) {
            return;
          }
          // A classic scrollbar sits inside the scroller's box but right of its client area.
          const target = event.target as HTMLElement;
          if (event.nativeEvent.offsetX > target.clientWidth && target.clientWidth > 0) {
            return;
          }
          startDrag(event);
        }}
        onPointerMove={updateDrag}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
        onLostPointerCapture={finishDrag}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className={cn(
          "settings-surface bg-background pointer-events-auto absolute flex max-h-full w-64 max-w-full cursor-grab select-none active:cursor-grabbing flex-col overflow-hidden rounded-2xl border border-border/70 p-3 shadow-border ring-0",
          layout ? "top-0 left-0" : "right-0 bottom-0",
        )}
        data-testid="floating-monitor"
        // Takes its own clicks over a macOS browser page (native-view.ts).
        data-native-clickable=""
        // biome-ignore lint/a11y/useSemanticElements: a floating panel landmark on the shared layout hook's div.
        role="region"
        aria-label={t("settings.resources.liveMonitor.title")}
        onKeyDown={(event) => {
          if (event.key !== "Escape" || event.defaultPrevented) return;
          event.preventDefault();
          onClose();
        }}
        style={
          layout
            ? {
                left: layout.left,
                top: layout.top,
                minWidth: Math.min(layout.minWidth, layout.maxWidth),
                minHeight: Math.min(layout.minHeight, layout.maxHeight),
                maxWidth: layout.maxWidth,
                maxHeight: layout.maxHeight,
                width: width ?? undefined,
              }
            : { width: width ?? undefined }
        }
      >
        {wantsModels && (
          <Suspense fallback={null}>
            <MonitorLoadedModels onEntries={setLoadedModels} />
          </Suspense>
        )}
        {RESIZE_HANDLES.map(({ edge, edges, className }) => (
          <div
            key={edge}
            aria-hidden
            data-testid="floating-monitor-resize"
            data-edge={edge}
            title={t("floatingPanels.monitor.resizeHint")}
            onPointerDown={(event) => startResize(event, edges)}
            onPointerMove={updateResize}
            onPointerUp={finishResize}
            onPointerCancel={finishResize}
            onLostPointerCapture={finishResize}
            onDoubleClick={resetWidth}
            className={cn("absolute z-10 touch-none", className)}
          />
        ))}
        <div className="mb-2 flex items-center justify-between gap-2 border-b border-border/60 pb-2">
          <div className="flex min-w-0 flex-1 items-center gap-1.5 truncate text-xs font-semibold text-foreground">
            <CpuIcon className="size-3.5 shrink-0 text-primary" />
            <span className="truncate">
              {t("settings.resources.liveMonitor.title")}
            </span>
            {stale && (
              <span
                className="truncate font-normal text-muted-foreground"
                aria-live="polite"
                data-testid="floating-monitor-stale"
              >
                {t("floatingPanels.monitor.reconnecting")}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <div
              data-testid="floating-monitor-drag-handle"
              // biome-ignore lint/a11y/useSemanticElements: a pointer-captured drag grip, not a pressable control.
              role="button"
              tabIndex={0}
              aria-label={t("floatingPanels.monitor.dragHandleLabel")}
              title={t("floatingPanels.monitor.dragHandleLabel")}
              onPointerDown={startDrag}
              onPointerMove={updateDrag}
              onPointerUp={finishDrag}
              onPointerCancel={finishDrag}
              onLostPointerCapture={finishDrag}
              onDoubleClick={resetPosition}
              onKeyDown={(event) => void nudge(event)}
              className="touch-none cursor-grab rounded-full px-1 text-muted-foreground/60 transition-colors hover:bg-muted/60 hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:cursor-grabbing"
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
              <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
            </Button>
          </div>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div
            ref={contentRef}
            data-testid="floating-monitor-content"
            className={cn("space-y-3 transition-opacity", stale && "opacity-50")}
          >
            <MeterRow
              testId="floating-monitor-cpu"
              label={t("settings.resources.liveMonitor.cpu")}
              readout={cpuPercent === null ? "--" : `${Math.round(cpuPercent)}%`}
              readoutClassName={
                cpuPercent === null ? undefined : usageTextClass(cpuPercent)
              }
            >
              <UsageBar percent={cpuPercent} />
            </MeterRow>
            <MeterRow
              testId={unifiedRow ? "floating-monitor-memory" : "floating-monitor-ram"}
              label={
                unifiedRow
                  ? t("settings.resources.liveMonitor.memory")
                  : t("settings.resources.liveMonitor.ram")
              }
              readout={ramAmount}
              footer={
                unifiedRow &&
                !ramSegments && (
                  <>
                    {unifiedRow && !ramSegments && (
                      <div className="flex justify-between gap-2">
                        <span>{t("settings.resources.liveMonitor.gpuInUse")}</span>
                        <span className="shrink-0 font-mono tabular-nums">
                          {vramUsageKnown ? formatGiB(vramUsed) : unknownLabel}
                        </span>
                      </div>
                    )}
                    {unifiedRow &&
                      !ramSegments &&
                      memoryDisplay.unifiedAvailableGb !== null && (
                        <div className="flex justify-between gap-2">
                          <span>
                            {t("settings.resources.liveMonitor.availableForModels")}
                          </span>
                          <span className="shrink-0 font-mono tabular-nums">
                            {formatGiB(memoryDisplay.unifiedAvailableGb)}
                          </span>
                        </div>
                      )}
                  </>
                )
              }
            >
              {ramSegments ? (
                <MemoryShareBar
                  segments={ramSegments}
                  noModels={!discreteVram && breakdown?.models.length === 0}
                />
              ) : (
                <UsageBar percent={ramKnown ? ramPercent : null} />
              )}
            </MeterRow>
            {/* One card, or several with the share on: one VRAM row over every card. */}
            {discreteVram && (vram.rows.length === 0 || vramSegments) && (
              <MeterRow
                testId="floating-monitor-vram"
                label={t("settings.resources.liveMonitor.vram")}
                readout={`${cardsUsedKnown ? formatGiB(cardsUsed) : "--"} / ${formatGiB(vram.totalGb)}`}
                footer={
                  cardLabel && (
                    <div className="truncate" title={cardLabel}>
                      {cardLabel}
                    </div>
                  )
                }
              >
                {vramSegments ? (
                  <MemoryShareBar
                    segments={vramSegments}
                    noModels={breakdown?.models.length === 0}
                  />
                ) : (
                  <UsageBar percent={cardsUsedKnown ? cardsPercent : null} />
                )}
              </MeterRow>
            )}
            {/* Several cards: each is a row of its own, the same size as the rest. */}
            {discreteVram &&
              vram.rows.map((row) => {
                const percent =
                  row.usedGb !== null && row.totalGb > 0
                    ? clampPercent((row.usedGb / row.totalGb) * 100)
                    : null;
                return (
                  <MeterRow
                    key={row.key}
                    testId="floating-monitor-gpu-card"
                    label={row.label}
                    readout={`${row.usedGb === null ? "--" : formatGiB(row.usedGb)} / ${formatGiB(row.totalGb)}`}
                    footer={
                      <div className="truncate" title={row.title}>
                        {row.name}
                      </div>
                    }
                  >
                    <UsageBar percent={percent} />
                  </MeterRow>
                );
              })}
            {hasGpu && memoryDisplay.sharedDevices.length > 0 && (
              <div className="space-y-1 text-xs">
                <div className="font-medium">
                  {t("settings.resources.gpu.sharedWithSystemRam")}
                </div>
                <div className="font-mono tabular-nums text-muted-foreground">
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
                <span className="block text-foreground">
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
            {telemetryRows.map((row) => (
              <div
                key={row.key}
                className="space-y-1.5 border-t border-border/60 pt-2"
                data-testid="floating-monitor-gpu-details"
              >
                <div className="truncate text-xs font-medium text-foreground">
                  {row.name}
                </div>
                {row.loadPct !== null && (
                  <DetailBar
                    label={t("settings.resources.liveMonitor.gpuLoad")}
                    value={`${Math.round(row.loadPct)}%`}
                    percent={row.loadPct}
                    indicatorClassName={usageIndicatorClass(row.loadPct)}
                    textClassName={usageTextClass(row.loadPct)}
                  />
                )}
                {row.temperatureC !== null && (
                  <DetailBar
                    label={t("settings.resources.liveMonitor.temperature")}
                    value={formatTemperature(row.temperatureC)}
                    percent={row.temperatureC}
                    indicatorClassName={
                      LEVEL_INDICATOR_CLASS[temperatureLevel(row.temperatureC)]
                    }
                    textClassName={
                      LEVEL_TEXT_CLASS[temperatureLevel(row.temperatureC)]
                    }
                  />
                )}
                {row.powerDrawW !== null &&
                  (row.powerPct !== null ? (
                    <DetailBar
                      label={t("settings.resources.liveMonitor.power")}
                      value={formatPower(row.powerDrawW, row.powerLimitW)}
                      percent={row.powerPct}
                      indicatorClassName={usageIndicatorClass(row.powerPct)}
                      textClassName={usageTextClass(row.powerPct)}
                    />
                  ) : (
                    <div className="flex justify-between gap-2 text-xs text-muted-foreground">
                      <span className="truncate">
                        {t("settings.resources.liveMonitor.power")}
                      </span>
                      <span className="shrink-0 font-mono tabular-nums text-primary">
                        {formatPower(row.powerDrawW, null)}
                      </span>
                    </div>
                  ))}
              </div>
            ))}
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
  const { width: committedSettingsWidth } = useChatSettingsWidth();
  const { pinned } = useSidebarPin();
  const { width: committedSidebarWidth } = useSidebarWidth();
  const isChatRoute = pathname === "/chat";

  // Dragging paints `--chat-settings-width` and commits only on pointer up; follow the painted width.
  const [paintedSettingsWidth, setPaintedSettingsWidth] = useState(0);
  useEffect(() => {
    if (!(isOpen && isChatRoute && settingsPanelOpen)) {
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
  }, [isOpen, isChatRoute, settingsPanelOpen, isMobile]);

  // The sidebar paints per frame and commits on release, so track painted width for docking.
  const [paintedSidebarWidth, setPaintedSidebarWidth] = useState(0);
  // The web shell's collapsed icon rail still holds a column when unpinned.
  const [sidebarHoldsRail, setSidebarHoldsRail] = useState(false);
  useEffect(() => {
    const sidebar = isOpen ? document.querySelector('[data-slot="sidebar"]') : null;
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
  }, [isOpen, isMobile]);

  // useIsMobile and the width stores do not notify on every resize, so subscribe directly.
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
  const unpinnedSidebarWidth = sidebarHoldsRail ? paintedSidebarWidth : 0;
  const sidebarWidth = pinned ? pinnedSidebarWidth : unpinnedSidebarWidth;
  // The panel is resizable, so dock using its measured width; the constant is the floor.
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
  const refreshMs = useMonitorOverlayStore((state) =>
    monitorRefreshMs(state.refreshMs),
  );
  const systemInfo = useSystemInfo({
    enabled: visible,
    pollMs: refreshMs,
    gpuFollowsPoll: true,
  });
  const [panelKey, setPanelKey] = useState(0);
  const wasOpenRef = useRef(isOpen);

  // Advance the key on close so reopening mid-exit mounts fresh native resize geometry.
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
          // The mobile sheet covers, not closes, the panel, so position and size survive.
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
