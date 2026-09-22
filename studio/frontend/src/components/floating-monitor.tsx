// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { FIND_PORTAL_ATTRIBUTE } from "@/features/find-in-page/lib/find-attributes";

import {
  useMonitorFrameStore,
  useMonitorOverlayStore,
} from "@/features/settings";
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
import {
  type PointerEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import {
  FLOATING_MONITOR_WIDTH,
  floatingMonitorConstraintStyle,
  getFloatingMonitorLayout,
} from "./floating-monitor-layout";

interface MonitorLayout {
  left: number;
  top: number;
  minWidth: number;
  minHeight: number;
  maxWidth: number;
  maxHeight: number;
}

interface DragSession {
  pointerId: number;
  startX: number;
  startY: number;
  left: number;
  top: number;
  maxLeft: number;
  maxTop: number;
  constraintsWidth: number;
  constraintsHeight: number;
  /** The committed left/top the drag's transform offsets from. */
  baseLeft: number;
  baseTop: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Anchored to the far edge until the user drags, then clamped where they left it.
function place(dragged: boolean, current: number, max: number): number {
  return dragged ? clamp(current, 0, max) : max;
}

function sameLayout(a: MonitorLayout, b: MonitorLayout): boolean {
  return (
    a.left === b.left &&
    a.top === b.top &&
    a.minWidth === b.minWidth &&
    a.minHeight === b.minHeight &&
    a.maxWidth === b.maxWidth &&
    a.maxHeight === b.maxHeight
  );
}

// Height the panel wants. Reading the rendered box instead hides growth once
// maxHeight caps it, so the observer never fires and the cap is never lifted.
function desiredPanelHeight(
  renderedHeight: number,
  scroll: HTMLDivElement | null,
  content: HTMLDivElement | null,
): number {
  if (!(scroll && content)) {
    return renderedHeight;
  }
  // The scroll region is the only flexible child, so the rest is fixed chrome.
  const chrome = renderedHeight - scroll.getBoundingClientRect().height;
  return chrome + content.getBoundingClientRect().height;
}

// Width the panel wants. While anchored, maxWidth equals the current width, so
// the cap is also a floor: a monitor opened in a narrow window never widens
// again. Lift the cap for one measurement to break that.
function naturalWidth(monitor: HTMLDivElement): number {
  const capped = monitor.style.maxWidth;
  // "none", not "", so the class-level max-w-full lifts too.
  monitor.style.maxWidth = "none";
  const width = monitor.getBoundingClientRect().width;
  monitor.style.maxWidth = capped;
  return width;
}

function useMonitorLayout(
  constraintsElement: HTMLDivElement | null,
  narrowed: boolean,
  hidden: boolean,
) {
  // This panel's claim on the shared frame. Reopening the monitor mid-exit
  // mounts the replacement while the old panel is still animating out, and the
  // old one unmounts last, so its cleanup must only clear its own frame.
  const publisher = useMemo(() => ({}), []);
  const monitorRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const dragSessionRef = useRef<DragSession | null>(null);
  const dragFrameRef = useRef(0);
  const hasDraggedRef = useRef(false);
  const preferredWidthRef = useRef<number | null>(null);
  const preferredHeightRef = useRef<number | null>(null);
  const surfaceWidthRef = useRef(0);
  const narrowedRef = useRef(narrowed);
  const hiddenRef = useRef(hidden);
  // The user's own placement while the container is at full width. Cleared when
  // they drag while it is narrowed, which is a newer choice, not a clamp.
  const chosenLeftRef = useRef<number | null>(null);
  const restoreLeftRef = useRef<number | null>(null);
  const remeasureRef = useRef(0);
  const [layout, setLayout] = useState<MonitorLayout | null>(null);

  useLayoutEffect(() => {
    const monitor = monitorRef.current;
    const constraints = constraintsElement;
    if (!(monitor && constraints)) {
      return;
    }
    // Deferred to the next frame: writing a style while ResizeObserver entries
    // are delivered makes Firefox report an observer loop. A hand-resized panel
    // keeps the user's width instead of re-measuring.
    const scheduleWidthRemeasure = (surfaceWidth: number) => {
      if (surfaceWidth === surfaceWidthRef.current) {
        return;
      }
      surfaceWidthRef.current = surfaceWidth;
      if (monitor.style.width || remeasureRef.current) {
        return;
      }
      remeasureRef.current = requestAnimationFrame(() => {
        remeasureRef.current = 0;
        preferredWidthRef.current = naturalWidth(monitor);
        reconcileGeometry();
      });
    };

    const reconcileGeometry = () => {
      const constraintsBox = constraints.getBoundingClientRect();
      const monitorBox = monitor.getBoundingClientRect();
      const desiredHeight = desiredPanelHeight(
        monitorBox.height,
        scrollRef.current,
        contentRef.current,
      );

      scheduleWidthRemeasure(constraintsBox.width);
      const desiredWidth = Math.max(
        monitorBox.width,
        preferredWidthRef.current ?? monitorBox.width,
      );

      // Content height is the floor, as a resolved number: intrinsic
      // min-content outranks max-height, a number clamped to it cannot.
      if (!monitor.style.height) {
        preferredHeightRef.current = desiredHeight;
      }

      const width = Math.min(desiredWidth, constraintsBox.width);
      // Clamp position against the height actually rendered. A hand-resized panel keeps its own
      // height and scrolls, so growing content must not drag it upwards and leave a gap below.
      const height = Math.min(
        monitor.style.height ? monitorBox.height : desiredHeight,
        constraintsBox.height,
      );
      const maxLeft = Math.max(0, constraintsBox.width - width);
      const maxTop = Math.max(0, constraintsBox.height - height);
      const currentLeft = monitorBox.left - constraintsBox.left;
      const currentTop = monitorBox.top - constraintsBox.top;
      // A restored position is a deliberate left the constraint had clamped
      // away, so it replaces `place()` for exactly one pass.
      const restoreTo = restoreLeftRef.current;
      const left =
        restoreTo === null
          ? place(hasDraggedRef.current, currentLeft, maxLeft)
          : clamp(restoreTo, 0, maxLeft);
      restoreLeftRef.current = null;
      if (!narrowedRef.current && hasDraggedRef.current) {
        chosenLeftRef.current = left;
      }
      const top = place(hasDraggedRef.current, currentTop, maxTop);

      const session = dragSessionRef.current;
      if (session) {
        session.left = left;
        session.top = top;
        session.maxLeft = maxLeft;
        session.maxTop = maxTop;
        session.constraintsWidth = constraintsBox.width;
        session.constraintsHeight = constraintsBox.height;
      }

      // Publish the real box so the overlay stack can keep clear of it.
      if (!hiddenRef.current) {
        useMonitorFrameStore.getState().setFrame(publisher, {
          left: monitorBox.left,
          top: monitorBox.top,
          right: monitorBox.right,
          bottom: monitorBox.bottom,
        });
      }

      setLayout((current) => {
        // Mid-drag the offset lives in a transform, and the measured box already includes it, so
        // committing left/top here would apply it twice. finishDrag lands the position instead.
        const held = session && current ? current : null;
        const restLeft = held?.left ?? left;
        const restTop = held?.top ?? top;
        const next = {
          left: restLeft,
          top: restTop,
          minWidth: preferredWidthRef.current ?? monitorBox.width,
          minHeight: preferredHeightRef.current ?? monitorBox.height,
          maxWidth: constraintsBox.width - restLeft,
          maxHeight: constraintsBox.height - restTop,
        };
        return current && sameLayout(current, next) ? current : next;
      });
    };

    reconcileGeometry();
    const observer = new ResizeObserver(reconcileGeometry);
    observer.observe(constraints);
    observer.observe(monitor);
    // The unclamped content wrapper is what makes late GPU rows reposition the
    // panel instead of being cut off.
    if (contentRef.current) {
      observer.observe(contentRef.current);
    }
    return () => {
      observer.disconnect();
      useMonitorFrameStore.getState().clearFrame(publisher);
      if (remeasureRef.current) {
        cancelAnimationFrame(remeasureRef.current);
        remeasureRef.current = 0;
      }
      if (dragFrameRef.current) {
        cancelAnimationFrame(dragFrameRef.current);
        dragFrameRef.current = 0;
      }
    };
  }, [constraintsElement, publisher]);

  // Narrowing clamps the monitor left, and `place()` keeps the clamped spot.
  // The position the user did drag to is put back when the container widens.
  // Settled in a layout effect, before the next observation can reconcile.
  // The API monitor treats any published frame as a live obstacle, so an
  // invisible resource monitor must not keep publishing its box. Visibility and
  // aria-hidden fire no ResizeObserver, so `hidden` also feeds the republish
  // below: it is what restores the box once the monitor is on screen again.
  useLayoutEffect(() => {
    hiddenRef.current = hidden;
    if (hidden) {
      useMonitorFrameStore.getState().clearFrame(publisher);
    }
  }, [hidden, publisher]);

  useLayoutEffect(() => {
    if (narrowedRef.current === narrowed) {
      return;
    }
    narrowedRef.current = narrowed;
    if (!narrowed) {
      restoreLeftRef.current = chosenLeftRef.current;
    }
  }, [narrowed]);

  // ResizeObserver never fires for a position-only change, so dragging alone would leave the
  // published frame at the monitor's old corner and the overlay stack dodging where it used to be.
  // Re-publish once each layout is committed, which after a drag is on release: the frames in
  // between are a transform, and republishing through them would re-render every overlay in the
  // stack for each one, which is most of what made dragging feel heavy.
  useLayoutEffect(() => {
    void layout;
    const monitor = monitorRef.current;
    if (!(monitor && constraintsElement) || hiddenRef.current) {
      return;
    }
    const box = monitor.getBoundingClientRect();
    useMonitorFrameStore.getState().setFrame(publisher, {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
    });
  }, [layout, constraintsElement, publisher, hidden]);

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    const monitor = monitorRef.current;
    if (event.button !== 0 || !(monitor && constraintsElement)) {
      return;
    }

    event.preventDefault();
    const constraintsBox = constraintsElement.getBoundingClientRect();
    const monitorBox = monitor.getBoundingClientRect();
    const left = monitorBox.left - constraintsBox.left;
    const top = monitorBox.top - constraintsBox.top;
    hasDraggedRef.current = true;
    if (narrowedRef.current) {
      chosenLeftRef.current = null;
    }

    // Native resize records attempted inline dimensions even when max-width
    // or max-height hides them. Normalize only hidden dimensions so an
    // auto-sized monitor can still grow when system rows arrive later.
    const inlineWidth = Number.parseFloat(monitor.style.width);
    const inlineHeight = Number.parseFloat(monitor.style.height);
    if (
      Number.isFinite(inlineWidth) &&
      Math.abs(inlineWidth - monitorBox.width) > 0.5
    ) {
      monitor.style.width = `${monitorBox.width}px`;
    }
    if (
      Number.isFinite(inlineHeight) &&
      Math.abs(inlineHeight - monitorBox.height) > 0.5
    ) {
      monitor.style.height = `${monitorBox.height}px`;
    }

    dragSessionRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left,
      top,
      maxLeft: Math.max(0, constraintsBox.width - monitorBox.width),
      maxTop: Math.max(0, constraintsBox.height - monitorBox.height),
      constraintsWidth: constraintsBox.width,
      constraintsHeight: constraintsBox.height,
      baseLeft: left,
      baseTop: top,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  // One paint per frame, and through a transform rather than left/top. The panel is
  // backdrop-blurred, so every layout-driven move re-sampled what is behind it; a trackpad also
  // reports moves faster than the display refreshes, so most of those renders were never shown.
  function paintDrag() {
    dragFrameRef.current = 0;
    const session = dragSessionRef.current;
    const monitor = monitorRef.current;
    if (!(session && monitor)) {
      return;
    }
    monitor.style.transform = `translate3d(${session.left - session.baseLeft}px, ${
      session.top - session.baseTop
    }px, 0)`;
  }

  function updateDrag(event: PointerEvent<HTMLDivElement>) {
    const session = dragSessionRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }

    const left = clamp(
      session.left + event.clientX - session.startX,
      0,
      session.maxLeft,
    );
    const top = clamp(
      session.top + event.clientY - session.startY,
      0,
      session.maxTop,
    );
    session.startX = event.clientX;
    session.startY = event.clientY;
    session.left = left;
    session.top = top;
    if (!dragFrameRef.current) {
      dragFrameRef.current = requestAnimationFrame(paintDrag);
    }
  }

  function finishDrag(event: PointerEvent<HTMLDivElement>) {
    const session = dragSessionRef.current;
    if (session?.pointerId !== event.pointerId) {
      return;
    }
    if (dragFrameRef.current) {
      cancelAnimationFrame(dragFrameRef.current);
      dragFrameRef.current = 0;
    }
    const { left, top, constraintsWidth, constraintsHeight } = session;
    dragSessionRef.current = null;
    // A position-only change does not fire the observer, and the next
    // reconcile may already be narrowed.
    if (!narrowedRef.current) {
      chosenLeftRef.current = left;
    }
    // Written to the node as well as to state, in this order, so handing the
    // offset back to left/top cannot show a frame at the spot it started from.
    const monitor = monitorRef.current;
    if (monitor) {
      monitor.style.left = `${left}px`;
      monitor.style.top = `${top}px`;
      monitor.style.transform = "";
    }
    setLayout((current) =>
      !current || (current.left === left && current.top === top)
        ? current
        : {
            ...current,
            left,
            top,
            maxWidth: constraintsWidth - left,
            maxHeight: constraintsHeight - top,
          },
    );
  }

  return {
    monitorRef,
    scrollRef,
    contentRef,
    layout,
    startDrag,
    updateDrag,
    finishDrag,
  };
}

function clampPercent(value: number): number {
  return clamp(value, 0, 100);
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
}
function FloatingMonitorPanel({
  dockedBesideRunSettings,
  onClose,
  onRenderedWidth,
  settingsWidth,
  suppressed,
  systemInfo,
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
  } = useMonitorLayout(
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
          "settings-surface pointer-events-auto absolute flex max-h-full w-64 max-w-full cursor-default select-none flex-col overflow-hidden rounded-xl border border-border/70 p-3 shadow-border ring-0 backdrop-blur-sm",
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
  useEffect(() => {
    const sidebar = document.querySelector('[data-slot="sidebar"]');
    if (!sidebar) {
      setPaintedSidebarWidth(0);
      return;
    }
    const measure = () => {
      if (sidebar.isConnected) {
        setPaintedSidebarWidth(sidebar.getBoundingClientRect().width);
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
  const sidebarWidth =
    paintedSidebarWidth > 0 ? paintedSidebarWidth : committedSidebarWidth;
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
      // A pinned sidebar holds its column; an unpinned one overlays the content
      // or collapses to an icon rail, so neither takes the monitor's room.
      sidebarWidth: pinned ? sidebarWidth : 0,
      viewportWidth,
      monitorWidth,
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
          onRenderedWidth={setMonitorWidth}
          systemInfo={systemInfo}
          onClose={() => setIsOpen(false)}
        />
      )}
    </AnimatePresence>
  );
}
