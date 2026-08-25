// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  useMonitorFrameStore,
  useMonitorOverlayStore,
} from "@/features/settings";
import { resolveGpuVramUsedGb } from "@/hooks/gpu-vram";
import { aggregateGpuMemoryTotalGb, useSystemInfo } from "@/hooks/use-system";
import { useT } from "@/i18n";
import {
  useFloatingPanelOrderStore,
  useFloatingPanelZIndex,
} from "@/lib/floating-panel-order";
import { cn } from "@/lib/utils";
import { CpuIcon, GripVerticalIcon, XIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import {
  type PointerEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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

function useMonitorLayout(constraintsElement: HTMLDivElement | null) {
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
      // Clamp position against the height actually rendered. A hand-resized
      // panel keeps its own height and scrolls, so growing content must not
      // drag it upwards and leave a gap below.
      const height = Math.min(
        monitor.style.height ? monitorBox.height : desiredHeight,
        constraintsBox.height,
      );
      const maxLeft = Math.max(0, constraintsBox.width - width);
      const maxTop = Math.max(0, constraintsBox.height - height);
      const currentLeft = monitorBox.left - constraintsBox.left;
      const currentTop = monitorBox.top - constraintsBox.top;
      const left = place(hasDraggedRef.current, currentLeft, maxLeft);
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
      useMonitorFrameStore.getState().setFrame(publisher, {
        left: monitorBox.left,
        top: monitorBox.top,
        right: monitorBox.right,
        bottom: monitorBox.bottom,
      });

      setLayout((current) => {
        // Mid-drag the offset lives in a transform, and the measured box
        // already includes it, so committing left/top here would apply it
        // twice. finishDrag lands the position instead.
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

  // ResizeObserver never fires for a position-only change, so dragging alone
  // would leave the published frame at the monitor's old corner and the overlay
  // stack dodging where it used to be. Re-publish once each layout is committed,
  // which after a drag is on release: the frames in between are a transform, and
  // republishing through them would re-render every overlay in the stack for
  // each one, which is most of what made dragging feel heavy.
  useLayoutEffect(() => {
    const monitor = monitorRef.current;
    if (!(monitor && constraintsElement)) {
      return;
    }
    const box = monitor.getBoundingClientRect();
    useMonitorFrameStore.getState().setFrame(publisher, {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
    });
  }, [layout, constraintsElement, publisher]);

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

  // One paint per frame, and through a transform rather than left/top. The
  // panel is backdrop-blurred, so every layout-driven move re-sampled what is
  // behind it; a trackpad also reports moves faster than the display refreshes,
  // so most of those renders were never shown.
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
  onClose: () => void;
  systemInfo: ReturnType<typeof useSystemInfo>;
}

function FloatingMonitorPanel({
  onClose,
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
  } = useMonitorLayout(constraintsElement);

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
  const inferenceVramTotal = separateInferenceGpu
    ? aggregateGpuMemoryTotalGb(separateInferenceGpu.devices)
    : 0;
  const devices = displayedGpu?.devices ?? [];
  const vramTotal = aggregateGpuMemoryTotalGb(devices);
  // null usage = unknown (e.g. Windows ROCm perf counter); 0 would fabricate a
  // readout. The host figure can still be known when no device's is (#7452).
  const resolvedVramUsed = resolveGpuVramUsedGb(displayedGpu);
  const vramUsageKnown = resolvedVramUsed !== null;
  const vramUsed = resolvedVramUsed ?? 0;
  const vramPercent = clampPercent(
    vramUsageKnown && vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0,
  );
  const unknownLabel = t("settings.resources.environment.unknown");

  const hasGpu = (displayedGpu?.available ?? false) && devices.length > 0;

  // The container sits on the floating panel layer, above the bottom-right
  // overlay stack. The stack is anchored to that same corner and does not move
  // for this monitor, so the two can overlap. The stack is passive status; this
  // is a window being dragged, resized and closed, so it wins. Still below the
  // startup screen and tooltips. See lib/z-layers.
  //
  // The API monitor panel shares this layer rather than sitting under it, and
  // whichever of the two the user touched last is the one in front.
  return (
    <div
      ref={setConstraintsElement}
      className="pointer-events-none fixed inset-4"
      style={{ zIndex }}
    >
      <motion.div
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

            {hasGpu && (
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
                  {formatGiB(vramTotal)}
                </div>
                <Progress
                  value={vramUsageKnown ? vramPercent : 0}
                  className="mt-1 h-1.5 rounded-full bg-muted"
                  indicatorClassName={usageIndicatorClass(vramPercent)}
                />
              </div>
            )}
            {separateInferenceGpu && (
              <div className="flex justify-between gap-2 text-ui-11 font-mono">
                <span className="text-muted-foreground">
                  {t("settings.resources.gpu.ggufInference")}
                </span>
                <span className="uppercase text-foreground">
                  {separateInferenceGpu.backend ?? "GPU"}
                  {separateInferenceGpu.available
                    ? inferenceVramTotal
                      ? ` · ${formatGiB(inferenceVramTotal)}`
                      : ""
                    : ` · ${t("settings.resources.gpu.unavailable")}`}
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
  const systemInfo = useSystemInfo({ enabled: isOpen, pollMs: 5000 });
  const [panelKey, setPanelKey] = useState(0);
  const wasOpenRef = useRef(isOpen);

  // Each visible panel owns native inline resize state. Advance the key on
  // close so reopening during the exit animation still mounts fresh geometry.
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      setPanelKey((current) => current + 1);
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <FloatingMonitorPanel
          key={panelKey}
          systemInfo={systemInfo}
          onClose={() => setIsOpen(false)}
        />
      )}
    </AnimatePresence>
  );
}
