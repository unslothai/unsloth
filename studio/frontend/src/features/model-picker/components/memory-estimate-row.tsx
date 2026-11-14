// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SystemGpuDevice } from "@/hooks/gpu-selection";
import type { GpuMemoryEstimate, MemoryEstimate } from "../api/memory-estimate";
import {
  type MemoryFitVerdict,
  NOTE_SEPARATOR,
  formatMemoryGb,
  glueNoteItems,
  memoryFigureCandidates,
  mlxOffloadPercent,
  resolveDraftCacheNote,
  resolveKvNote,
  resolveMemoryFit,
} from "../model-config/memory-fit";

const MEMORY_VALUE_TONE: Record<MemoryFitVerdict, string> = {
  fits: "text-foreground",
  tight: "text-amber-500",
  exceeds: "text-red-500",
  unknown: "text-foreground",
};

function MemoryFigure({
  label,
  bytes,
  bounded,
  tone,
}: {
  label: string;
  bytes: number;
  bounded: boolean;
  tone?: string;
}) {
  const candidates = useMemo(
    () => memoryFigureCandidates(bytes, bounded),
    [bytes, bounded],
  );
  const value = candidates[0];
  const [displayIndex, setDisplayIndex] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const measureRef = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const button = buttonRef.current;
    const measurement = measureRef.current;
    const row = button?.parentElement;
    if (!button || !measurement || !row) return;
    let active = true;
    const fit = () => {
      if (!active) return;
      // Measured on the row: the pill and label shrink to the figure and would confirm the first pass.
      const style = getComputedStyle(button);
      const gutters =
        Number.parseFloat(style.paddingLeft) +
        Number.parseFloat(style.paddingRight);
      const rowStyle = getComputedStyle(row);
      const gap = Number.parseFloat(rowStyle.columnGap) || 0;
      const siblings = Array.from(row.children).filter(
        (child) => child !== button,
      );
      const room =
        row.clientWidth -
        Number.parseFloat(rowStyle.paddingLeft) -
        Number.parseFloat(rowStyle.paddingRight) -
        gap * siblings.length -
        siblings.reduce(
          (total, child) =>
            total +
            Math.max(child.getBoundingClientRect().width, child.scrollWidth),
          0,
        ) -
        gutters;
      const index = Array.from(measurement.children).findIndex(
        (child) => child.getBoundingClientRect().width <= room + 0.5,
      );
      setDisplayIndex(index < 0 ? candidates.length - 1 : index);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(row);
    observer.observe(measurement);
    void document.fonts.ready.then(fit);
    return () => {
      active = false;
      observer.disconnect();
    };
  }, [candidates]);
  return (
    <div className="flex min-h-8 min-w-0 items-center justify-between gap-3">
      <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
        {label}
      </span>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            ref={buttonRef}
            type="button"
            aria-label={`${label}: ${value}`}
            className={`relative inline-flex h-8 w-auto min-w-[calc(64px*var(--ui-space-scale,1))] max-w-full shrink-0 cursor-default! items-center justify-center overflow-hidden rounded-full border-transparent bg-[var(--panel-input-surface)] px-3.5 text-ui-13 font-medium leading-none tabular-nums focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 ${tone ?? "text-foreground"}`}
          >
            <span aria-hidden="true" className="min-w-0 truncate">
              {candidates[displayIndex] ?? value}
            </span>
            <span
              ref={measureRef}
              aria-hidden="true"
              className="pointer-events-none invisible absolute left-0 top-0 flex w-max flex-col items-start whitespace-nowrap"
            >
              {candidates.map((candidate) => (
                <span key={candidate}>{candidate}</span>
              ))}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent>{value}</TooltipContent>
      </Tooltip>
    </div>
  );
}

function MemoryBreakdownLine({
  label,
  value,
  note,
  muted,
  exceeds,
}: {
  label: string;
  value: string;
  note?: string;
  muted?: boolean;
  exceeds?: boolean;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-1">
      <span className="min-w-0 text-ui-12 text-muted-foreground">{label}</span>
      <span
        className={`whitespace-nowrap pr-2 text-ui-12 tabular-nums ${exceeds ? MEMORY_VALUE_TONE.exceeds : muted ? "text-muted-foreground" : "text-foreground"}`}
      >
        {value}
      </span>
      {note ? (
        <span className="col-span-2 min-w-0 text-ui-11 leading-relaxed text-muted-foreground/80">
          {glueNoteItems(note)}
        </span>
      ) : null}
    </div>
  );
}

function gpuPlacementRow(
  estimate: GpuMemoryEstimate,
  device: SystemGpuDevice | undefined,
  residentGpuIds: readonly number[] | null | undefined,
) {
  const { gpuId, weightsBytes, cacheBytes, layerCount, outputLayer } = estimate;
  const subtotal = weightsBytes + (cacheBytes ?? 0);
  const gib = 1024 ** 3;
  // An APU's reported VRAM may be a reserved window onto host memory.
  const dedicated = device && !device.unifiedMemory && !device.sharedMemory;
  const total = dedicated && device.memoryTotalGb > 0 ? device.memoryTotalGb * gib : null;
  const free = dedicated && (device.memoryFreeKnown ?? device.memoryFreeGb > 0)
    ? device.memoryFreeGb * gib
    : null;
  const exceedsTotal = total !== null && subtotal > total;
  // Free VRAM on a resident GPU includes no credit for memory released by reload.
  const mayReleaseMemory = residentGpuIds === null || residentGpuIds?.includes(gpuId);
  const exceedsFree = !mayReleaseMemory && free !== null && subtotal > free;
  const name = `GPU ${gpuId}`;
  const details = [
    `${layerCount} repeating ${layerCount === 1 ? "layer" : "layers"}`,
    outputLayer ? "output layer" : null,
    `${formatMemoryGb(weightsBytes)} weights`,
    cacheBytes === null ? "cache unknown" : `${formatMemoryGb(cacheBytes)} cache`,
    free !== null ? `${formatMemoryGb(free)} free` : null,
    total !== null ? `${formatMemoryGb(total)} total` : null,
  ];
  return {
    gpuId,
    name,
    label: total === null ? name : `${name} · ${Number((total / gib).toFixed(2))} GiB`,
    deviceLabel: device?.name && device.name !== name ? `${name} · ${device.name}` : name,
    value: cacheBytes === null ? memoryFigureCandidates(subtotal, true)[0] : formatMemoryGb(subtotal),
    note: details.filter(Boolean).join(NOTE_SEPARATOR),
    exceedsTotal,
    exceedsFree,
  };
}

function gpuLayersNote(estimate: MemoryEstimate): string | null {
  if (estimate.perGpu.length > 0) {
    // Some GGUFs declare unused draft slots in layerCount. Count the blocks placed on devices.
    const layers = estimate.perGpu.reduce((sum, gpu) => sum + gpu.layerCount, 0);
    const output = estimate.perGpu.some((gpu) => gpu.outputLayer);
    return `${layers} repeating ${layers === 1 ? "layer" : "layers"}${output ? " and the output layer" : ""} on GPU`;
  }
  if (estimate.gpuLayers === null || estimate.gpuLayers < 0 || estimate.layerCount === null) return null;
  const slots = estimate.layerCount + 1;
  return `${Math.min(estimate.gpuLayers, slots)} of ${slots} layers on GPU`;
}

export function MemoryEstimateRow({
  estimate,
  loading,
  stale,
  gpuDevices,
  residentGpuIds,
  placementMode,
  gpuCapacityGb,
  totalCapacityGb,
  systemRamCapacityGb,
  freeGpuCapacityGb,
  freeGpuCapacityKnown,
  freeGpuReserveDeficitGb,
  usableSystemRamGb,
  usableSystemRamKnown,
  systemRamReserveDeficitGb,
  isUnifiedMemory,
  singleMemoryPool,
  reclaimableTotalBytes,
  reclaimableGpuBytes,
  expanded,
  onExpandedChange,
}: {
  estimate: MemoryEstimate | null;
  loading: boolean;
  stale: boolean;
  gpuDevices?: SystemGpuDevice[];
  /** Reloading devices, null when their identities are unknown, undefined for a fresh load. */
  residentGpuIds?: readonly number[] | null;
  placementMode?: "auto" | "manual";
  /** GiB; 0 means unknown. */
  gpuCapacityGb: number;
  /** GiB; 0 means unknown. */
  totalCapacityGb: number;
  systemRamCapacityGb: number;
  /** Warnings only, since replacement can free memory. */
  freeGpuCapacityGb: number;
  freeGpuCapacityKnown?: boolean;
  freeGpuReserveDeficitGb?: number;
  usableSystemRamGb: number;
  usableSystemRamKnown?: boolean;
  systemRamReserveDeficitGb?: number;
  isUnifiedMemory: boolean;
  singleMemoryPool: boolean;
  reclaimableTotalBytes?: number;
  reclaimableGpuBytes?: number;
  expanded: boolean;
  onExpandedChange: (next: boolean) => void;
}) {
  const contentId = useId();
  if (!estimate?.available) {
    return null;
  }
  const { gpuFit, totalFit, cpuOnly, bounded, advisory } = resolveMemoryFit(
    estimate,
    {
      gpuCapacityGb,
      totalCapacityGb,
      systemRamCapacityGb,
      freeGpuCapacityGb,
      freeGpuCapacityKnown,
      freeGpuReserveDeficitGb,
      usableSystemRamGb,
      usableSystemRamKnown,
      systemRamReserveDeficitGb,
      singleMemoryPool,
      reclaimableTotalBytes,
      reclaimableGpuBytes,
    },
  );
  const gpuRows = stale
    ? []
    : estimate.perGpu.map((gpu) =>
        gpuPlacementRow(gpu, gpuDevices?.find((device) => device.index === gpu.gpuId), residentGpuIds),
      );
  const overCapacity = gpuRows.find((row) => row.exceedsTotal);
  const pressured = overCapacity ?? gpuRows.find((row) => row.exceedsFree);
  const placementWarning = pressured
    ? `${pressured.name} needs more than its ${overCapacity ? "total" : "currently free"} VRAM.`
    : null;
  const kvNote = resolveKvNote(estimate);
  const weightsNote = [
    gpuLayersNote(estimate),
    estimate.mlxPleOffload ? `PLE ${mlxOffloadPercent(estimate.mlxPleOffload)} on disk` : null,
    estimate.mlxExpertOffload
      ? `experts ${mlxOffloadPercent(estimate.mlxExpertOffload)} on disk`
      : null,
  ]
    .filter(Boolean)
    .join(NOTE_SEPARATOR);
  const draftCacheNote = resolveDraftCacheNote(
    estimate.drafterRuntimeGpuBytes,
    estimate.drafterRuntimeBytes,
  );
  return (
    <div data-memory-estimate="" className="flex flex-col border-b border-border pb-5">
      <button
        type="button"
        onClick={() => onExpandedChange(!expanded)}
        aria-expanded={expanded}
        aria-controls={contentId}
        aria-label={`Estimated Memory Usage: ${expanded ? "Hide" : "Show"} breakdown`}
        className="group flex min-h-8 w-full items-center justify-between gap-3 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground">
            Estimated Memory Usage
          </span>
          <span className="shrink-0 rounded-md bg-[rgb(0_0_0_/_calc(0.04*var(--contrast-wash-gain,1)))] px-1.5 py-0.5 text-ui-10 font-medium uppercase tracking-wide text-muted-foreground dark:bg-muted">
            Beta
          </span>
        </span>
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          aria-hidden="true"
          className={`size-3.5 shrink-0 text-muted-foreground transition-transform duration-200 group-hover:text-foreground motion-reduce:transition-none ${expanded ? "rotate-0" : "-rotate-90"}`}
        />
      </button>
      <div
        aria-busy={loading || stale}
        className={`space-y-3 transition-opacity ${stale || loading ? "opacity-50" : ""}`}
      >
        <MemoryFigure
          label={
            cpuOnly
              ? "RAM"
              : singleMemoryPool
                ? isUnifiedMemory
                  ? "Unified"
                  : "Shared"
                : "GPU"
          }
          bytes={
            singleMemoryPool || cpuOnly
              ? estimate.totalBytes
              : estimate.gpuBytes
          }
          bounded={bounded}
          tone={
            overCapacity
              ? MEMORY_VALUE_TONE.exceeds
              : MEMORY_VALUE_TONE[singleMemoryPool || cpuOnly ? totalFit : gpuFit]
          }
        />
        {singleMemoryPool || cpuOnly ? null : (
          <MemoryFigure
            label="Total"
            bytes={estimate.totalBytes}
            bounded={bounded}
            tone={MEMORY_VALUE_TONE[totalFit]}
          />
        )}
      </div>
      <div id={contentId} hidden={!expanded} className="mt-3 space-y-3">
        <MemoryBreakdownLine
          label="Weights"
          value={formatMemoryGb(estimate.weightsBytes)}
          note={weightsNote || undefined}
        />
        <MemoryBreakdownLine
          label="KV cache"
          value={
            estimate.kvEstimable
              ? formatMemoryGb(estimate.kvBytes - estimate.kvCheckpointBytes)
              : "unknown"
          }
          note={estimate.kvEstimable ? kvNote : undefined}
          muted={!estimate.kvEstimable}
        />
        {estimate.kvEstimable && estimate.kvCheckpointBytes > 0 && (
          <MemoryBreakdownLine
            label="Context checkpoints"
            value={formatMemoryGb(estimate.kvCheckpointBytes)}
            note="host RAM, filled as conversations grow"
          />
        )}
        <MemoryBreakdownLine
          label="Compute buffers"
          value={formatMemoryGb(estimate.computeBytes)}
        />
        {estimate.projectorRuntimeBytes > 0 && (
          <MemoryBreakdownLine
            label="Vision encoder"
            value={formatMemoryGb(estimate.projectorRuntimeBytes)}
          />
        )}
        {/* Streamed: outside Weights and Total, only the rows in use page in. Resident:
            already inside Weights, so the note says so rather than adding it twice. */}
        {(estimate.lookupTableBytes ?? 0) > 0 && (
          <MemoryBreakdownLine
            label="Lookup tables"
            value={formatMemoryGb(estimate.lookupTableBytes ?? 0)}
            note={estimate.lookupTableOnDisk ? "on disk" : "in RAM, part of Weights"}
          />
        )}
        {estimate.drafterRuntimeBytes > 0 && (
          <MemoryBreakdownLine
            label="Draft cache"
            value={formatMemoryGb(estimate.drafterRuntimeBytes)}
            note={draftCacheNote}
          />
        )}
        {gpuRows.length > 0 ? (
          <div className="space-y-2 border-t border-border pt-3">
            <div className="text-ui-12 font-medium text-foreground">Weights + cache per GPU</div>
            {gpuRows.map((row) => (
              <MemoryBreakdownLine
                key={row.gpuId}
                label={row.label}
                value={row.value}
                exceeds={row.exceedsTotal || row.exceedsFree}
              />
            ))}
            <details className="text-ui-11 leading-relaxed text-muted-foreground">
              <summary className="w-fit cursor-pointer rounded-sm focus-visible:outline-ring">
                Layer and memory details
              </summary>
              {gpuRows.map((row) => (
                <p key={row.gpuId} className="mt-2">
                  <span className="font-medium">{row.deviceLabel}</span>
                  <br />
                  {glueNoteItems(row.note)}
                </p>
              ))}
              <p className="mt-2">
                Compute buffers and other runtime allocations need additional memory.
                {estimate.perGpuBasis === "free_memory" && " The split uses currently free VRAM and can change when a model unloads."}
              </p>
            </details>
          </div>
        ) : placementMode && !stale && !cpuOnly ? (
          <p className="text-ui-12 text-muted-foreground">
            {placementMode === "auto"
              ? "Per-GPU placement is determined at load time."
              : "Per-GPU placement is unavailable for these settings."}
          </p>
        ) : null}
      </div>
      {(placementWarning || advisory) && (
        <p
          data-memory-advisory=""
          className={`mt-1.5 text-pretty text-ui-12 leading-relaxed ${placementWarning || advisory?.tone === "warn" ? "text-amber-700 dark:text-amber-400" : "text-muted-foreground"}`}
        >
          {placementWarning ?? advisory?.text}
        </p>
      )}
    </div>
  );
}
