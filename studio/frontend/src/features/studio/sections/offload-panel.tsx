// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { authFetch } from "@/features/auth";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { RamMemoryIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import {
  type OffloadCard,
  groupLayersByCard,
  layerPlacement,
  vramUsage,
} from "./offload-panel-layout";

/** `GET /api/train/offload`: unsloth_zoo BlockSwap.stats() from the last logged step. */
export interface OffloadState {
  active: boolean;
  total_layers?: number;
  swapped?: number[];
  state?: Record<string, "gpu" | "host" | "copying" | "held">;
  prefetch_depth?: number;
  auto_depth?: boolean;
  depth_settled?: boolean;
  host_bytes?: number;
  pinned_bytes?: number;
  pool_bytes?: number;
  timing?: boolean;
  copies?: number;
  copy_ms?: number;
  stall_ms?: number;
  compute_ms?: number;
  layers?: number;
  vram_allocated_bytes?: number;
  vram_peak_bytes?: number;
  vram_total_bytes?: number;
  vram_fraction?: number;
  /** Every visible card, and the torch ordinal each decoder layer runs on. */
  vram_devices?: OffloadCard[];
  layer_device?: Record<string, number>;
}

const GIB = 1024 ** 3;
// One sweep of the window takes about this long on screen; real layers run in milliseconds.
const SWEEP_MS = 4000;

export function useOffloadState(enabled: boolean, intervalMs = 2000): OffloadState {
  const [data, setData] = useState<OffloadState>({ active: false });
  useEffect(() => {
    if (!enabled) {
      setData({ active: false });
      return;
    }
    let cancelled = false;
    async function poll() {
      try {
        const res = await authFetch("/api/train/offload");
        if (!res.ok || cancelled) return;
        const json = (await res.json()) as OffloadState;
        if (!cancelled) setData(json);
      } catch {
        // Retry on the next poll.
      }
    }
    void poll();
    const timer = setInterval(() => void poll(), intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [enabled, intervalMs]);
  return data;
}

function useSweep(count: number): number {
  const [pos, setPos] = useState(0);
  useEffect(() => {
    if (count <= 0) return;
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      setPos(Math.floor(((now - start) % SWEEP_MS) / (SWEEP_MS / count)));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [count]);
  return pos;
}

const CELL = {
  resident: "bg-control-accent/70",
  running: "bg-control-accent ring-2 ring-control-accent/40",
  copying: "bg-amber-500/80",
  host: "border border-border bg-transparent",
} as const;

function Stat({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl bg-muted/50 px-3 py-2">
      <span className="text-ui-11 text-muted-foreground">{label}</span>
      <span className="font-mono text-sm">{value}</span>
    </div>
  );
}

function VramBar({
  label,
  used,
  budget,
  total,
}: { label: string } & ReturnType<typeof vramUsage>): ReactElement {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-ui-11 text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono">
          {(used / GIB).toFixed(1)} / {((budget ?? total) / GIB).toFixed(1)} GiB
        </span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-control-accent"
          style={{ width: `${Math.min(100, (100 * used) / total)}%` }}
        />
        {budget != null && (
          <div
            className="absolute inset-y-0 w-0.5 bg-[color-mix(in_oklab,var(--foreground)_calc(70%*var(--contrast-wash-gain,1)),transparent)]"
            style={{ left: `${(100 * budget) / total}%` }}
          />
        )}
      </div>
    </div>
  );
}

function LayerGrid({
  cells,
  layers,
  label,
}: {
  cells: ReturnType<typeof layerPlacement>;
  layers: number[];
  label: string;
}): ReactElement {
  return (
    <div className="grid w-fit grid-cols-8 gap-1" aria-label={label}>
      {layers.map((layer) => (
        <div
          key={layer}
          title={`${layer}`}
          className={cn("size-4 rounded-[3px] transition-colors", CELL[cells[layer]])}
        />
      ))}
    </div>
  );
}

export function OffloadPanel({ isTrainingRunning }: { isTrainingRunning: boolean }): ReactElement | null {
  const t = useT();
  const s = useOffloadState(isTrainingRunning);
  const swapped = s.swapped ?? [];
  const total = s.total_layers ?? 0;
  const depth = s.prefetch_depth ?? 2;
  const pos = useSweep(swapped.length);
  if (!s.active || total === 0) return null;

  const cells = layerPlacement(total, swapped, depth, pos);
  const perCopy = s.copies ? (s.copy_ms ?? 0) / s.copies : null;
  const perLayer = s.layers ? (s.compute_ms ?? 0) / s.layers : null;
  const busy = (s.compute_ms ?? 0) + (s.stall_ms ?? 0);
  const waiting = busy > 0 ? (100 * (s.stall_ms ?? 0)) / busy : null;
  const vramTotal = s.vram_total_bytes ?? 0;
  const cards = s.vram_devices ?? [];
  const multi = cards.length > 1;
  const ms = (v: number | null) => (v == null ? "--" : `${v.toFixed(1)} ms`);

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={RamMemoryIcon} className="size-5" />}
      title={t("studio.params.offloadPanelTitle")}
      description={`${swapped.length} / ${total} ${t("studio.params.offloadPanelSwapped")} · ${t(
        "studio.params.offloadPanelDepth",
      )} ${depth}${s.auto_depth ? ` (${t("studio.params.offloadAuto")})` : ""}`}
      className="shadow-border border border-border/60 bg-card/90 ring-0 backdrop-blur-sm"
    >
      <div className="flex flex-wrap items-start gap-6">
        {multi ? (
          <div className="flex flex-col gap-4">
            {groupLayersByCard(total, s.layer_device, cards).map(({ card, layers }) => {
              const name = card
                ? t("studio.params.offloadPanelCard", { index: card.gpu_id ?? card.index })
                : t("studio.params.offloadPanelCardUnknown");
              return (
                <div key={card?.index ?? "unplaced"} className="flex w-44 flex-col gap-2">
                  <div className="flex items-baseline gap-1.5 text-ui-11">
                    <span className="font-medium">{name}</span>
                    {card?.name && (
                      <span className="truncate text-muted-foreground" title={card.name}>
                        {card.name}
                      </span>
                    )}
                  </div>
                  <LayerGrid cells={cells} layers={layers} label={name} />
                  {card && (card.total_bytes ?? 0) > 0 && (
                    <VramBar
                      label={t("studio.params.offloadPanelVram")}
                      {...vramUsage(card.total_bytes, card.peak_bytes, card.allocated_bytes, card.fraction)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="grid grid-cols-8 gap-1" aria-label={t("studio.params.offloadPanelTitle")}>
            {cells.map((cell, layer) => (
              <div
                // biome-ignore lint/suspicious/noArrayIndexKey: one cell per decoder layer, fixed order
                key={layer}
                title={`${layer}`}
                className={cn("size-4 rounded-[3px] transition-colors", CELL[cell])}
              />
            ))}
          </div>
        )}
        <div className="flex min-w-48 flex-1 flex-col gap-3">
          <div className="flex flex-wrap gap-3 text-ui-11 text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className={cn("size-2.5 rounded-[2px]", CELL.resident)} />
              {t("studio.params.offloadPanelGpu")}
            </span>
            <span className="flex items-center gap-1.5">
              <span className={cn("size-2.5 rounded-[2px]", CELL.copying)} />
              {t("studio.params.offloadPanelCopying")}
            </span>
            <span className="flex items-center gap-1.5">
              <span className={cn("size-2.5 rounded-[2px]", CELL.host)} />
              {t("studio.params.offloadPanelHost")}
            </span>
          </div>
          {!multi && vramTotal > 0 && (
            <VramBar
              label={t("studio.params.offloadPanelVram")}
              {...vramUsage(vramTotal, s.vram_peak_bytes, s.vram_allocated_bytes, s.vram_fraction)}
            />
          )}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label={t("studio.params.offloadPanelCopy")} value={ms(perCopy)} />
            <Stat label={t("studio.params.offloadPanelCompute")} value={ms(perLayer)} />
            <Stat
              label={t("studio.params.offloadPanelStall")}
              value={waiting == null ? "--" : `${waiting.toFixed(1)}%`}
            />
            <Stat
              label={t("studio.params.offloadPanelPinned")}
              value={`${((s.pinned_bytes ?? 0) / GIB).toFixed(1)} / ${((s.host_bytes ?? 0) / GIB).toFixed(1)} GiB`}
            />
          </div>
          <p className="text-ui-11 text-muted-foreground">{t("studio.params.offloadPanelSweepNote")}</p>
        </div>
      </div>
    </SectionCard>
  );
}
