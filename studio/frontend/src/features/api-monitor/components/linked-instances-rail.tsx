// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { useSettingsDialogStore } from "@/features/settings";
import type {
  LinkedInstance,
  LinkedInstanceInfo,
  LinkedInstanceStatus,
} from "@/features/settings/api/linked-instances";
import { LinkedInstanceDetailsDialog } from "@/features/settings/components/linked-instance-details-dialog";
import {
  acceleratorLabel,
  connectionKind,
  formatGb,
  gpuGroups,
  gpuPool,
  vramPercent,
} from "@/features/settings/components/linked-instance-format";
import type { useLinkedInstancesOverview } from "@/features/settings/hooks/use-linked-instances-overview";
import { cn } from "@/lib/utils";
import { RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";

type Overview = ReturnType<typeof useLinkedInstancesOverview>;

const CONNECTION = {
  tunnel: "Cloudflare",
  loopback: "Localhost",
  lan: "LAN",
  public: "Public",
} as const;

function InstanceCard({
  instance,
  status,
  info,
  onOpen,
}: {
  instance: LinkedInstance;
  status: LinkedInstanceStatus | undefined;
  info: LinkedInstanceInfo | undefined;
  onOpen: () => void;
}) {
  const online = status?.online;
  const prefix = `@${instance.name}/`;
  const serving = status?.loaded[0]?.slice(prefix.length);
  const ready = info?.online === true;
  const gpus = ready ? info.gpus : [];
  const pool = gpus.length > 0 ? gpuPool(gpus) : null;
  const pct =
    pool?.used != null && pool.total
      ? Math.min(100, (pool.used / pool.total) * 100)
      : null;
  const footer = [
    ready && info.version ? `Unsloth ${info.version}` : null,
    ready ? acceleratorLabel(info) : null,
    CONNECTION[connectionKind(instance.base_url)],
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <button
      type="button"
      onClick={onOpen}
      title={instance.base_url}
      className="flex min-w-0 flex-col gap-2 rounded-xl border border-border/60 bg-card px-3.5 py-3 text-left transition-colors hover:border-border hover:bg-accent/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
    >
      <div className="flex flex-wrap items-center justify-between gap-x-2">
        <span className="flex items-center gap-1.5 break-all font-mono text-ui-12 font-medium text-foreground">
          <span
            aria-hidden={true}
            className={cn(
              "size-2 shrink-0 rounded-full",
              online === undefined
                ? "animate-pulse bg-muted-foreground/50"
                : online
                  ? "bg-emerald-500"
                  : "bg-red-500",
            )}
          />
          @{instance.name}
        </span>
        {status?.latency_ms != null ? (
          <span className="text-ui-11 tabular-nums text-muted-foreground">
            {status.latency_ms} ms
          </span>
        ) : null}
      </div>

      {online === false ? (
        <p className="text-ui-12 text-destructive">
          {status?.error ?? "Offline"}
        </p>
      ) : !ready ? (
        <div className="flex flex-col gap-1.5">
          <Skeleton className="h-3 w-2/3" />
          <Skeleton className="h-1 w-full" />
        </div>
      ) : (
        <>
          {pool ? (
            <div className="flex flex-col gap-1">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2 text-ui-11">
                <span className="text-foreground">
                  {gpus.length > 1 ? `${gpus.length} GPUs` : gpus[0].name}
                </span>
                {pool.total != null ? (
                  <span className="tabular-nums text-muted-foreground">
                    {pool.used != null ? `${pool.used.toFixed(1)} / ` : ""}
                    {formatGb(pool.total)}
                  </span>
                ) : null}
              </div>
              {pct != null ? (
                <Progress
                  value={pct}
                  className="h-1"
                  indicatorClassName={cn(pct > 90 && "bg-amber-500")}
                />
              ) : null}
              {gpus.length > 1 ? (
                <>
                  <div className="flex flex-col text-ui-10 text-muted-foreground">
                    {gpuGroups(gpus).map(({ name, count }) => (
                      <span key={name}>
                        {count > 1 ? `${count}× ` : ""}
                        {name}
                      </span>
                    ))}
                  </div>
                  {/* One cell per card, so a busy or full card stands out at any count. */}
                  <div
                    className="grid gap-[3px]"
                    // Rows of up to 8, so 4, 8 and 16 cards all land on even rows.
                    style={{
                      gridTemplateColumns: `repeat(${Math.min(gpus.length, 8)}, minmax(0, 1fr))`,
                    }}
                  >
                    {gpus.map((gpu, i) => {
                      const cell = vramPercent(gpu);
                      return (
                        <span
                          // biome-ignore lint/suspicious/noArrayIndexKey: identical cards share a name
                          key={i}
                          title={`GPU ${i} · ${gpu.name}${
                            gpu.vram_total_gb != null
                              ? ` · ${gpu.vram_used_gb != null ? `${gpu.vram_used_gb.toFixed(1)} / ` : ""}${formatGb(gpu.vram_total_gb)}`
                              : ""
                          }`}
                          className="h-2 overflow-hidden rounded-[3px] bg-foreground/10"
                        >
                          <span
                            className={cn(
                              "block h-full bg-primary",
                              cell != null && cell > 90 && "bg-amber-500",
                            )}
                            style={{ width: `${cell ?? 0}%` }}
                          />
                        </span>
                      );
                    })}
                  </div>
                </>
              ) : null}
            </div>
          ) : (
            <span className="text-ui-11 text-muted-foreground">
              No GPU reported
            </span>
          )}
          <span
            className={cn(
              "break-all text-ui-11",
              serving
                ? "font-mono text-foreground/85"
                : "text-muted-foreground",
            )}
          >
            {serving ?? "No model loaded"}
          </span>
        </>
      )}

      <span className="text-ui-10 text-muted-foreground">{footer}</span>
    </button>
  );
}

/** Linked machines in a slim column beside the monitor; click one for its details. */
export function LinkedInstancesRail({
  overview,
  className,
}: {
  overview: Overview;
  className?: string;
}) {
  const { instances, statuses, infos, refresh, refreshing } = overview;
  const [openId, setOpenId] = useState<string | null>(null);
  const online = instances.filter((i) => statuses[i.id]?.online).length;
  const open = instances.find((i) => i.id === openId) ?? null;

  return (
    <aside className={cn("flex min-w-0 flex-col gap-2.5", className)}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-col">
          <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
            Linked instances
          </span>
          <span className="text-ui-11 text-muted-foreground">
            {online} of {instances.length} online
          </span>
        </div>
        <div className="flex items-center gap-0.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="size-7 p-0 text-muted-foreground hover:text-foreground"
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label="Refresh linked instances"
            title="Refresh"
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={cn("size-3.5", refreshing && "animate-spin")}
            />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-ui-12 text-muted-foreground hover:text-foreground"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("api-keys")
            }
          >
            Manage
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:max-xl:grid-cols-2">
        {instances.map((instance) => (
          <InstanceCard
            key={instance.id}
            instance={instance}
            status={statuses[instance.id]}
            info={infos[instance.id]}
            onOpen={() => setOpenId(instance.id)}
          />
        ))}
      </div>

      <LinkedInstanceDetailsDialog
        instance={open}
        status={open ? statuses[open.id] : undefined}
        info={open ? infos[open.id] : undefined}
        onOpenChange={(value) => !value && setOpenId(null)}
        onRefresh={() => void refresh()}
        refreshing={refreshing}
      />
    </aside>
  );
}
