// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { type TranslationKey, useT } from "@/i18n";
import type { InterpolationValues } from "@/i18n/types";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Copy01Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import type {
  LinkedInstance,
  LinkedInstanceGpu,
  LinkedInstanceInfo,
  LinkedInstanceStatus,
} from "../api/linked-instances";
import {
  acceleratorLabel,
  connectionKind,
  formatGb,
  formatUptime,
  gpuPool,
  installLabel,
  platformLabel,
} from "./linked-instance-format";

type LinkedKey = TranslationKey extends infer K
  ? K extends `settings.apiKeys.linkedInstances.${infer Rest}`
    ? Rest
    : never
  : never;

const CONNECTION_KEY = {
  tunnel: "infoTunnel",
  loopback: "infoLoopback",
  lan: "infoLan",
  public: "infoPublic",
} as const satisfies Record<string, LinkedKey>;

export function GpuMeter({
  gpu,
  index,
}: {
  gpu: LinkedInstanceGpu;
  index?: number;
}) {
  const t = useT();
  const total = formatGb(gpu.vram_total_gb);
  const used = formatGb(gpu.vram_used_gb);
  const pct =
    gpu.vram_used_gb != null && gpu.vram_total_gb
      ? Math.min(100, (gpu.vram_used_gb / gpu.vram_total_gb) * 100)
      : null;
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3">
        <span className="min-w-0 break-words text-ui-12 text-foreground">
          {index != null ? (
            <span className="mr-1.5 font-mono text-ui-10 text-muted-foreground">
              {index}
            </span>
          ) : null}
          {gpu.name}
        </span>
        <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
          {used && total
            ? t("settings.apiKeys.linkedInstances.infoVramUsed", {
                used,
                total,
              })
            : total}
        </span>
      </div>
      {pct != null ? (
        <Progress
          value={pct}
          className="h-1.5"
          indicatorClassName={cn(pct > 90 && "bg-amber-500")}
        />
      ) : null}
    </div>
  );
}

function Section({
  title,
  children,
  className,
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={cn(
        "flex min-w-0 flex-col gap-2.5 rounded-lg border border-border/60 p-3",
        className,
      )}
    >
      <h3 className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      {children}
    </section>
  );
}

function Rows({ rows }: { rows: [string, string | null, string?][] }) {
  const t = useT();
  return (
    <dl className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-ui-12">
      {rows.map(([label, value, title]) => (
        <div key={label} className="contents">
          <dt className="text-muted-foreground">{label}</dt>
          <dd
            className={cn(
              "truncate font-mono text-ui-11 leading-5",
              value ? "text-foreground" : "text-muted-foreground/70",
            )}
            title={title ?? value ?? undefined}
          >
            {value ?? t("settings.apiKeys.linkedInstances.infoUnknown")}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function LinkedInstanceDetailsDialog({
  instance,
  status,
  info,
  onOpenChange,
  onRefresh,
  refreshing = false,
}: {
  instance: LinkedInstance | null;
  status: LinkedInstanceStatus | undefined;
  info: LinkedInstanceInfo | undefined;
  onOpenChange: (open: boolean) => void;
  onRefresh: () => void;
  refreshing?: boolean;
}) {
  const t = useT();
  const k = (key: LinkedKey, values?: InterpolationValues) =>
    t(`settings.apiKeys.linkedInstances.${key}`, values);
  const copy = async (text: string) => {
    if (await copyToClipboard(text))
      toast.success(t("settings.apiKeys.copied"));
  };

  const prefix = instance ? `@${instance.name}/` : "";
  const online = status ? status.online : info?.online;
  const loaded = new Set(status?.loaded ?? []);
  const models = [...(status?.models ?? [])].sort(
    (a, b) => Number(loaded.has(b)) - Number(loaded.has(a)),
  );
  const kind = instance ? connectionKind(instance.base_url) : "public";
  const kindLabel = k(CONNECTION_KEY[kind]);
  const ready = info?.online === true;

  const skeleton = (
    <div className="flex flex-col gap-2">
      <Skeleton className="h-3.5 w-3/4" />
      <Skeleton className="h-3.5 w-1/2" />
      <Skeleton className="h-3.5 w-2/3" />
    </div>
  );

  return (
    <Dialog open={instance !== null} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 font-mono">
            <span
              aria-hidden={true}
              className={cn(
                "size-2.5 shrink-0 rounded-full",
                online === undefined
                  ? "animate-pulse bg-muted-foreground/50"
                  : online
                    ? "bg-emerald-500"
                    : "bg-red-500",
              )}
            />
            @{instance?.name}
          </DialogTitle>
          <DialogDescription className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
            <span className="truncate font-mono text-ui-11">
              {instance?.base_url}
            </span>
            <span aria-hidden={true}>·</span>
            <span>{kindLabel}</span>
            {status?.latency_ms != null ? (
              <>
                <span aria-hidden={true}>·</span>
                <span className="tabular-nums">{status.latency_ms} ms</span>
              </>
            ) : null}
          </DialogDescription>
        </DialogHeader>

        {info && !info.online ? (
          <p className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-ui-12 text-destructive">
            {info.error ?? k("infoError")}
          </p>
        ) : null}

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Section title={k("infoUnsloth")}>
            {ready ? (
              <Rows
                rows={[
                  [k("infoVersion"), info.version],
                  [k("infoInstall"), installLabel(info.install_source)],
                  [
                    k("infoUpdates"),
                    info.update_available && info.latest_version
                      ? k("infoUpdateAvailable", {
                          version: info.latest_version,
                        })
                      : info.version
                        ? k("infoUpToDate")
                        : null,
                  ],
                  [k("infoUptime"), formatUptime(info.uptime_seconds)],
                ]}
              />
            ) : (
              skeleton
            )}
          </Section>

          <Section title={k("infoHardware")}>
            {ready ? (
              <div className="flex flex-col gap-3">
                {info.gpus.length > 1 ? (
                  <GpuMeter
                    gpu={{
                      name: k("infoTotalVram"),
                      vram_total_gb: gpuPool(info.gpus).total,
                      vram_used_gb: gpuPool(info.gpus).used,
                      utilization_pct: null,
                    }}
                  />
                ) : null}
                {info.gpus.length > 0 ? (
                  <div
                    className={cn(
                      "flex flex-col gap-2.5",
                      info.gpus.length > 2 &&
                        "max-h-[calc(11rem*var(--ui-space-scale,1))] overflow-y-auto border-l border-border/60 pr-1 pl-2.5",
                    )}
                  >
                    {info.gpus.map((gpu, i) => (
                      <GpuMeter
                        // biome-ignore lint/suspicious/noArrayIndexKey: identical cards share a name
                        key={i}
                        gpu={gpu}
                        index={info.gpus.length > 1 ? i : undefined}
                      />
                    ))}
                  </div>
                ) : (
                  <span className="text-ui-12 text-muted-foreground">
                    {k("infoNoGpu")}
                  </span>
                )}
                <Rows
                  rows={[
                    [
                      k("infoCpu"),
                      info.cpu_count != null
                        ? k("infoCores", { count: String(info.cpu_count) })
                        : null,
                    ],
                    [
                      k("infoMemory"),
                      info.memory_total_gb != null
                        ? k("infoFreeOf", {
                            free: formatGb(info.memory_available_gb) ?? "?",
                            total: formatGb(info.memory_total_gb) ?? "?",
                          })
                        : null,
                    ],
                    [
                      k("infoDisk"),
                      info.disk_total_gb != null
                        ? k("infoFreeOf", {
                            free: formatGb(info.disk_free_gb) ?? "?",
                            total: formatGb(info.disk_total_gb) ?? "?",
                          })
                        : null,
                    ],
                  ]}
                />
              </div>
            ) : (
              skeleton
            )}
          </Section>

          <Section title={k("infoRuntime")} className="sm:col-span-2">
            {ready ? (
              <div className="grid grid-cols-1 gap-x-6 sm:grid-cols-2">
                <Rows
                  rows={[
                    [
                      k("infoPlatform"),
                      platformLabel(info.platform),
                      info.platform ?? undefined,
                    ],
                    [
                      k("infoAccelerator"),
                      acceleratorLabel(info) ??
                        (info.gpus.length === 0 ? k("infoCpuOnly") : null),
                    ],
                    ["Python", info.python_version],
                  ]}
                />
                <Rows
                  rows={[
                    ["PyTorch", info.torch],
                    ["Transformers", info.transformers],
                    ["llama.cpp", info.llama_cpp],
                  ]}
                />
              </div>
            ) : (
              skeleton
            )}
          </Section>

          <Section
            title={`${k("infoModels")}${models.length ? ` · ${models.length}` : ""}`}
            className="sm:col-span-2"
          >
            {models.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {models.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => void copy(id)}
                    title={k("copyId")}
                    className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-border/60 bg-muted/30 px-2 py-0.5 font-mono text-ui-11 text-muted-foreground transition-colors hover:border-border hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    {loaded.has(id) ? (
                      <span
                        aria-hidden={true}
                        className="size-1.5 shrink-0 rounded-full bg-emerald-500"
                      />
                    ) : null}
                    <span className="truncate">{id.slice(prefix.length)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <span className="text-ui-12 text-muted-foreground">
                {k("infoNoModels")}
              </span>
            )}
          </Section>
        </div>

        <DialogFooter className="gap-2 sm:justify-between">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void copy(prefix)}
          >
            <HugeiconsIcon icon={Copy01Icon} className="mr-1.5 size-3.5" />
            {k("copyPrefix")}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={refreshing}
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={cn("mr-1.5 size-3.5", refreshing && "animate-spin")}
            />
            {k("refresh")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
