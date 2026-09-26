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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Delete02Icon,
  InformationCircleIcon,
  Link01Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useId, useState } from "react";
import {
  type LinkedInstance,
  type LinkedInstanceInfo,
  type LinkedInstanceStatus,
  createLinkedInstance,
  deleteLinkedInstance,
  fetchLinkedInstances,
  fetchLinkedInstancesInfo,
  testLinkedInstance,
  updateLinkedInstance,
} from "../api/linked-instances";
import { LinkedInstanceDetailsDialog } from "./linked-instance-details-dialog";
import { acceleratorLabel, formatGb, gpuPool } from "./linked-instance-format";

const MODELS_SHOWN = 4;

type Status = LinkedInstanceStatus | "checking" | undefined;

function StatusDot({ status }: { status: Status }) {
  const checking = status === undefined || status === "checking";
  return (
    <span
      aria-hidden={true}
      className={cn(
        "size-2 shrink-0 rounded-full",
        checking
          ? "animate-pulse bg-muted-foreground/50"
          : status.online
            ? "bg-emerald-500"
            : "bg-red-500",
      )}
    />
  );
}

/** "Unsloth 2026.9.11 · NVIDIA L4 22.5 GB · CUDA 12.8" */
function machineSummary(info: LinkedInstanceInfo | undefined): string | null {
  if (!info?.online) return null;
  const pool = info.gpus.length > 0 ? gpuPool(info.gpus) : null;
  return [
    info.version ? `Unsloth ${info.version}` : null,
    pool ? [pool.label, formatGb(pool.total)].filter(Boolean).join(" ") : null,
    acceleratorLabel(info),
  ]
    .filter(Boolean)
    .join(" · ");
}

function InstanceRow({
  instance,
  status,
  info,
  onCheck,
  onDetails,
  onEdit,
  onRemove,
}: {
  instance: LinkedInstance;
  status: Status;
  info: LinkedInstanceInfo | undefined;
  onCheck: () => void;
  onDetails: () => void;
  onEdit: () => void;
  onRemove: () => void;
}) {
  const t = useT();
  const checking = status === undefined || status === "checking";
  const models = checking ? [] : status.models;
  const loaded = new Set(checking ? [] : status.loaded);
  const prefix = `@${instance.name}/`;
  const copy = async (text: string) => {
    if (await copyToClipboard(text)) {
      toast.success(t("settings.apiKeys.copied"));
    }
  };

  let meta: string;
  if (checking) {
    meta = t("settings.apiKeys.linkedInstances.checking");
  } else if (status.online) {
    meta = [
      models.length === 1
        ? t("settings.apiKeys.linkedInstances.modelCountOne")
        : t("settings.apiKeys.linkedInstances.modelCount", {
            count: String(models.length),
          }),
      loaded.size > 0
        ? t("settings.apiKeys.linkedInstances.loadedCount", {
            count: String(loaded.size),
          })
        : null,
      status.latency_ms != null ? `${status.latency_ms} ms` : null,
    ]
      .filter(Boolean)
      .join(" · ");
  } else {
    meta = status.error ?? t("settings.apiKeys.linkedInstances.offline");
  }

  return (
    <div className="group flex flex-col gap-2 px-4 py-3 transition-colors hover:bg-accent/30">
      <div className="flex min-w-0 items-center gap-3">
        <StatusDot status={status} />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <div className="flex min-w-0 items-baseline justify-between gap-3">
            <button
              type="button"
              onClick={onDetails}
              className="truncate rounded-sm font-mono text-sm font-medium text-foreground hover:underline hover:decoration-border hover:underline-offset-2 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              @{instance.name}
            </button>
            <span
              className={cn(
                "shrink-0 text-ui-11 tabular-nums",
                !checking && !status.online
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
            >
              {meta}
            </span>
          </div>
          <span
            className="truncate font-mono text-ui-11 text-muted-foreground"
            title={instance.base_url}
          >
            {instance.base_url}
          </span>
          {machineSummary(info) ? (
            <span className="truncate text-ui-11 text-muted-foreground">
              {machineSummary(info)}
            </span>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="size-7 p-0 text-muted-foreground hover:text-foreground"
            onClick={onCheck}
            disabled={checking}
            aria-label={t("settings.apiKeys.linkedInstances.checkAgain")}
            title={t("settings.apiKeys.linkedInstances.checkAgain")}
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={cn("size-3.5", checking && "animate-spin")}
            />
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="size-7 p-0 text-muted-foreground hover:text-foreground"
                aria-label={t("settings.apiKeys.linkedInstances.actions", {
                  name: instance.name,
                })}
              >
                <HugeiconsIcon icon={MoreHorizontalIcon} className="size-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onDetails}>
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  className="mr-2 size-3.5"
                />
                {t("settings.apiKeys.linkedInstances.details")}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onEdit}>
                <HugeiconsIcon
                  icon={PencilEdit02Icon}
                  className="mr-2 size-3.5"
                />
                {t("settings.apiKeys.linkedInstances.edit")}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => void copy(prefix)}>
                <HugeiconsIcon icon={Copy01Icon} className="mr-2 size-3.5" />
                {t("settings.apiKeys.linkedInstances.copyPrefix")}
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={onRemove}
                className="text-destructive focus:text-destructive"
              >
                <HugeiconsIcon icon={Delete02Icon} className="mr-2 size-3.5" />
                {t("settings.apiKeys.linkedInstances.remove")}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {models.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5 pl-5">
          {models.slice(0, MODELS_SHOWN).map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => void copy(id)}
              title={t("settings.apiKeys.linkedInstances.copyId")}
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
          {models.length > MODELS_SHOWN ? (
            <span className="text-ui-11 text-muted-foreground">
              {t("settings.apiKeys.linkedInstances.moreModels", {
                count: String(models.length - MODELS_SHOWN),
              })}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function InstanceForm({
  editing,
  onSaved,
  onCancel,
}: {
  /** The instance being edited; omitted when adding. A blank key keeps the saved one. */
  editing?: LinkedInstance;
  onSaved: (instance: LinkedInstance) => void;
  onCancel?: () => void;
}) {
  const t = useT();
  const id = useId();
  const [name, setName] = useState(editing?.name ?? "");
  const [baseUrl, setBaseUrl] = useState(editing?.base_url ?? "");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canSubmit =
    name.trim() !== "" &&
    baseUrl.trim() !== "" &&
    (editing !== undefined || apiKey.trim() !== "");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit || saving) return;
    setSaving(true);
    setError(null);
    try {
      onSaved(
        editing
          ? await updateLinkedInstance(editing.id, {
              name: name.trim(),
              base_url: baseUrl.trim(),
              ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
            })
          : await createLinkedInstance({
              name: name.trim(),
              base_url: baseUrl.trim(),
              api_key: apiKey.trim(),
            }),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("settings.apiKeys.linkedInstances.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const field = (
    key: string,
    label: string,
    input: React.ReactNode,
    className?: string,
  ) => (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <label
        htmlFor={`${id}-${key}`}
        className="text-ui-11 font-medium text-muted-foreground"
      >
        {label}
      </label>
      {input}
    </div>
  );

  return (
    <form
      onSubmit={submit}
      className={cn(
        "flex flex-col gap-3 bg-muted/10 p-4",
        !editing && "border-t border-border/60",
      )}
    >
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-[minmax(0,7rem)_minmax(0,1fr)_minmax(0,11rem)]">
        {field(
          "name",
          t("settings.apiKeys.linkedInstances.name"),
          <Input
            id={`${id}-name`}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={t("settings.apiKeys.linkedInstances.namePlaceholder")}
            autoComplete="off"
            spellCheck={false}
            className="h-8 font-mono text-xs"
            autoFocus={true}
          />,
        )}
        {field(
          "url",
          t("settings.apiKeys.linkedInstances.url"),
          <Input
            id={`${id}-url`}
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={t("settings.apiKeys.linkedInstances.urlPlaceholder")}
            autoComplete="off"
            spellCheck={false}
            inputMode="url"
            className="h-8 font-mono text-xs"
          />,
        )}
        {field(
          "key",
          t("settings.apiKeys.linkedInstances.apiKey"),
          <Input
            id={`${id}-key`}
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={
              editing
                ? t("settings.apiKeys.linkedInstances.apiKeyKeep")
                : t("settings.apiKeys.linkedInstances.apiKeyPlaceholder")
            }
            autoComplete="off"
            className="h-8 font-mono text-xs"
          />,
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p
          className={cn(
            "min-w-0 flex-1 text-ui-11 leading-snug",
            error ? "text-destructive" : "text-muted-foreground",
          )}
          role={error ? "alert" : undefined}
        >
          {error ?? t("settings.apiKeys.linkedInstances.formHint")}
        </p>
        <div className="flex shrink-0 items-center gap-2">
          {onCancel ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onCancel}
            >
              {t("common.cancel")}
            </Button>
          ) : null}
          <Button type="submit" size="sm" disabled={!canSubmit || saving}>
            {editing
              ? saving
                ? t("common.saving")
                : t("common.save")
              : saving
                ? t("settings.apiKeys.linkedInstances.linking")
                : t("settings.apiKeys.linkedInstances.link")}
          </Button>
        </div>
      </div>
    </form>
  );
}

export function LinkedInstancesSection() {
  const t = useT();
  const [instances, setInstances] = useState<LinkedInstance[] | null>(null);
  const [statuses, setStatuses] = useState<
    Record<string, LinkedInstanceStatus | "checking">
  >({});
  const [adding, setAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [removeTarget, setRemoveTarget] = useState<LinkedInstance | null>(null);
  const [removing, setRemoving] = useState(false);
  const [infos, setInfos] = useState<Record<string, LinkedInstanceInfo>>({});
  const [detailsId, setDetailsId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadInfo = useCallback(async () => {
    try {
      const rows = await fetchLinkedInstancesInfo();
      setInfos(Object.fromEntries(rows.map((r) => [r.id, r])));
    } catch {
      // Details are extra; the status line already says whether it is reachable.
    }
  }, []);

  const check = useCallback(async (instanceId: string) => {
    setStatuses((prev) => ({ ...prev, [instanceId]: "checking" }));
    try {
      const status = await testLinkedInstance(instanceId);
      setStatuses((prev) => ({ ...prev, [instanceId]: status }));
    } catch (e) {
      setStatuses((prev) => ({
        ...prev,
        [instanceId]: {
          id: instanceId,
          online: false,
          error: e instanceof Error ? e.message : null,
          models: [],
          loaded: [],
          latency_ms: null,
        },
      }));
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const loaded = await fetchLinkedInstances();
      setInstances(loaded);
      setError(null);
      for (const instance of loaded) void check(instance.id);
      void loadInfo();
    } catch {
      setError(t("settings.apiKeys.linkedInstances.loadError"));
    }
  }, [t, check, loadInfo]);

  useEffect(() => {
    void load();
  }, [load]);

  const confirmRemove = async () => {
    if (!removeTarget) return;
    setRemoving(true);
    try {
      await deleteLinkedInstance(removeTarget.id);
      setRemoveTarget(null);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : null);
    } finally {
      setRemoving(false);
    }
  };

  const total = instances?.length ?? 0;
  const online = (instances ?? []).filter((i) => {
    const s = statuses[i.id];
    return s !== undefined && s !== "checking" && s.online;
  }).length;
  const settled =
    instances !== null &&
    instances.every((i) => {
      const s = statuses[i.id];
      return s !== undefined && s !== "checking";
    });

  return (
    <section
      data-settings-label={t("settings.apiKeys.linkedInstances.title")}
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={Link01Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold font-heading text-foreground">
                {t("settings.apiKeys.linkedInstances.title")}
              </h2>
              <output
                className="flex items-center gap-1.5 text-xs text-muted-foreground"
                aria-live="polite"
              >
                <span
                  className={cn(
                    "size-2 rounded-full",
                    total > 0 && settled && online === total
                      ? "bg-emerald-500"
                      : total > 0 && settled && online < total
                        ? "bg-amber-500"
                        : "bg-muted-foreground",
                  )}
                />
                {total === 0
                  ? t("settings.apiKeys.linkedInstances.statusNone")
                  : t("settings.apiKeys.linkedInstances.statusOnline", {
                      online: String(online),
                      total: String(total),
                    })}
              </output>
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t("settings.apiKeys.linkedInstances.description")}
            </p>
          </div>
        </div>
        <Button
          type="button"
          size="sm"
          variant={adding ? "outline" : "default"}
          className="min-w-20 shrink-0"
          onClick={() => {
            setEditingId(null);
            setAdding((v) => !v);
          }}
        >
          {adding
            ? t("common.cancel")
            : t("settings.apiKeys.linkedInstances.add")}
        </Button>
      </div>

      {adding ? (
        <InstanceForm
          onSaved={(instance) => {
            setAdding(false);
            setInstances((prev) => [...(prev ?? []), instance]);
            void check(instance.id);
          }}
        />
      ) : null}

      {error ? (
        <p className="border-t border-border/60 px-4 py-2.5 text-xs text-destructive">
          {error}
        </p>
      ) : null}

      {instances && instances.length > 0 ? (
        <div className="divide-y divide-border/60 border-t border-border/60">
          {instances.map((instance) =>
            editingId === instance.id ? (
              <InstanceForm
                key={instance.id}
                editing={instance}
                onCancel={() => setEditingId(null)}
                onSaved={(saved) => {
                  setEditingId(null);
                  setInstances((prev) =>
                    (prev ?? []).map((i) => (i.id === saved.id ? saved : i)),
                  );
                  void check(saved.id);
                  void loadInfo();
                }}
              />
            ) : (
              <InstanceRow
                key={instance.id}
                instance={instance}
                status={statuses[instance.id]}
                info={infos[instance.id]}
                onCheck={() => {
                  void check(instance.id);
                  void loadInfo();
                }}
                onDetails={() => setDetailsId(instance.id)}
                onEdit={() => {
                  setAdding(false);
                  setEditingId(instance.id);
                }}
                onRemove={() => setRemoveTarget(instance)}
              />
            ),
          )}
        </div>
      ) : null}

      <LinkedInstanceDetailsDialog
        instance={instances?.find((i) => i.id === detailsId) ?? null}
        status={(() => {
          const s = detailsId ? statuses[detailsId] : undefined;
          return s === "checking" ? undefined : s;
        })()}
        info={detailsId ? infos[detailsId] : undefined}
        onOpenChange={(open) => !open && setDetailsId(null)}
        onRefresh={() => {
          if (detailsId) void check(detailsId);
          void loadInfo();
        }}
        refreshing={detailsId !== null && statuses[detailsId] === "checking"}
      />

      <Dialog
        open={removeTarget !== null}
        onOpenChange={(open) => !open && setRemoveTarget(null)}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.apiKeys.linkedInstances.removeTitle", {
                name: removeTarget?.name ?? "",
              })}
            </DialogTitle>
            <DialogDescription>
              {t("settings.apiKeys.linkedInstances.removeDescription", {
                name: removeTarget?.name ?? "",
              })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRemoveTarget(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={() => void confirmRemove()}
              disabled={removing}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("settings.apiKeys.linkedInstances.remove")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
