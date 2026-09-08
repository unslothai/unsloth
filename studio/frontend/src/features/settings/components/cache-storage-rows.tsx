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
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type CacheEntry,
  type CacheInventory,
  type CacheKey,
  bulkPurgeKeys,
  loadCacheInventory,
  purgeCaches,
} from "../api/caches";
import { SettingsRow } from "./settings-row";

/** Decimal units, matching the disk readings the Storage section already shows. */
export function formatCacheSize(bytes: number): string {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  if (safe >= 1e9) {
    const gb = safe / 1e9;
    return `${gb >= 10 ? gb.toFixed(1) : gb.toFixed(2)} GB`;
  }
  if (safe >= 1e6) return `${Math.round(safe / 1e6)} MB`;
  if (safe >= 1e3) return `${Math.round(safe / 1e3)} KB`;
  return `${Math.round(safe)} B`;
}

const CACHE_NAME_KEYS: Record<CacheKey, TranslationKey> = {
  uv: "settings.resources.storage.caches.names.uv",
  pip: "settings.resources.storage.caches.names.pip",
  npm: "settings.resources.storage.caches.names.npm",
  bun: "settings.resources.storage.caches.names.bun",
  // biome-ignore lint/style/useNamingConvention: API schema
  torch_inductor: "settings.resources.storage.caches.names.torchInductor",
  // biome-ignore lint/style/useNamingConvention: API schema
  torch_extensions: "settings.resources.storage.caches.names.torchExtensions",
  triton: "settings.resources.storage.caches.names.triton",
  cuda: "settings.resources.storage.caches.names.cuda",
  numba: "settings.resources.storage.caches.names.numba",
  matplotlib: "settings.resources.storage.caches.names.matplotlib",
  vllm: "settings.resources.storage.caches.names.vllm",
  // biome-ignore lint/style/useNamingConvention: API schema
  unsloth_compiled: "settings.resources.storage.caches.names.unslothCompiled",
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_xet: "settings.resources.storage.caches.names.hfXet",
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_assets: "settings.resources.storage.caches.names.hfAssets",
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_datasets: "settings.resources.storage.caches.names.hfDatasets",
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_hub: "settings.resources.storage.caches.names.hfHub",
};

const OPT_IN_COST_KEYS: Partial<Record<CacheKey, TranslationKey>> = {
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_hub: "settings.resources.storage.caches.hubCost",
  // biome-ignore lint/style/useNamingConvention: API schema
  hf_datasets: "settings.resources.storage.caches.datasetsCost",
};

/** What a confirmation is about: everything reclaimable, or one opt-in cache. */
type PurgeTarget = { kind: "bulk" } | { kind: "single"; key: CacheKey };

function presentCaches(inventory: CacheInventory | null): CacheEntry[] {
  if (!inventory) return [];
  return inventory.caches.filter((entry) => entry.present);
}

export function CacheStorageRows() {
  const t = useT();
  const [inventory, setInventory] = useState<CacheInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [target, setTarget] = useState<PurgeTarget | null>(null);
  const [clearing, setClearing] = useState(false);

  const refresh = useCallback(
    async (options: { refresh?: boolean } = {}) => {
      setLoading(true);
      try {
        setInventory(await loadCacheInventory(options));
        setLoadError(null);
      } catch (error) {
        setLoadError(
          error instanceof Error
            ? error.message
            : t("settings.resources.storage.caches.measureFailed"),
        );
      } finally {
        setLoading(false);
      }
    },
    [t],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const entries = presentCaches(inventory);
  const bulkKeys = inventory ? bulkPurgeKeys(inventory) : [];
  const reclaimable = inventory?.reclaimableBytes ?? 0;

  const runPurge = async (keys: readonly CacheKey[]) => {
    setClearing(true);
    try {
      const outcome = await purgeCaches(keys);
      setInventory(outcome.inventory);
      setTarget(null);
      const failures = outcome.results.flatMap((result) => result.errors);
      if (failures.length > 0) {
        toast.warning(t("settings.resources.storage.caches.partial"), {
          description: failures[0],
        });
        return;
      }
      toast.success(
        t("settings.resources.storage.caches.cleared", {
          size: formatCacheSize(outcome.freedBytes),
        }),
      );
    } catch (error) {
      toast.error(t("settings.resources.storage.caches.clearFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setClearing(false);
    }
  };

  const confirmTitle =
    target?.kind === "single"
      ? t("settings.resources.storage.caches.confirmOneTitle", {
          name: t(CACHE_NAME_KEYS[target.key]),
        })
      : t("settings.resources.storage.caches.confirmTitle");
  const confirmDescription =
    target?.kind === "single"
      ? [
          OPT_IN_COST_KEYS[target.key]
            ? t(OPT_IN_COST_KEYS[target.key] as TranslationKey)
            : "",
          t("settings.resources.storage.caches.safety"),
        ]
          .filter(Boolean)
          .join(" ")
      : `${t("settings.resources.storage.caches.confirmDescription", {
          size: formatCacheSize(reclaimable),
        })} ${t("settings.resources.storage.caches.safety")}`;

  return (
    <>
      <SettingsRow
        label={t("settings.resources.storage.caches.label")}
        description={
          loadError
            ? loadError
            : loading && !inventory
              ? t("settings.resources.storage.caches.measuring")
              : t("settings.resources.storage.caches.description", {
                  size: formatCacheSize(inventory?.totalBytes ?? 0),
                  reclaimable: formatCacheSize(reclaimable),
                })
        }
        hint={t("settings.resources.storage.caches.hint")}
      >
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={loading || clearing}
            onClick={() => void refresh({ refresh: true })}
          >
            {t("settings.resources.storage.caches.recheckAction")}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={loading || clearing}
            onClick={() => setExpanded((open) => !open)}
          >
            {expanded
              ? t("settings.resources.storage.caches.hideDetailsAction")
              : t("settings.resources.storage.caches.detailsAction")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-8"
            disabled={loading || clearing || bulkKeys.length === 0}
            onClick={() => setTarget({ kind: "bulk" })}
          >
            {t("settings.resources.storage.caches.clearAction")}
          </Button>
        </div>
      </SettingsRow>

      {expanded ? (
        <div className="flex flex-col pl-3.5">
          {entries.length === 0 ? (
            <p className="py-2 text-xs text-muted-foreground">
              {t("settings.resources.storage.caches.empty")}
            </p>
          ) : (
            entries.map((entry) => (
              <div
                key={entry.key}
                className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 py-2"
              >
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm text-foreground">
                    {t(CACHE_NAME_KEYS[entry.key])}
                  </span>
                  <span
                    title={entry.paths.join("\n")}
                    className="max-w-lg truncate font-mono text-[11px] text-muted-foreground"
                  >
                    {entry.paths[0] ?? ""}
                  </span>
                  {entry.optIn && OPT_IN_COST_KEYS[entry.key] ? (
                    <span className="text-xs text-amber-600 dark:text-amber-400">
                      {t(OPT_IN_COST_KEYS[entry.key] as TranslationKey)}
                    </span>
                  ) : null}
                  {entry.blockedReason ? (
                    <span className="text-xs text-muted-foreground">
                      {t("settings.resources.storage.caches.blocked", {
                        reason: entry.blockedReason,
                      })}
                    </span>
                  ) : null}
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {formatCacheSize(entry.sizeBytes)}
                  </span>
                  <Button
                    variant="ghost"
                    size="xs"
                    disabled={
                      clearing || !entry.purgeable || entry.sizeBytes === 0
                    }
                    onClick={() =>
                      entry.optIn
                        ? setTarget({ kind: "single", key: entry.key })
                        : void runPurge([entry.key])
                    }
                  >
                    {t("settings.resources.storage.caches.clearOneAction")}
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      ) : null}

      <Dialog
        open={target !== null}
        onOpenChange={(open) => {
          if (!open) setTarget(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{confirmTitle}</DialogTitle>
            <DialogDescription>{confirmDescription}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setTarget(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              disabled={clearing}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
              onClick={() =>
                void runPurge(
                  target?.kind === "single" ? [target.key] : bulkKeys,
                )
              }
            >
              {clearing
                ? t("settings.resources.storage.caches.clearingAction")
                : t("settings.resources.storage.caches.clearAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
