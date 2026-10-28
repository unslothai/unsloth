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
  bumpInventoryVersion,
  getInventoryVersion,
  useInventoryVersion,
} from "@/features/hub/stores/inventory-events";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type CacheEntry,
  type CacheInventory,
  type CacheKey,
  bulkPurgeKeys,
  loadCacheInventory,
  purgeCaches,
} from "../api/caches";
import { SettingsCardRow } from "./storage-overview";

/** Decimal units, matching the Storage section's disk readings. */
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

const HUB_INVENTORY_KEYS: ReadonlySet<CacheKey> = new Set<CacheKey>([
  "hf_hub",
  "hf_datasets",
]);

type PurgeTarget = { kind: "bulk" } | { kind: "single"; key: CacheKey };

/** The generic text says downloaded models are untouched, which is false for the hub cache. */
export function singleClearDescriptionKeys(key: CacheKey): TranslationKey[] {
  const cost = OPT_IN_COST_KEYS[key];
  const spared: TranslationKey[] =
    key === "hf_hub" ? [] : ["settings.resources.storage.caches.safety"];
  return cost ? [cost, ...spared] : spared;
}

function presentCaches(inventory: CacheInventory | null): CacheEntry[] {
  if (!inventory) return [];
  return inventory.caches.filter((entry) => entry.present);
}

// A slow cache can answer from its last walk while it re-walks; ask again until every size is current.
const FOLLOW_UP_MS = 2000;
const FOLLOW_UP_LIMIT = 60;

export function useCacheStorage() {
  const t = useT();
  const [inventory, setInventory] = useState<CacheInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);

  // Only the newest measurement installs itself; overlapping ones finish in any order.
  const latestRequest = useRef(0);
  const followUp = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelFollowUp = useCallback(() => {
    if (followUp.current !== null) clearTimeout(followUp.current);
    followUp.current = null;
  }, []);

  // Runs only if nothing newer was asked for in the meantime.
  const scheduleFollowUp = useCallback(
    (wanted: boolean, request: number, run: () => void) => {
      if (!wanted) return;
      followUp.current = setTimeout(() => {
        followUp.current = null;
        if (request === latestRequest.current) run();
      }, FOLLOW_UP_MS);
    },
    [],
  );

  const refresh = useCallback(
    async (options: { refresh?: boolean } = {}, firstAttempt = 0) => {
      cancelFollowUp();
      // A follow-up only swaps in the newer sizes: no measuring state, and a failed one keeps the rows.
      const load = async (attempt: number) => {
        const request = ++latestRequest.current;
        if (attempt === 0) setLoading(true);
        try {
          const next = await loadCacheInventory(attempt === 0 ? options : {});
          if (request !== latestRequest.current) return;
          setInventory(next);
          setLoadError(null);
          scheduleFollowUp(
            next.refreshing && attempt < FOLLOW_UP_LIMIT,
            request,
            () => void load(attempt + 1),
          );
        } catch (error) {
          if (request !== latestRequest.current || attempt > 0) return;
          setLoadError(
            error instanceof Error
              ? error.message
              : t("settings.resources.storage.caches.measureFailed"),
          );
        } finally {
          if (attempt === 0 && request === latestRequest.current) {
            setLoading(false);
          }
        }
      };
      await load(firstAttempt);
    },
    [t, cancelFollowUp, scheduleFollowUp],
  );

  // Saving Models Folder bumps the inventory version; re-measure so rows match the new folder.
  const inventoryVersion = useInventoryVersion();
  const measuredVersion = useRef(inventoryVersion);

  useEffect(() => {
    // A move forces a re-measure; the first load does not, since a cold walk costs seconds.
    const moved = measuredVersion.current !== inventoryVersion;
    measuredVersion.current = inventoryVersion;
    void refresh(moved ? { refresh: true } : {});
    // Leaving the page stops any follow-up read.
    return () => {
      if (followUp.current !== null) clearTimeout(followUp.current);
      followUp.current = null;
    };
  }, [refresh, inventoryVersion]);

  const entries = presentCaches(inventory);
  const bulkKeys = inventory ? bulkPurgeKeys(inventory) : [];
  const reclaimable = inventory?.reclaimableBytes ?? 0;

  const runPurge = async (keys: readonly CacheKey[]): Promise<boolean> => {
    setClearing(true);
    cancelFollowUp();
    const request = ++latestRequest.current;
    try {
      const outcome = await purgeCaches(keys);
      if (request === latestRequest.current) {
        setInventory(outcome.inventory);
        scheduleFollowUp(
          outcome.inventory.refreshing,
          request,
          () => void refresh({}, 1),
        );
      }
      if (keys.some((key) => HUB_INVENTORY_KEYS.has(key))) {
        // Our own bump is marked so the rows skip a cold walk for data they already hold.
        bumpInventoryVersion();
        measuredVersion.current = getInventoryVersion();
      }
      const failures = outcome.results.flatMap((result) => result.errors);
      if (failures.length > 0) {
        toast.warning(t("settings.resources.storage.caches.partial"), {
          description: failures[0],
        });
        return true;
      }
      toast.success(
        t("settings.resources.storage.caches.cleared", {
          size: formatCacheSize(outcome.freedBytes),
        }),
      );
      return true;
    } catch (error) {
      toast.error(t("settings.resources.storage.caches.clearFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
      return false;
    } finally {
      setClearing(false);
      if (request === latestRequest.current) setLoading(false);
    }
  };

  return {
    inventory,
    loading,
    loadError,
    clearing,
    entries,
    bulkKeys,
    reclaimable,
    refresh,
    runPurge,
  };
}

export type CacheStorage = ReturnType<typeof useCacheStorage>;

export function CacheStorageRows({ cache }: { cache: CacheStorage }) {
  const t = useT();
  const {
    inventory,
    loading,
    loadError,
    clearing,
    entries,
    bulkKeys,
    reclaimable,
  } = cache;
  const [expanded, setExpanded] = useState(false);
  const [target, setTarget] = useState<PurgeTarget | null>(null);
  const runPurge = async (keys: readonly CacheKey[]) => {
    if (await cache.runPurge(keys)) setTarget(null);
  };

  const confirmTitle =
    target?.kind === "single"
      ? t("settings.resources.storage.caches.confirmOneTitle", {
          name: t(CACHE_NAME_KEYS[target.key]),
        })
      : t("settings.resources.storage.caches.confirmTitle");
  const confirmDescription =
    target?.kind === "single"
      ? singleClearDescriptionKeys(target.key)
          .map((key) => t(key))
          .join(" ")
      : `${t("settings.resources.storage.caches.confirmDescription", {
          size: formatCacheSize(reclaimable),
        })} ${t("settings.resources.storage.caches.safety")}`;

  return (
    <>
      <SettingsCardRow
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
        <Button
          variant="outline"
          size="sm"
          disabled={loading || clearing}
          onClick={() => setExpanded((open) => !open)}
        >
          {expanded
            ? t("settings.resources.storage.caches.hideDetailsAction")
            : t("settings.resources.storage.caches.detailsAction")}
        </Button>
        <Button
          variant="destructive"
          size="sm"
          disabled={
            loading || clearing || loadError !== null || bulkKeys.length === 0
          }
          onClick={() => setTarget({ kind: "bulk" })}
        >
          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
          {t("settings.resources.storage.caches.clearAction")}
        </Button>
      </SettingsCardRow>

      {expanded ? (
        <div className="flex flex-col pb-2">
          {entries.length === 0 ? (
            <p className="py-2 text-xs text-muted-foreground">
              {t("settings.resources.storage.caches.empty")}
            </p>
          ) : (
            entries.map((entry) => (
              <div
                key={entry.key}
                className="flex flex-wrap items-start justify-between gap-2 border-t border-border/60 py-2.5"
              >
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm text-foreground">
                    {t(CACHE_NAME_KEYS[entry.key])}
                  </span>
                  <span
                    title={entry.paths.join("\n")}
                    className="max-w-lg truncate font-mono text-ui-11 text-muted-foreground"
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
                <div data-title-line-control={true} className="flex gap-3">
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {formatCacheSize(entry.sizeBytes)}
                  </span>
                  <Button
                    variant="ghost"
                    size="xs"
                    disabled={
                      loading ||
                      clearing ||
                      loadError !== null ||
                      !entry.purgeable ||
                      entry.entryCount === 0
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
              variant="destructive"
              // A failed measurement is the same hazard as one still running.
              disabled={loading || clearing || loadError !== null}
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
