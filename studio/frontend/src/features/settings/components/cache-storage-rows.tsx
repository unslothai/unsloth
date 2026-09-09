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
import { useCallback, useEffect, useRef, useState } from "react";
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

/** Caches whose contents are the repositories the Hub inventory reports. */
const HUB_INVENTORY_KEYS: ReadonlySet<CacheKey> = new Set<CacheKey>([
  "hf_hub",
  "hf_datasets",
]);

/** What a confirmation is about: everything reclaimable, or one opt-in cache. */
type PurgeTarget = { kind: "bulk" } | { kind: "single"; key: CacheKey };

/**
 * The body of the confirmation for clearing one cache, as translation keys.
 *
 * The generic assurance ends with "downloaded models ... are not touched". That
 * is true of a bulk clear, which never includes an opt-in cache, and it is the
 * opposite of the truth for the model cache itself: putting both sentences in
 * one dialog contradicts itself immediately before deleting those models. The
 * hub clear therefore says only what it costs.
 */
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

export function CacheStorageRows() {
  const t = useT();
  const [inventory, setInventory] = useState<CacheInventory | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [target, setTarget] = useState<PurgeTarget | null>(null);
  const [clearing, setClearing] = useState(false);

  // A measurement installs itself only while it is still the newest one asked
  // for. Two can overlap, the mount's load still walking a large hub when a
  // folder save starts a forced one, and they finish in whichever order the
  // walks happen to take, so without this the rows can settle on the folder the
  // user moved off. A purge claims a number too, so a walk that started before
  // it cannot overwrite the post-purge inventory.
  const latestRequest = useRef(0);

  const refresh = useCallback(
    async (options: { refresh?: boolean } = {}) => {
      const request = ++latestRequest.current;
      setLoading(true);
      try {
        const next = await loadCacheInventory(options);
        if (request !== latestRequest.current) return;
        setInventory(next);
        setLoadError(null);
      } catch (error) {
        if (request !== latestRequest.current) return;
        setLoadError(
          error instanceof Error
            ? error.message
            : t("settings.resources.storage.caches.measureFailed"),
        );
      } finally {
        if (request === latestRequest.current) setLoading(false);
      }
    },
    [t],
  );

  // Models Folder sits directly above these rows in the same section, and saving
  // it bumps the inventory version. Without this the Hugging Face rows would go
  // on showing the path and the size of the folder the user just moved off,
  // beside the field that now names the new one, with a Clear button that acts
  // on the new one.
  const inventoryVersion = useInventoryVersion();
  const measuredVersion = useRef(inventoryVersion);

  useEffect(() => {
    // The backend memoises a size for a minute, so the reading that has to be
    // thrown away is exactly the one a plain load would return. The first load
    // is not forced: a cold walk of a large uv cache costs tens of seconds.
    const moved = measuredVersion.current !== inventoryVersion;
    measuredVersion.current = inventoryVersion;
    void refresh(moved ? { refresh: true } : {});
  }, [refresh, inventoryVersion]);

  const entries = presentCaches(inventory);
  const bulkKeys = inventory ? bulkPurgeKeys(inventory) : [];
  const reclaimable = inventory?.reclaimableBytes ?? 0;

  const runPurge = async (keys: readonly CacheKey[]) => {
    setClearing(true);
    const request = ++latestRequest.current;
    try {
      const outcome = await purgeCaches(keys);
      if (request === latestRequest.current) setInventory(outcome.inventory);
      if (keys.some((key) => HUB_INVENTORY_KEYS.has(key))) {
        // Every cached model or dataset just went, so the Hub and the model
        // picker have to hear about it the way they do for a delete. The mark
        // takes our own bump: the rows already hold the post-purge inventory,
        // and reading it as a folder move would buy a cold walk for nothing.
        bumpInventoryVersion();
        measuredVersion.current = getInventoryVersion();
      }
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
      // Whatever measurement this superseded will not clear it.
      if (request === latestRequest.current) setLoading(false);
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
      ? singleClearDescriptionKeys(target.key)
          .map((key) => t(key))
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
            disabled={
              loading || clearing || loadError !== null || bulkKeys.length === 0
            }
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
                    // loading and loadError, like the buttons above: these rows
                    // show the last inventory that arrived, and a clear resolves
                    // its key against the current one, so a measurement in
                    // flight or a failed one means the two can disagree.
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
              // A failed measurement is the same hazard as one still running:
              // the rows are the old folder's and the purge resolves the new.
              disabled={loading || clearing || loadError !== null}
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
