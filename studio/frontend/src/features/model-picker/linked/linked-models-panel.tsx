// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import type {
  CachedGgufRepo,
  CachedModelRepo,
  GgufVariantDetail,
} from "@/features/hub/inventory/api";
import type {
  LinkedInstance,
  LinkedInstanceInfo,
} from "@/features/settings/api/linked-instances";
import { formatBytes } from "@/features/hub/lib/format";
import { gpuPool } from "@/features/settings/components/linked-instance-format";
import type { VramFitStatus } from "@/lib/vram";
import { cn } from "@/lib/utils";
import { CloudServerIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  type CatalogGroup,
  type ModelArtifact,
} from "../components/model-selector/model-catalog";
import {
  CapabilityScope,
  ModelRow,
  SizeText,
  VramBadge,
} from "../components/model-selector/pickers";
import type { ModelSelectorChangeMeta } from "../components/model-selector/types";
import { linkedCachedModels, linkedGgufVariants } from "./linked-api";

type Section = "recommended" | "downloaded";


const GB = 1024 ** 3;

function formatSize(bytes?: number | null): string | null {
  if (!bytes || bytes <= 0) return null;
  const gb = bytes / GB;
  return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
}

// Weights plus a working margin against the linked machine's first GPU.
function fitOn(
  vramGb: number | null,
  bytes?: number | null,
): VramFitStatus | null {
  if (!vramGb || !bytes) return null;
  const need = (bytes / GB) * 1.15;
  if (need <= vramGb * 0.9) return "fits";
  if (need <= vramGb) return "tight";
  return "exceeds";
}

function useLinkedInventory(instanceId: string) {
  const [state, setState] = useState<{
    loading: boolean;
    error: string | null;
    gguf: CachedGgufRepo[];
    models: CachedModelRepo[];
  }>({ loading: true, error: null, gguf: [], models: [] });

  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    linkedCachedModels(instanceId)
      .then((cached) => {
        if (live) setState({ loading: false, error: null, ...cached });
      })
      .catch((error: unknown) => {
        if (live) {
          setState((s) => ({
            ...s,
            loading: false,
            error: error instanceof Error ? error.message : String(error),
          }));
        }
      });
    return () => {
      live = false;
    };
  }, [instanceId]);
  return state;
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-2.5 pt-2 pb-1 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

function Note({ children }: { children: ReactNode }) {
  return (
    <div className="px-2.5 py-2 text-xs leading-relaxed text-muted-foreground">
      {children}
    </div>
  );
}

/** A GGUF repo's quants on the linked machine; On Device shows only the ones already there. */
function VariantRows({
  instanceId,
  repoId,
  downloadedOnly,
  vramGb,
  selectedVariant,
  onPick,
}: {
  instanceId: string;
  repoId: string;
  downloadedOnly: boolean;
  vramGb: number | null;
  selectedVariant: string | null;
  onPick: (variant: GgufVariantDetail) => void;
}) {
  const [variants, setVariants] = useState<GgufVariantDetail[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    linkedGgufVariants(instanceId, repoId)
      .then((res) => live && setVariants(res.variants))
      .catch(
        (e: unknown) =>
          live && setError(e instanceof Error ? e.message : String(e)),
      );
    return () => {
      live = false;
    };
  }, [instanceId, repoId]);

  if (error) return <Note>{error}</Note>;
  if (!variants) {
    return (
      <div className="flex items-center gap-2 py-1.5 pl-7 text-xs text-muted-foreground">
        <Spinner className="size-3.5" /> Reading quants
      </div>
    );
  }
  const shown = downloadedOnly
    ? variants.filter((v) => v.downloaded)
    : variants;
  if (shown.length === 0) return <Note>No quants found.</Note>;
  return (
    <div className="my-1 ml-3 border-l-2 border-accent/50 pl-4">
      {shown.map((variant) => {
        const status = variant.downloaded
          ? "downloaded"
          : variant.partial
            ? "partial"
            : null;
        return (
          <button
            key={variant.filename}
            type="button"
            onClick={() => onPick(variant)}
            className={cn(
              "flex w-full min-w-0 items-center justify-between gap-2 rounded-full py-1 pr-1.5 pl-2 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              selectedVariant === variant.quant && "bg-sidebar-accent",
            )}
          >
            <span className="min-w-0 flex-1 truncate font-mono text-xs">
              {variant.display_label || variant.quant}
              {status ? (
                <span
                  className={cn(
                    "ml-1.5 font-sans text-ui-9 font-medium",
                    status === "downloaded"
                      ? "text-green-600/90 dark:text-green-400/80"
                      : "text-amber-700 dark:text-amber-300",
                  )}
                >
                  {status}
                </span>
              ) : null}
            </span>
            <span className="flex shrink-0 items-center gap-1.5">
              <VramBadge status={fitOn(vramGb, variant.size_bytes)} />
              <span className="font-mono text-ui-10 tabular-nums text-muted-foreground">
                <SizeText
                  value={formatBytes(
                    variant.download_size_bytes || variant.size_bytes,
                  )}
                />
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** Recommended and On Device for a linked machine: what it has, and what to download there. */
export function LinkedModelsPanel({
  instance,
  info,
  section,
  sectionToggle,
  catalog,
  value,
  onPickImage,
}: {
  instance: LinkedInstance;
  info?: LinkedInstanceInfo;
  section: Section;
  sectionToggle: ReactNode;
  catalog?: CatalogGroup[];
  value?: string;
  onPickImage: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const inventory = useLinkedInventory(instance.id);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  // llama.cpp splits a GGUF across every card, so fit is judged against the pool.
  const pool = info?.gpus?.length ? gpuPool(info.gpus) : null;
  const vramGb = pool?.total ?? null;
  const needle = query.trim().toLowerCase();
  const matches = (text: string) =>
    !needle || text.toLowerCase().includes(needle);

  const cachedIds = useMemo(() => {
    const ids = new Set<string>();
    for (const repo of [...inventory.gguf, ...inventory.models]) {
      if (!repo.partial) ids.add(repo.repo_id.toLowerCase());
    }
    return ids;
  }, [inventory.gguf, inventory.models]);

  const isSelected = (repoId: string) =>
    !!value && value.toLowerCase().startsWith(repoId.toLowerCase());

  const imageArtifact = (group: CatalogGroup, artifact: ModelArtifact) => {
    const key = `${group.canonicalId}::${artifact.repoId}`;
    const gguf = artifact.loadKind === "gguf";
    const open = expanded === key;
    const downloaded = cachedIds.has(artifact.repoId.toLowerCase());
    const base: ModelSelectorChangeMeta = {
      source: "hub",
      isLora: false,
      linkedInstanceId: instance.id,
    };
    return (
      <div key={key}>
        <ModelRow
          label={artifact.repoId}
          meta={
            artifact.approxSizeGb
              ? `${artifact.label} · ${artifact.approxSizeGb} GB`
              : artifact.label
          }
          selected={!open && isSelected(artifact.repoId)}
          downloaded={downloaded}
          alignMeta="hub"
          vramStatus={
            gguf
              ? null
              : fitOn(
                  vramGb,
                  artifact.approxSizeGb ? artifact.approxSizeGb * GB : null,
                )
          }
          gpuGb={vramGb ?? undefined}
          onClick={() =>
            gguf
              ? setExpanded(open ? null : key)
              : onPickImage(artifact.repoId, {
                  ...base,
                  isDownloaded: downloaded,
                })
          }
        />
        {gguf && open ? (
          <VariantRows
            instanceId={instance.id}
            repoId={artifact.repoId}
            downloadedOnly={section === "downloaded"}
            vramGb={vramGb}
            selectedVariant={null}
            onPick={(variant) =>
              onPickImage(artifact.repoId, {
                ...base,
                ggufVariant: variant.quant,
                ggufFilename: variant.filename,
                isDownloaded: !!variant.downloaded,
              })
            }
          />
        ) : null}
      </div>
    );
  };

  let body: ReactNode;
  if (inventory.loading) {
    body = (
      <div className="flex items-center gap-2 px-2.5 py-3 text-xs text-muted-foreground">
        <Spinner className="size-3.5" /> Reading models on @{instance.name}
      </div>
    );
  } else if (inventory.error) {
    body = (
      <Note>
        Couldn't list models on @{instance.name}: {inventory.error}
      </Note>
    );
  } else {
    const groups = (catalog ?? []).filter(
      (g) =>
        matches(g.displayName) || g.artifacts.some((a) => matches(a.repoId)),
    );
    const shown =
      section === "downloaded"
        ? groups
            .map((g) => ({
              group: g,
              artifacts: g.artifacts.filter((a) =>
                cachedIds.has(a.repoId.toLowerCase()),
              ),
            }))
            .filter((g) => g.artifacts.length > 0)
        : groups.map((g) => ({ group: g, artifacts: g.artifacts }));
    body =
      shown.length === 0 ? (
        <Note>
          {section === "downloaded"
            ? `No image models downloaded on @${instance.name} yet. Pick one from Recommended and it downloads there.`
            : "No models match your search."}
        </Note>
      ) : (
        shown.map(({ group, artifacts }) => (
          <div key={group.canonicalId}>
            <SectionLabel>
              {group.displayName}
              <span className="ml-1.5 font-normal normal-case tracking-normal">
                {group.description}
              </span>
            </SectionLabel>
            {artifacts.map((a) => imageArtifact(group, a))}
          </div>
        ))
      );
  }

  return (
    <CapabilityScope.Provider value={[]}>
      <div className="relative space-y-2">
        <div className="flex items-center gap-2 pr-2 pb-1">
          <div className="relative flex-1">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={`Search models on @${instance.name}`}
              data-model-picker-search-input={true}
              className="field-soft h-(--picker-control-h) border-0 pl-8 pr-8"
            />
          </div>
        </div>
        <div className="-mr-2 flex flex-wrap items-center gap-2">
          {sectionToggle}
        </div>
        <div className="flex items-center gap-2 rounded-xl bg-muted/50 px-2.5 py-2 text-ui-11 text-muted-foreground mr-2">
          <HugeiconsIcon
            icon={CloudServerIcon}
            className="size-3.5 shrink-0 text-foreground"
          />
          <span className="min-w-0">
            <span className="font-medium text-foreground">
              @{instance.name}
            </span>
            {pool
              ? ` · ${pool.label}${vramGb ? ` · ${formatSize(vramGb * GB)}` : ""}`
              : info?.online
                ? " · CPU only"
                : ""}
            {" · "}
            <span className="whitespace-nowrap">
              downloads and loads run there
            </span>
          </span>
        </div>
        <div className="model-list-scroll mr-1 max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto px-0.5 pb-4">
          {body}
        </div>
      </div>
    </CapabilityScope.Provider>
  );
}
