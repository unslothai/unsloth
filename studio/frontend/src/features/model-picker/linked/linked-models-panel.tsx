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
import { gpuPool } from "@/features/settings/components/linked-instance-format";
import type { VramFitStatus } from "@/lib/vram";
import { cn } from "@/lib/utils";
import { CloudServerIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  type CatalogGroup,
  type ModelArtifact,
  AUDIO_CATALOG,
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  artifactForRepoId,
} from "../components/model-selector/model-catalog";
import {
  CapabilityScope,
  ModelRow,
} from "../components/model-selector/pickers";
import { detectCapabilities } from "../components/model-selector/model-capabilities";
import type { ModelSelectorChangeMeta } from "../components/model-selector/types";
import {
  linkedCachedModels,
  linkedDefaultModels,
  linkedGgufVariants,
} from "./linked-api";
import type { LinkedPickerKind } from "./linked-machines";

export type LinkedChatPick = {
  repoId: string;
  loadId?: string | null;
  variant?: string | null;
  downloaded: boolean;
  expectedBytes?: number;
};

type Section = "recommended" | "downloaded";

const MEDIA_CATALOGS = [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG];

const GB = 1024 ** 3;
const isGgufRepo = (id: string) => /gguf/i.test(id);
const leaf = (id: string) => id.slice(id.lastIndexOf("/") + 1);

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

function useLinkedInventory(instanceId: string, kind: LinkedPickerKind) {
  const [state, setState] = useState<{
    loading: boolean;
    error: string | null;
    defaults: string[];
    gguf: CachedGgufRepo[];
    models: CachedModelRepo[];
  }>({ loading: true, error: null, defaults: [], gguf: [], models: [] });

  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    Promise.all([
      linkedCachedModels(instanceId),
      kind === "chat" ? linkedDefaultModels(instanceId) : Promise.resolve([]),
    ])
      .then(([cached, defaults]) => {
        if (live)
          setState({ loading: false, error: null, defaults, ...cached });
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
  }, [instanceId, kind]);
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
    <div className="ml-4 border-l border-border/60 pl-1">
      {shown.map((variant) => {
        const bytes = variant.download_size_bytes || variant.size_bytes;
        const fit = fitOn(vramGb, variant.size_bytes);
        return (
          <ModelRow
            key={variant.filename}
            label={variant.display_label || variant.quant}
            meta={formatSize(bytes)}
            hideOwner={true}
            selected={selectedVariant === variant.quant}
            downloaded={variant.downloaded}
            partial={variant.partial}
            vramStatus={fit}
            vramEst={
              variant.size_bytes
                ? Math.ceil((variant.size_bytes / GB) * 1.15)
                : undefined
            }
            gpuGb={vramGb ?? undefined}
            alignMeta="hub"
            onClick={() => onPick(variant)}
          />
        );
      })}
    </div>
  );
}

/** Recommended and On Device for a linked machine: what it has, and what to download there. */
export function LinkedModelsPanel({
  kind,
  instance,
  info,
  section,
  sectionToggle,
  catalog,
  value,
  onPickChat,
  onPickImage,
}: {
  kind: LinkedPickerKind;
  instance: LinkedInstance;
  info?: LinkedInstanceInfo;
  section: Section;
  sectionToggle: ReactNode;
  catalog?: CatalogGroup[];
  value?: string;
  onPickChat: (pick: LinkedChatPick) => void;
  onPickImage: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const inventory = useLinkedInventory(instance.id, kind);
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

  // Chat value is "@name/<repo>[:quant]"; image picks follow the Images page's own value.
  const prefix = `@${instance.name}/`;
  const selectedRepo = value?.startsWith(prefix)
    ? value.slice(prefix.length)
    : value;
  const isSelected = (repoId: string) =>
    !!selectedRepo &&
    selectedRepo.toLowerCase().startsWith(repoId.toLowerCase());

  const imageRepo = (repoId: string) =>
    catalog ? artifactForRepoId(repoId, catalog) !== null : false;
  // Chat's On Device skips media models; the remote's cache reports no task for most GGUFs.
  const isImageRepo = (repo: CachedGgufRepo | CachedModelRepo) =>
    imageRepo(repo.repo_id) ||
    MEDIA_CATALOGS.some((c) => artifactForRepoId(repo.repo_id, c) !== null) ||
    ((caps) => caps.imageGen || caps.videoGen)(
      detectCapabilities({
        id: repo.repo_id,
        tags: repo.tags,
        pipelineTag: repo.pipeline_tag ?? undefined,
      }),
    ) ||
    /image|diffusion|video|audio|speech/i.test(repo.pipeline_tag ?? "");

  const chatRepo = (
    repoId: string,
    meta: string | null,
    downloaded: boolean,
    extra?: { loadId?: string | null; bytes?: number },
  ) => {
    const gguf = isGgufRepo(repoId);
    const open = expanded === repoId;
    return (
      <div key={repoId}>
        <ModelRow
          label={repoId}
          meta={meta}
          selected={!open && isSelected(repoId)}
          downloaded={downloaded}
          alignMeta="hub"
          vramStatus={gguf ? null : fitOn(vramGb, extra?.bytes)}
          gpuGb={vramGb ?? undefined}
          onClick={() =>
            gguf
              ? setExpanded(open ? null : repoId)
              : onPickChat({
                  repoId,
                  loadId: extra?.loadId,
                  downloaded,
                  expectedBytes: extra?.bytes,
                })
          }
          tags={gguf ? ["GGUF"] : undefined}
        />
        {gguf && open ? (
          <VariantRows
            instanceId={instance.id}
            repoId={repoId}
            downloadedOnly={section === "downloaded"}
            vramGb={vramGb}
            selectedVariant={
              isSelected(repoId) ? (selectedRepo?.split(":")[1] ?? null) : null
            }
            onPick={(variant) =>
              onPickChat({
                repoId,
                variant: variant.quant,
                downloaded: !!variant.downloaded,
                expectedBytes:
                  variant.download_size_bytes || variant.size_bytes,
              })
            }
          />
        ) : null}
      </div>
    );
  };

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
  } else if (kind === "chat" && section === "recommended") {
    const ids = inventory.defaults.filter((id) => matches(id));
    body = ids.length ? (
      ids.map((id) => chatRepo(id, null, cachedIds.has(id.toLowerCase())))
    ) : (
      <Note>No models match your search.</Note>
    );
  } else if (kind === "chat") {
    const gguf = inventory.gguf.filter(
      (r) => !isImageRepo(r) && matches(r.repo_id),
    );
    const models = inventory.models.filter(
      (r) => !isImageRepo(r) && matches(r.repo_id),
    );
    body =
      gguf.length + models.length === 0 ? (
        <Note>
          Nothing downloaded on @{instance.name} yet. Pick one from Recommended
          and it downloads there.
        </Note>
      ) : (
        <>
          {gguf.map((r) => chatRepo(r.repo_id, formatSize(r.size_bytes), true))}
          {models.map((r) =>
            chatRepo(r.repo_id, formatSize(r.size_bytes), !r.partial, {
              loadId: r.load_id,
              bytes: r.size_bytes,
            }),
          )}
        </>
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
    <CapabilityScope.Provider value={kind === "image" ? [] : null}>
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
            {" · downloads and loads run there"}
          </span>
        </div>
        <div className="model-list-scroll mr-1 max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto px-0.5 pb-4">
          {body}
        </div>
      </div>
    </CapabilityScope.Provider>
  );
}
