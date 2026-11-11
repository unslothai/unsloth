// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelMemoryBar } from "@/components/model-memory-bar";
import { SectionBox } from "@/components/section-box";
import { SegmentedControl } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Spinner } from "@/components/ui/spinner";
import { useChatRuntimeStore } from "@/features/chat";
import {
  detectBaseModel,
  type GgufVariantDetail,
  type LocalModelInfo,
  hfApiToken,
  listGgufVariants,
  useHfTokenStore,
  useHubModelSearch,
} from "@/features/hub";
import {
  formatFootprintBytes,
  ListLabel,
  ModelRow,
} from "@/features/model-picker";
import { useNpuStatus } from "@/features/npu";
import { useDebouncedValue } from "@/hooks";
import { getHfEndpoint } from "@/lib/hf-endpoint";
import { hubFetch } from "@/lib/hub-fetch";
import { FolderSearchIcon } from "@/lib/folder-icons";
import { cn } from "@/lib/utils";
import { CpuIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { modelInfo } from "@huggingface/hub";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { convertGgufToQ4nx } from "../api/export-api";
import {
  getCachedLocalModels,
  refreshLocalModels,
} from "../export-navigation-cache";
import {
  localGgufPath,
  packsDirectly,
  Q4NX_INFO_FIELDS,
  type Q4nxMemory,
  q4nxArchFit,
  q4nxBaseModel,
  q4nxIsExpanded,
  q4nxMemoryWarning,
  q4nxNpuMemory,
  q4nxParamsFromName,
  q4nxQuantLabel,
  q4nxSummaryRows,
  q4nxToggleExpanded,
  q4nxVariantLookup,
  q4nxWeights,
  sortQ4nxVariants,
} from "../lib/q4nx-source";

type Source = "hub" | "local";
const SOURCE_OPTIONS = [
  { value: "hub", label: "Hugging Face" },
  { value: "local", label: "Local file" },
] as const satisfies readonly { value: Source; label: string }[];

const FIELD_LABEL =
  "flex items-center gap-1.5 text-xs font-medium text-muted-foreground";
const HUB_REPO_RE = /^[^/\s]+\/[^/\s]+$/;
const GGUF_FILE_RE = /\.gguf$/i;
const LISTING_WORKERS = 4;
const LOCAL_SOURCE_HINT: Record<string, string> = {
  hf_cache: "HF cache",
  custom: "Custom Folders",
};

interface SourceRow {
  id: string;
  label: string;
  /** Lists the model's GGUF files; null for a row that is one file. */
  listingKey: string | null;
  folder: string | null;
  hint?: string;
  sizeBytes?: number | null;
  totalParams?: number | null;
}

interface Listing {
  list: GgufVariantDetail[];
  error: string | null;
}
const NO_LISTINGS: Record<string, Listing> = {};

function isGgufRow(row: LocalModelInfo): boolean {
  return (
    row.model_format === "gguf" || row.path.toLowerCase().endsWith(".gguf")
  );
}

function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** The chat picker's memory bar over the NPU window: weights, then the load margin. */
function NpuMemoryBar({
  memory,
  windowGb,
  className,
}: {
  memory: Q4nxMemory | null;
  windowGb: number | null;
  className?: string;
}) {
  if (!memory || memory.fits === null || !windowGb) return null;
  return (
    <ModelMemoryBar
      compact={true}
      className={className}
      segments={{
        status: "fits",
        budgetGb: windowGb,
        modelGb: memory.weightsGb,
        kvGb: memory.neededGb - memory.weightsGb,
        specGb: 0,
        totalGb: memory.neededGb,
        modelPct: memory.weightsPct,
        kvPct: memory.marginPct,
        specPct: 0,
        kvBytesPerToken: 0,
        fillPct: (memory.neededGb / windowGb) * 100,
        pressure: memory.fits ? "normal" : "high",
      }}
    />
  );
}

/** A model's files. Opening one by hand scrolls its row to the top of the list, again once the files load, so they are in view. */
function ModelFiles({
  reveal,
  loaded,
  children,
}: {
  reveal: boolean;
  loaded: boolean;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const row = ref.current?.parentElement;
    const list = row?.parentElement;
    if (!reveal || !row || !list) return;
    list.scrollTop +=
      row.getBoundingClientRect().top - list.getBoundingClientRect().top;
  }, [reveal, loaded]);
  return (
    <div ref={ref} className="my-1 ml-3 border-l-2 border-accent/50 pl-4">
      {children}
    </div>
  );
}

/** The card's data-tour value, which the export page uses to tell the card is in use. */
export const Q4NX_CARD_MARK = "export-q4nx";

export interface Q4nxSummary {
  ready: boolean;
  rows: ReturnType<typeof q4nxSummaryRows>;
}

export function Q4nxConvertCard({
  onSummary,
}: {
  /** Feeds the export page's summary panel; pass a stable function. */
  onSummary?: (summary: Q4nxSummary) => void;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const accessToken = hfApiToken(hfToken);
  const expandQuantizations = useChatRuntimeStore((s) => s.expandQuantizations);
  const [npuStatus] = useNpuStatus();
  const windowGb = npuStatus?.memory_window_gb ?? null;
  const [source, setSource] = useState<Source>("hub");
  const [hubQuery, setHubQuery] = useState("");
  const [repoId, setRepoId] = useState("");
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>(
    () => getCachedLocalModels() ?? [],
  );
  const [loadingLocal, setLoadingLocal] = useState(
    () => getCachedLocalModels() === null,
  );
  const [localQuery, setLocalQuery] = useState("");
  const [localId, setLocalId] = useState("");
  const [listed, setListed] = useState<{
    token: string | undefined;
    byKey: Record<string, Listing>;
  }>({ token: undefined, byKey: {} });
  const [toggled, setToggled] = useState<{
    key: string;
    ids: readonly string[];
  }>({ key: "", ids: [] });
  const [filename, setFilename] = useState("");
  const [cardInfo, setCardInfo] = useState<{
    repoId: string;
    baseModel: string | null;
    architecture: string | null;
  } | null>(null);
  const [baseModel, setBaseModel] = useState<string | null>(null);
  const [saveDirectory, setSaveDirectory] = useState("q4nx");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(
    null,
  );

  const debouncedHubQuery = useDebouncedValue(hubQuery);
  const { results: hubResults, isLoading: searchingHub } = useHubModelSearch(
    debouncedHubQuery,
    {
      accessToken,
      channel: { tags: ["gguf"] },
      enabled: source === "hub",
    },
  );

  useEffect(() => {
    if (source !== "local") return;
    let cancelled = false;
    void refreshLocalModels()
      .then((models) => {
        if (!cancelled) setLocalModels(models);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setLoadingLocal(false);
      });
    return () => {
      cancelled = true;
    };
  }, [source]);

  const hubRows = useMemo(() => {
    const rows: SourceRow[] = hubResults
      .map((r) => ({ r, fit: q4nxArchFit(r.ggufArchitecture, r.id) }))
      // An unsupported architecture is left out; an unsupported size of a supported one stays, marked.
      .filter(({ fit }) => fit.status !== "unsupported" || fit.sized)
      .map(({ r, fit }) => ({
        id: r.id,
        label: r.id,
        listingKey: q4nxVariantLookup("hub", r.id, null).key,
        folder: null,
        hint: fit.status === "unsupported" ? "Not convertible" : undefined,
        totalParams: r.totalParams ?? q4nxParamsFromName(r.id),
      }));
    // A typed repo id the search does not return is used as is; the pick stays listed.
    const typed = hubQuery.trim();
    for (const id of [HUB_REPO_RE.test(typed) ? typed : "", repoId]) {
      if (!id || rows.some((row) => row.id === id)) continue;
      rows.unshift({
        id,
        label: id,
        listingKey: q4nxVariantLookup("hub", id, null).key,
        folder: null,
        totalParams: q4nxParamsFromName(id),
      });
    }
    return rows;
  }, [hubResults, hubQuery, repoId]);

  const localById = useMemo(
    () => new Map(localModels.map((row) => [row.id, row])),
    [localModels],
  );
  const localRows = useMemo(() => {
    const toRow = (row: LocalModelInfo): SourceRow => {
      const single = GGUF_FILE_RE.test(row.path.trim());
      return {
        id: row.id,
        label: row.display_name,
        listingKey: single ? null : q4nxVariantLookup("local", "", row.path).key,
        folder: single ? null : row.path,
        hint: LOCAL_SOURCE_HINT[row.source] ?? "Local dir",
        sizeBytes: row.size_bytes,
        totalParams: q4nxParamsFromName(row.display_name),
      };
    };
    const q = localQuery.trim().toLowerCase();
    const rows = localModels
      .filter(
        (row) =>
          isGgufRow(row) &&
          (!q ||
            row.id.toLowerCase().includes(q) ||
            row.display_name.toLowerCase().includes(q) ||
            row.path.toLowerCase().includes(q)),
      )
      .map(toRow);
    // A typed path that is not a known row is used as the file itself.
    const typed = localQuery.trim();
    for (const id of [GGUF_FILE_RE.test(typed) ? typed : "", localId]) {
      if (!id || rows.some((row) => row.id === id)) continue;
      const known = localById.get(id);
      rows.unshift(
        known && isGgufRow(known)
          ? toRow(known)
          : { id, label: id, listingKey: null, folder: null },
      );
    }
    return rows;
  }, [localModels, localById, localQuery, localId]);

  const rows = source === "hub" ? hubRows : localRows;
  const selectedId = source === "hub" ? repoId : localId;
  // "Expand quantizations" covers On Device in the chat picker, so it covers the local list here.
  const expandAll = source === "local" && expandQuantizations;
  const expansionKey = `${source}:${expandAll}`;
  const expansion = useMemo(
    () => ({
      expandAll,
      toggled: toggled.key === expansionKey ? toggled.ids : [],
    }),
    [expandAll, toggled, expansionKey],
  );

  // Listings are per token: a new one can list a repo the last one could not.
  const listings = listed.token === accessToken ? listed.byKey : NO_LISTINGS;
  const inFlight = useRef(new Set<string>());
  // The token requests settle against; null once unmounted.
  const liveToken = useRef<string | undefined | null>(accessToken);
  useEffect(() => {
    liveToken.current = accessToken;
    return () => {
      liveToken.current = null;
    };
  }, [accessToken]);
  useEffect(() => {
    for (const row of rows) {
      if (inFlight.current.size >= LISTING_WORKERS) return;
      const key = row.listingKey;
      // Local folders are listed up front so a closed row can show its memory need; Hub rows estimate it from the parameter count until opened or picked.
      if (
        !key ||
        (source === "hub" &&
          row.id !== repoId &&
          !q4nxIsExpanded(expansion, row.id))
      ) {
        continue;
      }
      if (listings[key] || inFlight.current.has(key)) continue;
      inFlight.current.add(key);
      const request = row.folder
        ? listGgufVariants(row.folder, accessToken, {
            preferLocalCache: true,
            localPath: row.folder,
          })
        : listGgufVariants(row.id, accessToken);
      const settle = (listing: Listing) => {
        inFlight.current.delete(key);
        if (liveToken.current === null) return;
        // An answer for a replaced token is dropped; the new state reruns this effect.
        const stale = liveToken.current !== accessToken;
        setListed((prev) => ({
          token: stale ? prev.token : accessToken,
          byKey: {
            ...(stale || prev.token === accessToken ? prev.byKey : NO_LISTINGS),
            ...(stale ? null : { [key]: listing }),
          },
        }));
      };
      request
        .then((response) =>
          settle({ list: sortQ4nxVariants(response.variants), error: null }),
        )
        .catch((error: unknown) =>
          settle({
            list: [],
            error: error instanceof Error ? error.message : String(error),
          }),
        );
    }
  }, [source, rows, repoId, expansion, listings, accessToken]);

  const selectedRow = rows.find((row) => row.id === selectedId) ?? null;
  const selectedList = selectedRow?.listingKey
    ? (listings[selectedRow.listingKey]?.list ?? [])
    : [];
  const selectedVariant =
    selectedList.find((v) => v.filename === filename) ?? null;
  const localRow = source === "local" ? selectedRow : null;
  const localInfo = localRow ? localById.get(localRow.id) : undefined;
  const localPath = localRow ? (localInfo?.path ?? localRow.id) : null;

  // The Hub's "Base" chip reads base_model from the card tags; a typed repo is looked up.
  const hubResult = hubResults.find((r) => r.id === repoId);
  useEffect(() => {
    if (source !== "hub" || !repoId || hubResult) return;
    let cancelled = false;
    // Not cachedModelInfo: its shared cache never asks for the gguf expansion this gate reads.
    void modelInfo({
      name: repoId,
      accessToken,
      hubUrl: getHfEndpoint(),
      fetch: hubFetch,
      additionalFields: Q4NX_INFO_FIELDS,
    })
      .then((info) => {
        if (cancelled) return;
        const typed = info as { tags?: string[]; gguf?: { architecture?: string } };
        setCardInfo({
          repoId,
          baseModel: detectBaseModel(typed.tags),
          architecture: typed.gguf?.architecture ?? null,
        });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [source, repoId, hubResult, accessToken]);
  const looked = cardInfo?.repoId === repoId ? cardInfo : null;

  const autoBase =
    source === "hub"
      ? repoId
        ? q4nxBaseModel(repoId, hubResult?.baseModel ?? looked?.baseModel)
        : ""
      : localInfo
        ? q4nxBaseModel(
            localInfo.model_id ??
              (localInfo.source === "hf_cache" ? localInfo.id : ""),
            null,
          )
        : "";
  const effectiveBase = baseModel ?? autoBase;

  const ggufPath =
    localRow && localPath
      ? localGgufPath(localPath, localRow.listingKey ? selectedVariant : null)
      : null;

  const archFit =
    source === "hub" && repoId
      ? q4nxArchFit(
          hubResult?.ggufArchitecture ?? looked?.architecture,
          `${repoId} ${selectedVariant?.filename ?? ""}`,
        )
      : localRow
        ? q4nxArchFit(null, `${localRow.id} ${ggufPath ?? localPath}`)
        : null;
  const sourceReady =
    source === "hub" ? !!(repoId && selectedVariant) : !!ggufPath;
  const canConvert =
    !running &&
    sourceReady &&
    archFit?.status !== "unsupported" &&
    !!effectiveBase.trim() &&
    !!saveDirectory.trim();

  function pick(id: string, file: string) {
    if (id !== selectedId) setBaseModel(null);
    if (source === "hub") setRepoId(id);
    else setLocalId(id);
    setFilename(file);
  }

  function variantMemory(
    variant: GgufVariantDetail | null,
    list: GgufVariantDetail[],
    totalParams?: number | null,
  ) {
    return q4nxNpuMemory(q4nxWeights(variant, list, totalParams), windowGb);
  }

  /** A one-file row's own need; a model's is its picked file's, else its default file's. */
  function rowMemory(row: SourceRow): Q4nxMemory | null {
    if (!row.listingKey) {
      const path = localById.get(row.id)?.path ?? row.id;
      return variantMemory(
        {
          filename: path,
          quant: fileName(path).replace(GGUF_FILE_RE, ""),
          size_bytes: row.sizeBytes ?? 0,
        },
        [],
        row.totalParams,
      );
    }
    const list = listings[row.listingKey]?.list ?? [];
    const variant =
      (row.id === selectedId ? selectedVariant : null) ?? list[0] ?? null;
    return variantMemory(variant, list, row.totalParams);
  }

  const selectedMemory =
    sourceReady && selectedRow
      ? selectedRow.listingKey
        ? variantMemory(selectedVariant, selectedList, selectedRow.totalParams)
        : rowMemory(selectedRow)
      : null;
  const memoryWarning = q4nxMemoryWarning(selectedMemory, windowGb);

  const selectedFile = sourceReady
    ? fileName(source === "hub" ? filename : (ggufPath ?? ""))
    : null;
  const summaryModel = source === "hub" ? repoId : (localRow?.label ?? "");
  const summaryQuant = sourceReady
    ? (selectedVariant?.quant ??
      fileName(ggufPath ?? "").replace(GGUF_FILE_RE, ""))
    : null;
  const selectedWeights =
    sourceReady && selectedRow
      ? q4nxWeights(
          selectedVariant ?? {
            filename: ggufPath ?? "",
            quant: summaryQuant ?? "",
            size_bytes: selectedRow.sizeBytes,
          },
          selectedList,
          selectedRow.totalParams,
        )
      : null;
  const summarySize = selectedWeights
    ? `${selectedWeights.estimated ? "~" : ""}${formatFootprintBytes(selectedWeights.bytes)}`
    : null;
  // Serialized so the page hears about a change of content, not every render.
  const summaryJson = JSON.stringify({
    ready: canConvert,
    rows: q4nxSummaryRows({
      model: summaryModel,
      file: selectedFile,
      quant: summaryQuant,
      baseModel: effectiveBase,
      saveDirectory,
      size: summarySize,
      memory: selectedMemory,
      archFit,
    }),
  } satisfies Q4nxSummary);
  useEffect(() => onSummary?.(JSON.parse(summaryJson)), [onSummary, summaryJson]);

  async function handleConvert() {
    setRunning(true);
    setResult(null);
    try {
      const response = await convertGgufToQ4nx({
        save_directory: saveDirectory.trim(),
        gguf_path: source === "local" ? ggufPath : null,
        repo_id: source === "hub" ? repoId : null,
        filename: source === "hub" ? filename : null,
        base_model: effectiveBase.trim(),
        hf_token: accessToken ?? null,
      });
      setResult({
        ok: true,
        text: `Saved to ${response.details?.output_path ?? saveDirectory}`,
      });
    } catch (error) {
      setResult({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setRunning(false);
    }
  }

  const loadingRows = source === "hub" ? searchingHub : loadingLocal;

  const renderVariants = (row: SourceRow) => {
    const listing = row.listingKey ? listings[row.listingKey] : undefined;
    if (!listing) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
          <Spinner className="size-3.5" /> Loading files...
        </div>
      );
    }
    if (listing.list.length === 0) {
      return (
        <p
          className={cn(
            "px-2 py-1 text-xs",
            listing.error ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {listing.error
            ? `Could not list the files: ${listing.error}`
            : "No GGUF files found"}
        </p>
      );
    }
    return listing.list.map((v) => {
      const picked = row.id === selectedId && v.filename === filename;
      const memory = variantMemory(v, listing.list, row.totalParams);
      return (
        <div key={v.filename}>
          <button
            type="button"
            data-testid="q4nx-source-file"
            aria-pressed={picked}
            onClick={() => pick(row.id, v.filename)}
            className={cn(
              "flex w-full min-w-0 items-center justify-between gap-2 rounded-full py-1 pl-2 pr-1.5 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              picked && "bg-sidebar-accent",
            )}
          >
            <span className="min-w-0 flex-1 truncate font-mono text-xs">
              {fileName(v.filename)}
              {!packsDirectly(v.quant) && (
                <span className="ml-1.5 font-sans text-ui-9 font-medium text-muted-foreground">
                  re-quantized
                </span>
              )}
            </span>
            <span className="flex shrink-0 items-center gap-1.5 font-mono text-ui-10 text-muted-foreground tabular-nums">
              {memory && (
                <span
                  data-testid="q4nx-source-memory"
                  className={cn(
                    memory.fits === false &&
                      "text-amber-600 dark:text-amber-500",
                  )}
                >
                  {memory.label}
                </span>
              )}
              <span>{formatFootprintBytes(v.size_bytes)}</span>
            </span>
          </button>
          {picked && (
            <NpuMemoryBar
              memory={memory}
              windowGb={windowGb}
              className="mb-1 px-2"
            />
          )}
        </div>
      );
    });
  };

  return (
    <SectionBox
      dataTour={Q4NX_CARD_MARK}
      icon={CpuIcon}
      chipTint="var(--chart-1)"
      title="Convert a GGUF to Q4NX (AMD NPU)"
      description="Convert a Q4_0, Q4_1 or Q4_K_M GGUF to Q4NX. Only supported on AMD NPUs."
      titleAction={
        <SegmentedControl
          value={source}
          options={SOURCE_OPTIONS}
          onValueChange={(next) => {
            setSource(next);
            setFilename("");
            setBaseModel(null);
          }}
          ariaLabel="GGUF source"
          size="compact"
          className="@md/train-card:w-[calc(220px*var(--ui-space-scale,1))]"
        />
      }
    >
      <div className="flex flex-col gap-6">
        <div className="grid gap-4 @md/train-section:grid-cols-2">
          <div className="flex flex-col gap-2 @md/train-section:col-span-2">
            <label htmlFor="q4nx-search" className={FIELD_LABEL}>
              GGUF model
            </label>
            <InputGroup>
              <InputGroupAddon>
                <HugeiconsIcon
                  icon={source === "hub" ? Search01Icon : FolderSearchIcon}
                  className="size-4"
                />
              </InputGroupAddon>
              <InputGroupInput
                id="q4nx-search"
                placeholder={
                  source === "hub"
                    ? "Search GGUF models or type owner/name..."
                    : "Filter, or type /path/to/model.Q4_1.gguf"
                }
                value={source === "hub" ? hubQuery : localQuery}
                onChange={(e) =>
                  (source === "hub" ? setHubQuery : setLocalQuery)(
                    e.target.value,
                  )
                }
              />
            </InputGroup>
            <div
              data-testid="q4nx-source-list"
              className="max-h-[calc(280px*var(--ui-space-scale,1))] overflow-y-auto rounded-xl border border-border px-1 pb-1"
            >
              <ListLabel>
                {source === "hub" ? "Hugging Face" : "On device"}
              </ListLabel>
              {rows.map((row) => {
                const memory = rowMemory(row);
                const selected = row.id === selectedId;
                const expanded = q4nxIsExpanded(expansion, row.id);
                return (
                  <div key={row.id} data-testid="q4nx-source-row">
                    <ModelRow
                      label={row.label}
                      meta={[
                        "GGUF",
                        row.hint,
                        memory?.label,
                        memory?.fits === false
                          ? "May not fit NPU memory"
                          : null,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                      selected={selected}
                      quantChip={
                        selected && selectedVariant
                          ? q4nxQuantLabel(selectedVariant.quant)
                          : null
                      }
                      vramStatus={null}
                      onClick={() => {
                        if (!row.listingKey) {
                          pick(row.id, "");
                          return;
                        }
                        const next = q4nxToggleExpanded(expansion, row.id);
                        setToggled({ key: expansionKey, ids: next.toggled });
                        // Reopening a model whose files failed to list asks again.
                        const key = row.listingKey;
                        if (q4nxIsExpanded(next, row.id) && listings[key]?.error) {
                          setListed((prev) => ({
                            ...prev,
                            byKey: Object.fromEntries(
                              Object.entries(prev.byKey).filter(
                                ([k]) => k !== key,
                              ),
                            ),
                          }));
                        }
                      }}
                    />
                    {!row.listingKey && selected && (
                      <NpuMemoryBar
                        memory={memory}
                        windowGb={windowGb}
                        className="mb-1 px-3"
                      />
                    )}
                    {expanded && row.listingKey && (
                      <ModelFiles
                        reveal={expansion.toggled.includes(row.id)}
                        loaded={row.listingKey in listings}
                      >
                        {renderVariants(row)}
                      </ModelFiles>
                    )}
                  </div>
                );
              })}
              {loadingRows ? (
                <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                  <Spinner className="size-4" />
                  {source === "hub" ? "Searching..." : "Scanning..."}
                </div>
              ) : rows.length === 0 ? (
                <p className="py-4 text-center text-xs text-muted-foreground">
                  {source === "hub"
                    ? "No GGUF models found"
                    : "No local GGUF models found"}
                </p>
              ) : null}
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="q4nx-base" className={FIELD_LABEL}>
              Original model (config and tokenizer)
            </label>
            <Input
              id="q4nx-base"
              placeholder="unsloth/Qwen3-0.6B"
              value={effectiveBase}
              onChange={(e) => setBaseModel(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="q4nx-save" className={FIELD_LABEL}>
              Save directory
            </label>
            <Input
              id="q4nx-save"
              value={saveDirectory}
              onChange={(e) => setSaveDirectory(e.target.value)}
            />
          </div>
        </div>
        {(archFit?.note ||
          memoryWarning ||
          (selectedVariant && !packsDirectly(selectedVariant.quant))) && (
          <div className="flex flex-col gap-1">
            {archFit?.note && (
              <p
                className={
                  archFit.status === "unsupported"
                    ? "text-xs text-destructive"
                    : "text-xs text-muted-foreground"
                }
              >
                {archFit.note}
              </p>
            )}
            {memoryWarning && (
              <p
                data-testid="q4nx-memory-warning"
                className="text-xs text-amber-600 dark:text-amber-500"
              >
                {memoryWarning}
              </p>
            )}
            {selectedVariant && !packsDirectly(selectedVariant.quant) && (
              <p className="text-xs text-muted-foreground">
                {q4nxQuantLabel(selectedVariant.quant)} is dequantized and rounded again to
                4-bit, so it loses some quality. Q4_0, Q4_1 and Q4_K_M pack
                directly.
              </p>
            )}
          </div>
        )}
        <div className="flex items-center justify-between gap-3">
          <p
            className={
              result?.ok === false
                ? "text-xs text-destructive"
                : "text-xs text-muted-foreground"
            }
          >
            {result?.text ??
              (selectedFile
                ? `Selected ${selectedFile}. No training or GPU needed.`
                : "Pick a GGUF file. No training or GPU needed.")}
          </p>
          <Button variant="dark" disabled={!canConvert} onClick={handleConvert}>
            {running && <Spinner />}
            Convert
          </Button>
        </div>
      </div>
    </SectionBox>
  );
}
