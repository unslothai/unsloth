// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ModelMemoryBar } from "@/components/model-memory-bar";
import { SectionBox } from "@/components/section-box";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent } from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { TooltipProvider } from "@/components/ui/tooltip";
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
  ModelLoadSettingsAction,
  ModelRow,
  ModelSelector,
  PillTabs,
} from "@/features/model-picker";
import { useNpuStatus } from "@/features/npu";
import { useDebouncedValue } from "@/hooks";
import { ChevronLeftStandardIcon } from "@/lib/chevron-icons";
import { getHfEndpoint } from "@/lib/hf-endpoint";
import { hubFetch } from "@/lib/hub-fetch";
import { cn } from "@/lib/utils";
import {
  CpuIcon,
  Download01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
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
  Q4NX_INFO_FIELDS,
  Q4NX_PACKING_TEXT,
  type Q4nxMemory,
  q4nxArchFit,
  q4nxBaseModel,
  q4nxIsExpanded,
  q4nxConvertibleFiles,
  q4nxIsRefusal,
  q4nxMemoryWarning,
  q4nxNpuMemory,
  q4nxPacking,
  q4nxParamsFromName,
  q4nxQuantName,
  q4nxSummaryRows,
  q4nxToggleExpanded,
  q4nxVariantLookup,
  q4nxWeights,
  sortQ4nxVariants,
} from "../lib/q4nx-source";

type Source = "hub" | "local";
const SOURCE_TABS = [
  {
    value: "hub",
    label: "Hugging Face",
    icon: <HugeiconsIcon icon={Search01Icon} className="size-3.5 shrink-0" />,
  },
  {
    value: "local",
    label: "On Device",
    icon: <HugeiconsIcon icon={Download01Icon} className="size-3.5 shrink-0" />,
  },
];

const FIELD_LABEL =
  "flex items-center gap-1.5 text-xs font-medium text-muted-foreground";
const HUB_REPO_RE = /^[^/\s]+\/[^/\s]+$/;
const GGUF_FILE_RE = /\.gguf$/i;
const LISTING_WORKERS = 4;
const LOCAL_SOURCE_HINT: Record<string, string> = {
  hf_cache: "HF cache",
  custom: "Custom Folders",
};
const DEFAULT_SAVE_DIRECTORY = "q4nx";

/** The card's data-tour value, which the export page uses to tell the card is in use. */
export const Q4NX_CARD_MARK = "export-q4nx";

export interface Q4nxSummary {
  ready: boolean;
  rows: ReturnType<typeof q4nxSummaryRows>;
}

interface SourceRow {
  id: string;
  label: string;
  /** Lists the model's GGUF files; null for a row that is one file. */
  listingKey: string | null;
  folder: string | null;
  hint?: string;
  architecture?: string | null;
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
    const list = row?.closest<HTMLElement>("[data-q4nx-list]");
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
  // Reported by the backend from #17531 on; without it rows show the need alone.
  const windowGb = npuStatus?.memory_window_gb ?? null;
  const [open, setOpen] = useState(false);
  // The popover's second page, like the chat picker's run settings.
  const [settingsOpen, setSettingsOpen] = useState(false);
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
  const [saveDirectory, setSaveDirectory] = useState(DEFAULT_SAVE_DIRECTORY);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{
    ok: boolean;
    text: string;
    /** The request the backend refused; asking again would get the same answer. */
    refused?: string;
  } | null>(null);

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

  // The Hub's "Base" chip reads base_model from the card tags; a typed repo is looked up.
  const hubResult = hubResults.find((r) => r.id === repoId);
  const looked = cardInfo?.repoId === repoId ? cardInfo : null;
  const lookedArchitecture = looked?.architecture ?? null;

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
        architecture: r.ggufArchitecture,
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
        architecture: id === repoId ? lookedArchitecture : null,
        totalParams: q4nxParamsFromName(id),
      });
    }
    return rows;
  }, [hubResults, hubQuery, repoId, lookedArchitecture]);

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
    // Nothing is listed until the picker opens; a pick made in it stays listed.
    if (!open && !selectedId) return;
    const lane = `${accessToken ?? ""}\n`;
    for (const row of rows) {
      // Per token, so listings still in flight for a replaced token do not hold up the new one.
      const busy = [...inFlight.current].filter((f) => f.startsWith(lane));
      if (busy.length >= LISTING_WORKERS) return;
      const key = row.listingKey;
      // Local folders are listed up front so a closed row can show its memory need; Hub rows estimate it from the parameter count until opened or picked.
      if (
        !key ||
        (source === "hub" &&
          row.id !== repoId &&
          !q4nxIsExpanded(expansion, row.id)) ||
        (!open && row.id !== selectedId)
      ) {
        continue;
      }
      const flight = lane + key;
      if (listings[key] || inFlight.current.has(flight)) continue;
      inFlight.current.add(flight);
      const request = row.folder
        ? listGgufVariants(row.folder, accessToken, {
            preferLocalCache: true,
            localPath: row.folder,
          })
        : listGgufVariants(row.id, accessToken);
      const settle = (listing: Listing) => {
        inFlight.current.delete(flight);
        // An answer for a replaced token is dropped.
        if (liveToken.current !== accessToken) return;
        setListed((prev) => ({
          token: accessToken,
          byKey: {
            ...(prev.token === accessToken ? prev.byKey : NO_LISTINGS),
            [key]: listing,
          },
        }));
      };
      request
        .then((response) =>
          settle({ list: q4nxConvertibleFiles(response.variants), error: null }),
        )
        .catch((error: unknown) =>
          settle({
            list: [],
            error: error instanceof Error ? error.message : String(error),
          }),
        );
    }
  }, [open, source, rows, repoId, selectedId, expansion, listings, accessToken]);

  /** A model's files, the ones its converter packs directly first. */
  const filesOf = (row: SourceRow): GgufVariantDetail[] =>
    row.listingKey
      ? sortQ4nxVariants(
          listings[row.listingKey]?.list ?? [],
          row.architecture,
        )
      : [];

  const selectedRow = rows.find((row) => row.id === selectedId) ?? null;
  const selectedFiles = selectedRow ? filesOf(selectedRow) : [];
  const selectedVariant =
    selectedFiles.find((v) => v.filename === filename) ?? null;
  const localRow = source === "local" ? selectedRow : null;
  const localInfo = localRow ? localById.get(localRow.id) : undefined;
  const localPath = localRow ? (localInfo?.path ?? localRow.id) : null;

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

  const selectedArchitecture =
    source === "hub"
      ? (hubResult?.ggufArchitecture ?? lookedArchitecture)
      : null;
  const archFit =
    source === "hub" && repoId
      ? q4nxArchFit(
          selectedArchitecture,
          `${repoId} ${selectedVariant?.filename ?? ""}`,
        )
      : localRow
        ? q4nxArchFit(null, `${localRow.id} ${ggufPath ?? localPath}`)
        : null;
  const sourceReady =
    source === "hub" ? !!(repoId && selectedVariant) : !!ggufPath;
  const request = JSON.stringify([
    source === "hub" ? [repoId, filename] : ggufPath,
    effectiveBase.trim(),
    saveDirectory.trim(),
    accessToken ?? null,
  ]);
  const canConvert =
    !running &&
    sourceReady &&
    result?.refused !== request &&
    archFit?.status !== "unsupported" &&
    !!effectiveBase.trim() &&
    !!saveDirectory.trim();

  function pick(id: string, file: string) {
    if (id !== selectedId) setBaseModel(null);
    if (source === "hub") setRepoId(id);
    else setLocalId(id);
    setFilename(file);
    setResult(null);
  }

  /** A one-file row is its own variant, named by its file stem. */
  function ownVariant(row: SourceRow): GgufVariantDetail {
    const path = localById.get(row.id)?.path ?? row.id;
    return {
      filename: path,
      quant: fileName(path).replace(GGUF_FILE_RE, ""),
      size_bytes: row.sizeBytes ?? 0,
    };
  }

  function weightsOf(row: SourceRow, variant: GgufVariantDetail | null) {
    return q4nxWeights(
      row.listingKey ? variant : ownVariant(row),
      filesOf(row),
      row.totalParams,
      row.architecture,
    );
  }

  /** A model row's need is its picked file's, else its default file's. */
  function rowMemory(row: SourceRow): Q4nxMemory | null {
    const variant =
      (row.id === selectedId ? selectedVariant : null) ??
      filesOf(row)[0] ??
      null;
    return q4nxNpuMemory(weightsOf(row, variant), windowGb);
  }

  const selectedQuant =
    sourceReady && selectedRow
      ? (selectedVariant ?? ownVariant(selectedRow)).quant
      : null;
  const selectedPacking = selectedQuant
    ? q4nxPacking(selectedQuant, selectedArchitecture)
    : null;
  const selectedWeights =
    sourceReady && selectedRow ? weightsOf(selectedRow, selectedVariant) : null;
  const selectedMemory = q4nxNpuMemory(selectedWeights, windowGb);
  const memoryWarning = q4nxMemoryWarning(selectedMemory, windowGb);
  const selectedFile = sourceReady
    ? fileName(source === "hub" ? filename : (ggufPath ?? ""))
    : null;
  const selectedLabel = selectedRow?.label ?? "";

  // Serialized so the page hears about a change of content, not every render.
  const summaryJson = JSON.stringify({
    ready: canConvert,
    rows: q4nxSummaryRows({
      model: selectedLabel,
      file: selectedFile,
      quant: selectedQuant,
      packing: selectedPacking,
      baseModel: effectiveBase,
      saveDirectory,
      size: selectedWeights
        ? `${selectedWeights.estimated ? "~" : ""}${formatFootprintBytes(selectedWeights.bytes)}`
        : null,
      memory: selectedMemory,
      archFit,
    }),
  } satisfies Q4nxSummary);
  useEffect(() => {
    onSummary?.(JSON.parse(summaryJson));
  }, [onSummary, summaryJson]);

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
        // A server fault or dropped connection can be retried as is; a refusal needs a different file or setting.
        refused: q4nxIsRefusal(error) ? request : undefined,
      });
    } finally {
      setRunning(false);
    }
  }

  const loadingRows = source === "hub" ? searchingHub : loadingLocal;

  const packingTag = (quant: string, architecture?: string | null) => {
    const packing = q4nxPacking(quant, architecture);
    return packing === "direct" ? null : (
      <span className="ml-1.5 font-sans text-ui-9 font-medium text-muted-foreground">
        {packing === "requantized" ? "re-quantized" : "direct for Qwen3.5"}
      </span>
    );
  };

  const renderFiles = (row: SourceRow) => {
    const listing = row.listingKey ? listings[row.listingKey] : undefined;
    if (!listing) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
          <Spinner className="size-3.5" /> Loading files...
        </div>
      );
    }
    const files = filesOf(row);
    if (files.length === 0) {
      return (
        <p
          className={cn(
            "px-2 py-1 text-xs",
            listing.error ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {listing.error
            ? `Could not list the files: ${listing.error}`
            : "No single-file GGUF found"}
        </p>
      );
    }
    return files.map((v) => {
      const picked = row.id === selectedId && v.filename === filename;
      const memory = q4nxNpuMemory(weightsOf(row, v), windowGb);
      return (
        <div key={v.filename}>
          <button
            type="button"
            data-testid="q4nx-source-file"
            aria-pressed={picked}
            onClick={() => {
              pick(row.id, v.filename);
              setSettingsOpen(true);
            }}
            className={cn(
              "flex w-full min-w-0 items-center justify-between gap-2 rounded-full py-1 pl-2 pr-1.5 text-left text-sm transition-colors hover:bg-sidebar-accent focus-visible:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              picked && "bg-sidebar-accent",
            )}
          >
            <span className="min-w-0 flex-1 truncate font-mono text-xs">
              {fileName(v.filename)}
              {packingTag(v.quant, row.architecture)}
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

  // The chat picker's list page: search, section tabs, then the rows.
  const listPage = (
    <div className="relative space-y-2">
      <div className="flex items-center gap-2 pb-1 pr-2">
        <div className="relative flex-1">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={source === "hub" ? hubQuery : localQuery}
            onChange={(e) =>
              (source === "hub" ? setHubQuery : setLocalQuery)(e.target.value)
            }
            placeholder={
              source === "hub"
                ? "Search GGUF models or type owner/name"
                : "Search local models or type a .gguf path"
            }
            data-model-picker-search-input={true}
            className="field-soft h-(--picker-control-h) border-0 pl-8 pr-8 text-sm"
          />
          {loadingRows && (
            <Spinner className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          )}
        </div>
      </div>
      <div className="-mr-2 flex flex-wrap items-center gap-2">
        <PillTabs
          className="[&_[role=tab]]:px-[calc(0.75rem*var(--ui-space-scale,1)_+_var(--picker-tab-pad)/2)]"
          ariaLabel="GGUF source"
          tabs={SOURCE_TABS}
          value={source}
          onValueChange={(next) => {
            setSource(next as Source);
            setFilename("");
            setBaseModel(null);
          }}
          fit={true}
        />
      </div>
      <div
        data-q4nx-list=""
        data-testid="q4nx-source-list"
        className="model-list-scroll max-h-[calc(335px*var(--ui-space-scale,1))] overflow-y-auto scroll-py-1.5 pl-0.5"
      >
        <div className="model-list-gutter pb-4">
          <ListLabel>{source === "hub" ? "Hugging Face" : "On device"}</ListLabel>
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
                    memory?.fits === false ? "May not fit NPU memory" : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                  selected={selected}
                  quantChip={
                    selected && selectedQuant
                      ? q4nxQuantName(selectedQuant)
                      : null
                  }
                  vramStatus={null}
                  onClick={() => {
                    const key = row.listingKey;
                    if (!key) {
                      pick(row.id, "");
                      setSettingsOpen(true);
                      return;
                    }
                    const next = q4nxToggleExpanded(expansion, row.id);
                    setToggled({ key: expansionKey, ids: next.toggled });
                    // Reopening a model whose files failed to list asks again.
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
                    {renderFiles(row)}
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
    </div>
  );

  const notes = (archFit?.note ||
    memoryWarning ||
    (selectedQuant && selectedPacking !== "direct")) && (
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
      {selectedQuant && selectedPacking === "requantized" && (
        <p className="text-xs text-muted-foreground">
          {q4nxQuantName(selectedQuant)} is dequantized and rounded again to
          4-bit, so it loses some quality. Q4_0 and Q4_1 pack directly, and
          Q4_K_M does for Qwen3.5.
        </p>
      )}
      {selectedQuant && selectedPacking === "by-architecture" && (
        <p className="text-xs text-muted-foreground">
          Q4_K_M packs directly only for Qwen3.5 models. For any other model it
          is dequantized and rounded again to 4-bit.
        </p>
      )}
    </div>
  );

  // The popover's second page, laid out like the chat picker's run settings.
  const settingsPage = (
    <div className="min-h-0 w-full overflow-y-auto pt-4.5 pr-[calc(var(--spacing)*5+min(var(--overlay-scrollbar-gutter,0px),var(--spacing)*2))] pb-4.5 pl-7 [scrollbar-gutter:stable]">
      <div className="flex flex-col" data-testid="q4nx-convert-settings">
        <div className="flex items-center gap-2.5 pb-5">
          <button
            type="button"
            onClick={() => setSettingsOpen(false)}
            className="nav-icon-btn -ml-2.75 shrink-0 text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
            aria-label="Back to model list"
          >
            <HugeiconsIcon
              icon={ChevronLeftStandardIcon}
              className="size-4"
              strokeWidth={1.75}
            />
          </button>
          <div className="min-w-0 flex-1">
            <div className="text-ui-10 font-semibold uppercase leading-none tracking-wider text-muted-foreground">
              Convert settings
            </div>
            <div className="mt-1.5 truncate text-ui-14 font-semibold leading-tight text-foreground">
              {selectedLabel}
            </div>
            <div className="mt-1 truncate text-ui-11 leading-tight text-muted-foreground">
              {selectedMemory?.label ?? " "}
            </div>
          </div>
        </div>
        <div className="space-y-5">
          <div className="flex flex-col gap-2">
            <span className={FIELD_LABEL}>Source quant</span>
            {selectedRow?.listingKey ? (
              <Select value={filename} onValueChange={setFilename}>
                <SelectTrigger className="w-full" aria-label="Source quant">
                  <SelectValue placeholder="Select a file" />
                </SelectTrigger>
                <SelectContent>
                  {selectedFiles.map((v) => (
                    <SelectItem key={v.filename} value={v.filename}>
                      <span className="flex min-w-0 items-center">
                        <span className="truncate">{fileName(v.filename)}</span>
                        {packingTag(v.quant, selectedRow.architecture)}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <p className="truncate font-mono text-xs text-foreground/90">
                {selectedFile}
              </p>
            )}
            {selectedQuant && selectedPacking && (
              <p className="text-ui-11 text-muted-foreground">
                {q4nxQuantName(selectedQuant)},{" "}
                {Q4NX_PACKING_TEXT[selectedPacking]}
              </p>
            )}
            <NpuMemoryBar memory={selectedMemory} windowGb={windowGb} />
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
          {notes}
        </div>
        <div className="flex flex-wrap items-center gap-2 pt-5">
          <Button
            variant="dark"
            type="button"
            size="sm"
            disabled={!canConvert}
            onClick={() => {
              setOpen(false);
              setSettingsOpen(false);
              void handleConvert();
            }}
          >
            Convert
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="text-muted-foreground"
            disabled={
              baseModel === null && saveDirectory === DEFAULT_SAVE_DIRECTORY
            }
            onClick={() => {
              setBaseModel(null);
              setSaveDirectory(DEFAULT_SAVE_DIRECTORY);
            }}
          >
            Reset all
          </Button>
        </div>
      </div>
    </div>
  );

  const showSettings = settingsOpen && !!selectedRow;

  return (
    <SectionBox
      dataTour={Q4NX_CARD_MARK}
      icon={CpuIcon}
      chipTint="var(--chart-1)"
      title="Convert a GGUF to Q4NX (AMD NPU)"
      description="Convert a Q4_0, Q4_1 or Q4_K_M GGUF to Q4NX. Only supported on AMD NPUs."
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className={FIELD_LABEL}>GGUF model</span>
          <div className="flex min-w-0 items-center gap-2">
            <Popover
              open={open}
              onOpenChange={(next) => {
                setOpen(next);
                if (!next) setSettingsOpen(false);
              }}
            >
              <ModelSelector.Trigger
                currentModel={
                  selectedRow
                    ? {
                        id: selectedRow.id,
                        name: selectedRow.label,
                        description: selectedQuant
                          ? q4nxQuantName(selectedQuant)
                          : "Pick a GGUF file",
                        descriptionSuffix: selectedMemory?.label,
                      }
                    : undefined
                }
                isLoaded={canConvert}
                placeholder="Select GGUF model"
                className="min-w-0 max-w-full"
              />
              <PopoverContent
                align="start"
                alignOffset={10}
                dropdownGlow="picker"
                data-tour={Q4NX_CARD_MARK}
                aria-label={
                  showSettings
                    ? `Convert settings for ${selectedLabel}`
                    : undefined
                }
                className={cn(
                  "unsloth-model-selector-menu menu-soft-surface ring-0 w-[min(var(--picker-panel-w),calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0",
                  showSettings
                    ? "max-h-[var(--radix-popover-content-available-height)] overflow-hidden p-0 !px-0"
                    : "pt-4 pb-0 pl-4 pr-2",
                )}
              >
                <TooltipProvider
                  delayDuration={0}
                  skipDelayDuration={0}
                  disableHoverableContent={true}
                >
                  {showSettings ? settingsPage : listPage}
                </TooltipProvider>
              </PopoverContent>
            </Popover>
            {sourceReady && (
              <ModelLoadSettingsAction
                ariaLabel="Convert settings"
                tooltip="Original model and save directory"
                onConfigure={() => {
                  setSettingsOpen(true);
                  setOpen(true);
                }}
              />
            )}
          </div>
        </div>
        {notes}
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
                ? `Converts ${selectedFile} with ${effectiveBase.trim() || "no original model"} into ${saveDirectory.trim() || "no save directory"}. No training or GPU needed.`
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
