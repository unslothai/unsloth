// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionBox } from "@/components/section-box";
import { SegmentedControl } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { InputGroupAddon } from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  cachedModelInfo,
  detectBaseModel,
  type GgufVariantDetail,
  type LocalModelInfo,
  hfApiToken,
  listGgufVariants,
  useHfTokenStore,
  useHubModelSearch,
} from "@/features/hub";
import { useDebouncedValue } from "@/hooks";
import { FolderSearchIcon } from "@/lib/folder-icons";
import { CpuIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { convertGgufToQ4nx } from "../api/export-api";
import {
  getCachedLocalModels,
  refreshLocalModels,
} from "../export-navigation-cache";
import {
  localGgufPath,
  packsDirectly,
  q4nxArchFit,
  q4nxBaseModel,
  sortQ4nxVariants,
} from "../lib/q4nx-source";

type Source = "hub" | "local";
const SOURCE_OPTIONS = [
  { value: "hub", label: "Hugging Face" },
  { value: "local", label: "Local file" },
] as const satisfies readonly { value: Source; label: string }[];

const FIELD_LABEL =
  "flex items-center gap-1.5 text-xs font-medium text-muted-foreground";
const SEARCH_INPUT_REASONS = new Set([
  "input-change",
  "input-paste",
  "input-clear",
]);

function isGgufRow(row: LocalModelInfo): boolean {
  return (
    row.model_format === "gguf" || row.path.toLowerCase().endsWith(".gguf")
  );
}

function SourceCombobox({
  label,
  items,
  value,
  onSelect,
  onTyped,
  placeholder,
  icon,
  loading,
  loadingText,
  emptyText,
  itemLabel,
  itemHint,
  accept,
}: {
  label: string;
  items: string[];
  value: string;
  onSelect: (id: string) => void;
  onTyped: (value: string) => void;
  placeholder: string;
  icon: typeof Search01Icon;
  loading: boolean;
  loadingText: string;
  emptyText: string;
  itemLabel?: (id: string) => string;
  itemHint?: (id: string) => string | null;
  /** Whether typed text that matches no row can be used as is. */
  accept: (value: string) => boolean;
}) {
  const anchorRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef(value);
  const selectingRef = useRef(false);
  const commitTyped = () => {
    const typed = inputRef.current.trim();
    if (typed && (items.includes(typed) || accept(typed))) onSelect(typed);
  };
  return (
    <div className="flex flex-col gap-2">
      <label className={FIELD_LABEL}>{label}</label>
      <div ref={anchorRef}>
        <Combobox
          items={items}
          filteredItems={items}
          filter={null}
          value={value || null}
          onValueChange={(id: string | null) => {
            selectingRef.current = true;
            inputRef.current = id ?? "";
            onSelect(id ?? "");
          }}
          onInputValueChange={(next: string, details?: { reason?: string }) => {
            inputRef.current = next;
            if (selectingRef.current) {
              selectingRef.current = false;
              return;
            }
            if (SEARCH_INPUT_REASONS.has(details?.reason ?? "")) onTyped(next);
          }}
          itemToStringValue={(id: string) => id}
          autoHighlight={true}
        >
          <ComboboxInput
            placeholder={placeholder}
            className="w-full"
            onBlur={() => commitTyped()}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              event.preventDefault();
              commitTyped();
            }}
          >
            <InputGroupAddon>
              <HugeiconsIcon icon={icon} className="size-4" />
            </InputGroupAddon>
          </ComboboxInput>
          <ComboboxContent anchor={anchorRef}>
            {loading ? (
              <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
                <Spinner className="size-4" /> {loadingText}
              </div>
            ) : (
              <ComboboxEmpty>{emptyText}</ComboboxEmpty>
            )}
            <ComboboxList>
              {(id: string) => {
                const hint = itemHint?.(id);
                return (
                  <ComboboxItem key={id} value={id} className="gap-2">
                    <span className="block min-w-0 flex-1 truncate">
                      {itemLabel?.(id) ?? id}
                    </span>
                    {hint && (
                      <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                        {hint}
                      </span>
                    )}
                  </ComboboxItem>
                );
              }}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
      </div>
    </div>
  );
}

export function Q4nxConvertCard() {
  const hfToken = useHfTokenStore((s) => s.token);
  const accessToken = hfApiToken(hfToken);
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
  const [variants, setVariants] = useState<{
    key: string;
    list: GgufVariantDetail[];
    error: string | null;
  } | null>(null);
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
  const hubIds = useMemo(() => {
    const ids = hubResults
      .filter((r) => q4nxArchFit(r.ggufArchitecture).status !== "unsupported")
      .map((r) => r.id);
    if (repoId && !ids.includes(repoId)) ids.unshift(repoId);
    return ids;
  }, [hubResults, repoId]);

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

  const localRows = useMemo(() => localModels.filter(isGgufRow), [localModels]);
  const localById = useMemo(
    () => new Map(localRows.map((row) => [row.id, row])),
    [localRows],
  );
  const localIds = useMemo(() => {
    const q = localQuery.trim().toLowerCase();
    const ids = localRows
      .filter(
        (row) =>
          !q ||
          row.id.toLowerCase().includes(q) ||
          row.display_name.toLowerCase().includes(q) ||
          row.path.toLowerCase().includes(q),
      )
      .map((row) => row.id);
    if (localId && !ids.includes(localId)) ids.unshift(localId);
    return ids;
  }, [localRows, localQuery, localId]);

  // A typed path that is not a known row is used as the file itself.
  const localRow: LocalModelInfo | null = localId
    ? (localById.get(localId) ?? {
        id: localId,
        display_name: localId,
        path: localId,
        source: "custom",
      })
    : null;
  const localSingleFile = !!localRow && /\.gguf$/i.test(localRow.path.trim());

  const localDirPath = localRow && !localSingleFile ? localRow.path : null;
  const variantKey =
    source === "hub"
      ? repoId
        ? `hub::${repoId}`
        : null
      : localDirPath
        ? `local::${localDirPath}`
        : null;
  useEffect(() => {
    if (!variantKey) return;
    const controller = new AbortController();
    const request = localDirPath
      ? listGgufVariants(localDirPath, accessToken, {
          preferLocalCache: true,
          localPath: localDirPath,
          signal: controller.signal,
        })
      : listGgufVariants(repoId, accessToken, { signal: controller.signal });
    request
      .then((response) => {
        const list = sortQ4nxVariants(response.variants);
        setVariants({ key: variantKey, list, error: null });
        setFilename((current) => current || list[0]?.filename || "");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setVariants({
          key: variantKey,
          list: [],
          error: error instanceof Error ? error.message : String(error),
        });
      });
    return () => controller.abort();
  }, [variantKey, localDirPath, repoId, accessToken]);
  const variantState = variants?.key === variantKey ? variants : null;
  const variantList = variantState?.list ?? [];

  // The Hub's "Base" chip reads base_model from the card tags; a typed repo is looked up.
  const hubResult = hubResults.find((r) => r.id === repoId);
  useEffect(() => {
    if (source !== "hub" || !repoId || hubResult) return;
    let cancelled = false;
    void cachedModelInfo({ name: repoId, accessToken })
      .then((info) => {
        if (cancelled) return;
        setCardInfo({
          repoId,
          baseModel: detectBaseModel(info.tags),
          architecture:
            (info as { gguf?: { architecture?: string } }).gguf?.architecture ??
            null,
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
      : localRow
        ? q4nxBaseModel(
            localRow.model_id ??
              (localRow.source === "hf_cache" ? localRow.id : ""),
            localRow.base_model ?? detectBaseModel(localRow.tags),
          )
        : "";
  const effectiveBase = baseModel ?? autoBase;

  const archFit =
    source === "hub" && repoId
      ? q4nxArchFit(hubResult?.ggufArchitecture ?? looked?.architecture)
      : localRow
        ? q4nxArchFit(null)
        : null;

  const selectedVariant =
    variantList.find((v) => v.filename === filename) ?? null;
  const ggufPath =
    source === "local" && localRow
      ? localGgufPath(localRow.path, localSingleFile ? null : selectedVariant)
      : null;
  const sourceReady =
    source === "hub" ? !!(repoId && selectedVariant) : !!ggufPath;
  const canConvert =
    !running &&
    sourceReady &&
    archFit?.status !== "unsupported" &&
    !!effectiveBase.trim() &&
    !!saveDirectory.trim();

  function pickRepo(id: string) {
    if (id === repoId) return;
    setRepoId(id);
    setHubQuery(id);
    setFilename("");
    setBaseModel(null);
  }

  function pickLocal(id: string) {
    if (id === localId) return;
    setLocalId(id);
    setLocalQuery("");
    setFilename("");
    setBaseModel(null);
  }

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

  const fileDisabled =
    source === "hub" ? !repoId : !localRow || localSingleFile;
  const filePlaceholder = fileDisabled
    ? localSingleFile
      ? (localRow?.path.split(/[\\/]/).pop() ?? "")
      : "Pick a model first"
    : variantState
      ? variantState.error
        ? "Could not list the files"
        : variantList.length === 0
          ? "No GGUF files found"
          : "Select a file"
      : "Loading files...";

  return (
    <SectionBox
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
          {source === "hub" ? (
            <SourceCombobox
              key="hub"
              label="GGUF model"
              items={hubIds}
              value={repoId}
              onSelect={pickRepo}
              onTyped={(value) => {
                setHubQuery(value);
                if (!value.trim()) pickRepo("");
              }}
              placeholder="Search GGUF models..."
              icon={Search01Icon}
              loading={searchingHub}
              loadingText="Searching..."
              emptyText="No GGUF models found"
              accept={(value) => /^[^/\s]+\/[^/\s]+$/.test(value)}
            />
          ) : (
            <SourceCombobox
              key="local"
              label="GGUF model"
              items={localIds}
              value={localId}
              onSelect={pickLocal}
              onTyped={(value) => {
                setLocalQuery(value);
                if (!value.trim()) pickLocal("");
              }}
              placeholder={
                loadingLocal
                  ? "Scanning local and cached models..."
                  : "/path/to/model.Q4_1.gguf"
              }
              icon={FolderSearchIcon}
              loading={loadingLocal}
              loadingText="Scanning..."
              emptyText="No local GGUF models found"
              accept={(value) => /\.gguf$/i.test(value)}
              itemLabel={(id) => localById.get(id)?.display_name ?? id}
              itemHint={(id) => {
                const row = localById.get(id);
                if (!row) return null;
                return row.source === "hf_cache"
                  ? "HF cache"
                  : row.source === "custom"
                    ? "Custom Folders"
                    : "Local dir";
              }}
            />
          )}
          <div className="flex flex-col gap-2">
            <label className={FIELD_LABEL}>GGUF file</label>
            <Select
              value={localSingleFile ? "" : filename}
              onValueChange={setFilename}
              disabled={fileDisabled || variantList.length === 0}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder={filePlaceholder} />
              </SelectTrigger>
              <SelectContent>
                {variantList.map((v) => (
                  <SelectItem key={v.filename} value={v.filename}>
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="truncate">
                        {v.filename.split("/").pop()}
                      </span>
                      {!packsDirectly(v.quant) && (
                        <span className="shrink-0 text-ui-10 text-muted-foreground">
                          re-quantized
                        </span>
                      )}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
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
            {selectedVariant && !packsDirectly(selectedVariant.quant) && (
              <p className="text-xs text-muted-foreground">
                {selectedVariant.quant} is dequantized and rounded again to
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
              variantState?.error ??
              "No training or GPU needed."}
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
