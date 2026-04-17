// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  useDebouncedValue,
  useHfDatasetSearch,
  useHfTokenValidation,
  useInfiniteScroll,
} from "@/hooks";
import {
  HfDatasetSubsetSplitSelectors,
  uploadTrainingDataset,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
} from "@/features/training";
import { listLocalDatasets } from "@/features/training/api/datasets-api";
import type { LocalDatasetInfo } from "@/features/training/types/datasets";
import { useNavigate } from "@tanstack/react-router";
import {
  ArrowDown01Icon,
  Cancel01Icon,
  CloudUploadIcon,
  Database02Icon,
  FileAttachmentIcon,
  InformationCircleIcon,
  Search01Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useShallow } from "zustand/react/shallow";
import { DocumentUploadRedirectDialog } from "./document-upload-redirect-dialog";

const DOCUMENT_REDIRECT_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);

const SEARCH_INPUT_REASONS = new Set(["input-change", "input-paste", "input-clear"]);
const OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY =
  "data-recipes:open-learning-recipes";

function isLikelyLocalDatasetRef(value: string) {
  return (
    value.startsWith("/") ||
    value.startsWith("./") ||
    value.startsWith("../") ||
    value.includes("\\") ||
    /\.(jsonl|json|csv|parquet)$/i.test(value)
  );
}

function deriveLocalDatasetName(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  const parts = normalized.split("/").filter(Boolean);
  const parquetIndex = parts.lastIndexOf("parquet-files");
  if (parquetIndex > 0) return parts[parquetIndex - 1];
  const basename = parts[parts.length - 1] ?? path;
  // Strip UUID prefix from uploaded files (format: {32hex}_{original})
  const uuidPrefixMatch = basename.match(/^[a-f0-9]{32}_(.+)$/);
  if (uuidPrefixMatch) return uuidPrefixMatch[1];
  return basename;
}

function formatUpdatedDate(timestamp: number | null): string {
  if (typeof timestamp !== "number") return "--";
  return new Date(timestamp * 1000).toLocaleDateString();
}

function normalizeSliceInput(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!/^\d+$/.test(trimmed)) return null;
  return trimmed;
}

export function DatasetSection() {
  const navigate = useNavigate();
  const {
    dataset,
    datasetSource,
    selectHfDataset,
    selectLocalDataset,
    datasetFormat,
    setDatasetFormat,
    datasetSubset,
    setDatasetSubset,
    datasetSplit,
    setDatasetSplit,
    datasetEvalSplit,
    setDatasetEvalSplit,
    uploadedFile,
    uploadedEvalFile,
    setUploadedEvalFile,
    hfToken,
    modelType,
    datasetSliceStart,
    setDatasetSliceStart,
    datasetSliceEnd,
    setDatasetSliceEnd,
  } = useTrainingConfigStore(
    useShallow((s) => ({
      dataset: s.dataset,
      datasetSource: s.datasetSource,
      selectHfDataset: s.selectHfDataset,
      selectLocalDataset: s.selectLocalDataset,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      datasetSubset: s.datasetSubset,
      setDatasetSubset: s.setDatasetSubset,
      datasetSplit: s.datasetSplit,
      setDatasetSplit: s.setDatasetSplit,
      datasetEvalSplit: s.datasetEvalSplit,
      setDatasetEvalSplit: s.setDatasetEvalSplit,
      uploadedFile: s.uploadedFile,
      uploadedEvalFile: s.uploadedEvalFile,
      setUploadedEvalFile: s.setUploadedEvalFile,
      hfToken: s.hfToken,
      modelType: s.modelType,
      datasetSliceStart: s.datasetSliceStart,
      setDatasetSliceStart: s.setDatasetSliceStart,
      datasetSliceEnd: s.datasetSliceEnd,
      setDatasetSliceEnd: s.setDatasetSliceEnd,
    })),
  );

  const [searchQuery, setSearchQuery] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [pickerTab, setPickerTab] = useState<"huggingface" | "local">(
    datasetSource === "upload" ? "local" : "huggingface",
  );
  const [localDatasets, setLocalDatasets] = useState<LocalDatasetInfo[]>([]);
  const [hasLoadedLocalDatasets, setHasLoadedLocalDatasets] = useState(false);
  const [localLoading, setLocalLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const openPreview = useDatasetPreviewDialogStore((s) => s.openPreview);
  const selectingRef = useRef(false);
  const pendingSourceTabRef = useRef<"huggingface" | "local" | null>(null);
  const debouncedQuery = useDebouncedValue(searchQuery);

  useEffect(() => {
    setPickerTab(datasetSource === "upload" ? "local" : "huggingface");
  }, [datasetSource]);

  const refreshLocalDatasets = useCallback(async () => {
    setLocalLoading(true);
    setLocalError(null);
    try {
      const response = await listLocalDatasets();
      setLocalDatasets(response.datasets ?? []);
    } catch (error) {
      setLocalError(
        error instanceof Error ? error.message : "Failed to load local datasets.",
      );
    } finally {
      setHasLoadedLocalDatasets(true);
      setLocalLoading(false);
    }
  }, []);

  useEffect(() => {
    if (pickerTab !== "local") return;
    void refreshLocalDatasets();
  }, [pickerTab, refreshLocalDatasets]);

  useEffect(() => {
    const handleRefresh = () => {
      if (document.hidden) return;
      if (pickerTab !== "local" && datasetSource !== "upload") return;
      void refreshLocalDatasets();
    };

    window.addEventListener("focus", handleRefresh);
    document.addEventListener("visibilitychange", handleRefresh);
    return () => {
      window.removeEventListener("focus", handleRefresh);
      document.removeEventListener("visibilitychange", handleRefresh);
    };
  }, [datasetSource, pickerTab, refreshLocalDatasets]);

  function handleDatasetSelect(id: string | null) {
    selectingRef.current = true;
    pendingSourceTabRef.current = "huggingface";
    selectHfDataset(id);
  }

  function handleLocalDatasetSelect(path: string) {
    selectingRef.current = true;
    pendingSourceTabRef.current = "local";
    selectLocalDataset(path);
  }

  function clearSelectionForTab(tab: "huggingface" | "local") {
    pendingSourceTabRef.current = tab;
    if (tab === "huggingface") {
      handleDatasetSelect(null);
      return;
    }
    selectingRef.current = true;
    selectLocalDataset(null);
  }

  function handleInputChange(
    val: string,
    eventDetails?: {
      reason?: string;
    },
  ) {
    if (selectingRef.current) {
      selectingRef.current = false;
      return;
    }
    if (!SEARCH_INPUT_REASONS.has(eventDetails?.reason ?? "")) {
      return;
    }
    setSearchQuery(val);
  }

  const effectiveModelType = modelType ?? "text";

  const {
    results: hfResults,
    isLoading,
    isLoadingMore,
    fetchMore,
    error: hfSearchError,
  } = useHfDatasetSearch(pickerTab === "huggingface" ? debouncedQuery : "", {
    modelType: effectiveModelType,
    accessToken: hfToken || undefined,
    enabled: pickerTab === "huggingface",
  });

  const { error: tokenValidationError, isChecking: isCheckingToken } =
    useHfTokenValidation(hfToken);

  const hfResultIds = useMemo(() => {
    const ids = hfResults.map((r) => r.id);
    if (dataset && !ids.includes(dataset)) {
      ids.push(dataset);
    }
    return ids;
  }, [hfResults, dataset]);

  const localFilteredDatasets = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return localDatasets;
    return localDatasets.filter(
      (item) =>
        item.label.toLowerCase().includes(query) ||
        item.path.toLowerCase().includes(query),
    );
  }, [localDatasets, searchQuery]);

  const localPathById = useMemo(() => {
    return new Map(localDatasets.map((item) => [item.id, item.path]));
  }, [localDatasets]);

  const localLabelById = useMemo(() => {
    return new Map(localDatasets.map((item) => [item.id, item.label]));
  }, [localDatasets]);

  const selectedLocalDataset = useMemo(() => {
    if (!uploadedFile) return null;
    return localDatasets.find((item) => item.path === uploadedFile) ?? null;
  }, [localDatasets, uploadedFile]);

  const selectedLocalId = selectedLocalDataset?.id ?? null;

  const localResultIds = useMemo(() => {
    const ids = localFilteredDatasets.map((item) => item.id);
    if (selectedLocalDataset && selectedLocalId && !ids.includes(selectedLocalId)) {
      ids.push(selectedLocalId);
    }
    return ids;
  }, [localFilteredDatasets, selectedLocalDataset, selectedLocalId]);

  useEffect(() => {
    if (!hasLoadedLocalDatasets) return;
    if (localLoading) return;
    if (localError) return;
    if (datasetSource !== "upload") return;
    if (!uploadedFile) return;
    if (selectedLocalDataset) return;
    // Don't clear if this is a direct file upload (e.g. user uploaded a .jsonl/.csv)
    if (/\.(jsonl|json|csv|parquet|arrow)$/i.test(uploadedFile)) return;
    selectLocalDataset(null);
  }, [
    datasetSource,
    hasLoadedLocalDatasets,
    localError,
    localLoading,
    uploadedFile,
    selectedLocalDataset,
    selectLocalDataset,
  ]);

  const activeSourceTab = datasetSource === "upload" ? "local" : "huggingface";
  const comboboxItems = pickerTab === "huggingface" ? hfResultIds : localResultIds;
  const comboboxValue =
    pickerTab === "huggingface"
      ? datasetSource === "huggingface"
        ? dataset
        : null
      : datasetSource === "upload"
        ? selectedLocalId
        : null;
  const isHfDatasetSelected =
    datasetSource === "huggingface" &&
    !!dataset &&
    !isLikelyLocalDatasetRef(dataset);

  const selectedDatasetName = datasetSource === "upload" ? uploadedFile : dataset;
  const selectedLocalMetadata = selectedLocalDataset?.metadata ?? null;
  const selectedLocalColumns = selectedLocalMetadata?.columns ?? [];
  const selectedLocalRows =
    selectedLocalDataset?.rows ?? selectedLocalMetadata?.actual_num_records ?? null;
  const selectedLocalUpdatedAt = selectedLocalDataset?.updated_at ?? null;

  const comboboxAnchorRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const evalFileInputRef = useRef<HTMLInputElement>(null);
  const { scrollRef, sentinelRef } = useInfiniteScroll(
    fetchMore,
    hfResults.length,
  );

  const [isUploading, setIsUploading] = useState(false);
  const [documentRedirectOpen, setDocumentRedirectOpen] = useState(false);
  const [redirectFileName, setRedirectFileName] = useState<string | null>(null);

  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileUpload = async (
    file: File,
    onSuccess: (storedPath: string) => void,
    successMessage: string,
  ) => {
    setIsUploading(true);
    try {
      const uploaded = await uploadTrainingDataset(file);
      onSuccess(uploaded.stored_path);
      toast.success(successMessage, { description: uploaded.filename });
    } catch (error) {
      toast.error("Upload failed", {
        description: error instanceof Error ? error.message : "Unknown error",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDatasetFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (DOCUMENT_REDIRECT_EXTENSIONS.has(extension)) {
      setRedirectFileName(file.name);
      setDocumentRedirectOpen(true);
      return;
    }

    await handleFileUpload(file, selectLocalDataset, "Dataset uploaded");
  };

  const handleEvalFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    await handleFileUpload(file, setUploadedEvalFile, "Eval dataset uploaded");
  };

  const handleOpenLearningRecipes = useCallback(() => {
    sessionStorage.setItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY, "1");
    setDocumentRedirectOpen(false);
    void navigate({ to: "/data-recipes" });
  }, [navigate]);

  return (
    <div data-tour="studio-dataset" className="min-w-0">
      <SectionCard
        icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
        title="Dataset"
        description="Select or upload training data"
        accent="indigo"
        className={`dark:shadow-border ${
          advancedOpen || (datasetSource === "upload" && uploadedFile)
            ? "min-h-studio-config-column"
            : "h-studio-config-column"
        }`}
      >
        <div className="flex min-w-0 flex-col gap-4">
          <div className="flex min-w-0 flex-col gap-2">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              Choose dataset
              <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[10px] font-medium text-foreground/80">
                {datasetSource === "upload" ? "Local" : "Hugging Face"}
              </span>
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    className="text-foreground/70 hover:text-foreground"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-3"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Use the popup tabs to switch between Hugging Face and local
                  recipe outputs.{" "}
                  <a
                    href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline"
                  >
                    Read more
                  </a>
                </TooltipContent>
              </Tooltip>
            </span>
            <div
              ref={comboboxAnchorRef}
              className="min-w-0"
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                if (!(event.target instanceof HTMLInputElement)) return;
                event.preventDefault();
                if (pickerTab === "huggingface") {
                  if (hfResults.length > 0) {
                    handleDatasetSelect(hfResults[0].id);
                  } else {
                    const text = event.target.value.trim();
                    if (text) handleDatasetSelect(text);
                  }
                  return;
                }

                if (localResultIds.length > 0) {
                  const selectedId = localResultIds[0];
                  const path = localPathById.get(selectedId);
                  if (path) {
                    handleLocalDatasetSelect(path);
                  }
                }
              }}
            >
              <Combobox
                items={comboboxItems}
                filteredItems={comboboxItems}
                filter={null}
                value={comboboxValue}
                onOpenChange={(open) => {
                  setSearchQuery("");
                  if (open && (pickerTab === "local" || activeSourceTab === "local")) {
                    void refreshLocalDatasets();
                  }
                  if (!open) {
                    setPickerTab(pendingSourceTabRef.current ?? activeSourceTab);
                    pendingSourceTabRef.current = null;
                  }
                }}
                onValueChange={(value) => {
                  if (!value) {
                    clearSelectionForTab(pickerTab);
                    return;
                  }
                  if (pickerTab === "huggingface") {
                    handleDatasetSelect(value);
                    return;
                  }
                  const path = localPathById.get(value);
                  if (path) {
                    handleLocalDatasetSelect(path);
                  }
                }}
                onInputValueChange={(value, eventDetails) =>
                  handleInputChange(value, eventDetails)
                }
                itemToStringValue={(id) =>
                  pickerTab === "local"
                    ? localLabelById.get(id) ?? id
                    : id
                }
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder={
                    pickerTab === "huggingface"
                      ? "Search Hugging Face datasets..."
                      : "Search local datasets..."
                  }
                  className="w-full min-w-0 overflow-hidden leading-5"
                  showClear={true}
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  <div className="px-2 pt-2 pb-2">
                    <Tabs
                      value={pickerTab}
                      onValueChange={(value) => {
                        setPickerTab(value as "huggingface" | "local");
                        setSearchQuery("");
                      }}
                      className="w-full"
                    >
                      <TabsList className=" w-full">
                        <TabsTrigger value="huggingface">Hugging Face</TabsTrigger>
                        <TabsTrigger value="local">Local</TabsTrigger>
                      </TabsList>

                      <TabsContent value="huggingface" className="m-0">
                        {isLoading ? (
                          <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
                            <Spinner className="size-4" /> Searching...
                          </div>
                        ) : (
                          <ComboboxEmpty>No datasets found</ComboboxEmpty>
                        )}
                        <div
                          ref={scrollRef}
                          className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]"
                        >
                          <ComboboxList className="p-1 !max-h-none !overflow-visible">
                            {(id: string) => {
                              return (
                                <ComboboxItem key={id} value={id} className="gap-2">
                                  <Tooltip>
                                    <TooltipTrigger asChild={true}>
                                      <span className="block min-w-0 flex-1 truncate">
                                        {id}
                                      </span>
                                    </TooltipTrigger>
                                    <TooltipContent
                                      side="left"
                                      className="max-w-xs break-all"
                                    >
                                      {id}
                                    </TooltipContent>
                                  </Tooltip>
                                </ComboboxItem>
                              );
                            }}
                          </ComboboxList>
                          <div ref={sentinelRef} className="h-px" />
                          {isLoadingMore && (
                            <div className="flex items-center justify-center py-2">
                              <Spinner className="size-3.5 text-muted-foreground" />
                            </div>
                          )}
                        </div>
                      </TabsContent>

                      <TabsContent value="local" className="m-0">
                        {localLoading ? (
                          <div className="flex items-center justify-center py-4 gap-2 text-xs text-muted-foreground">
                            <Spinner className="size-4" /> Loading local datasets...
                          </div>
                        ) : (
                          <>
                            {localError ? (
                              <p className="px-2 py-2 text-xs text-destructive">{localError}</p>
                            ) : (
                              <ComboboxEmpty className="px-2 py-3">
                                <div className="flex w-full flex-col items-center gap-2 text-center">
                                  <p className="text-xs text-muted-foreground">
                                    {localDatasets.length === 0
                                      ? "No local datasets yet."
                                      : "No local datasets match search."}
                                  </p>
                                  {localDatasets.length === 0 ? (
                                    <Button asChild={true} size="sm" variant="outline">
                                      <a href="/data-recipes">Open Data Recipes</a>
                                    </Button>
                                  ) : null}
                                </div>
                              </ComboboxEmpty>
                            )}
                            <div className="max-h-64 overflow-y-auto overscroll-contain [scrollbar-width:thin]">
                              <ComboboxList className="p-1 !max-h-none !overflow-visible">
                                {(id: string) => {
                                  const label = localLabelById.get(id) ?? id;
                                  return (
                                    <ComboboxItem key={id} value={id} className="gap-2">
                                      <Tooltip>
                                        <TooltipTrigger asChild={true}>
                                          <span className="block min-w-0 flex-1 truncate">
                                            {label}
                                          </span>
                                        </TooltipTrigger>
                                        <TooltipContent
                                          side="left"
                                          className="max-w-xs break-all"
                                        >
                                          {label}
                                        </TooltipContent>
                                      </Tooltip>
                                    </ComboboxItem>
                                  );
                                }}
                              </ComboboxList>
                            </div>
                          </>
                        )}
                      </TabsContent>
                    </Tabs>
                  </div>
                </ComboboxContent>
              </Combobox>
            </div>
            {(tokenValidationError ?? hfSearchError) && (
              <p className="text-xs text-destructive">
                {tokenValidationError ?? hfSearchError}
                {" — "}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  Get or update token
                </a>
              </p>
            )}
            {isCheckingToken && (
              <p className="text-xs text-muted-foreground">Checking token…</p>
            )}
            {pickerTab !== activeSourceTab && (
              <p className="text-[11px] text-muted-foreground">
                Browsing {pickerTab === "local" ? "Local datasets" : "Hugging Face"}.
                Current selection stays {datasetSource === "upload" ? "Local" : "Hugging Face"}.
              </p>
            )}
          </div>

          {isHfDatasetSelected ? (
            <HfDatasetSubsetSplitSelectors
              variant="studio"
              enabled={true}
              datasetName={dataset}
              accessToken={hfToken || undefined}
              datasetSubset={datasetSubset}
              setDatasetSubset={setDatasetSubset}
              datasetSplit={datasetSplit}
              setDatasetSplit={setDatasetSplit}
              datasetEvalSplit={datasetEvalSplit}
              setDatasetEvalSplit={setDatasetEvalSplit}
            />
          ) : !selectedDatasetName ? (
            <HfDatasetSubsetSplitSelectors
              variant="studio"
              enabled={false}
              datasetName={null}
              accessToken={hfToken || undefined}
              datasetSubset={datasetSubset}
              setDatasetSubset={setDatasetSubset}
              datasetSplit={datasetSplit}
              setDatasetSplit={setDatasetSplit}
              datasetEvalSplit={datasetEvalSplit}
              setDatasetEvalSplit={setDatasetEvalSplit}
            />
          ) : datasetSource === "upload" && selectedLocalDataset ? (
            <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">
                    Local dataset metadata
                  </p>
                  <p className="text-[10px] text-muted-foreground/80">
                    Data Recipe output.
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-3">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <MetadataRow
                    label="Rows"
                    value={
                      typeof selectedLocalRows === "number"
                        ? selectedLocalRows.toLocaleString()
                        : "--"
                    }
                  />
                  <MetadataRow
                    label="Columns"
                    value={
                      selectedLocalColumns.length > 0
                        ? String(selectedLocalColumns.length)
                        : "--"
                    }
                  />
                  <MetadataRow
                    label="Batches"
                    value={
                      typeof selectedLocalMetadata?.num_completed_batches === "number" &&
                      typeof selectedLocalMetadata?.total_num_batches === "number"
                        ? `${selectedLocalMetadata.num_completed_batches}/${selectedLocalMetadata.total_num_batches}`
                        : "--"
                    }
                  />
                  <MetadataRow
                    label="Updated"
                    value={formatUpdatedDate(selectedLocalUpdatedAt)}
                  />
                </div>
              </div>
            </div>
          ) : null}

          {datasetSource === "upload" && uploadedFile && (
            <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
              <p className="mb-2 text-xs font-medium text-muted-foreground">
                Eval dataset
              </p>
              {uploadedEvalFile ? (
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 overflow-hidden">
                    <HugeiconsIcon icon={FileAttachmentIcon} className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate text-xs">
                      {deriveLocalDatasetName(uploadedEvalFile)}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 shrink-0 cursor-pointer p-0"
                    onClick={() => setUploadedEvalFile(null)}
                  >
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full cursor-pointer gap-1.5"
                    disabled={isUploading}
                    onClick={() => evalFileInputRef.current?.click()}
                  >
                    {isUploading ? (
                      <Spinner className="size-3.5" />
                    ) : (
                      <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
                    )}
                    {isUploading ? "Uploading..." : "Upload eval file"}
                  </Button>
                  <p className="text-[10px] text-muted-foreground/80">
                    Optional. If not provided, a small portion will be split from the training data.
                  </p>
                </div>
              )}
            </div>
          )}

          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                className={`size-3.5 transition-transform ${advancedOpen ? "rotate-180" : ""}`}
              />
              Advanced
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 data-[state=open]:overflow-visible">
              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-2">
                  <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                    Target Format
                    <Tooltip>
                      <TooltipTrigger asChild={true}>
                        <button
                          type="button"
                          className="text-foreground/70 hover:text-foreground"
                        >
                          <HugeiconsIcon
                            icon={InformationCircleIcon}
                            className="size-3"
                          />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent>
                        Format of your training data. Auto-detect works for most
                        datasets.{" "}
                        <a
                          href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary underline"
                        >
                          Read more
                        </a>
                      </TooltipContent>
                    </Tooltip>
                  </span>
                  <Select
                    value={datasetFormat}
                    onValueChange={(v) =>
                      setDatasetFormat(v as typeof datasetFormat)
                    }
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Auto</SelectItem>
                      <SelectItem value="alpaca">Alpaca</SelectItem>
                      <SelectItem value="chatml">ChatML</SelectItem>
                      <SelectItem value="sharegpt">ShareGPT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col gap-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                      Train Split Start
                      <Tooltip>
                        <TooltipTrigger asChild={true}>
                          <button
                            type="button"
                            className="text-foreground/70 hover:text-foreground"
                          >
                            <HugeiconsIcon
                              icon={InformationCircleIcon}
                              className="size-3"
                            />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          Only train on a subset of your training split by
                          specifying a start row index (inclusive, 0-based).
                          Leave empty to start from the first row.
                        </TooltipContent>
                      </Tooltip>
                    </span>
                    <Input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      step={1}
                      placeholder="0"
                      value={datasetSliceStart ?? ""}
                      onChange={(e) =>
                        setDatasetSliceStart(normalizeSliceInput(e.target.value))
                      }
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                      Train Split End
                      <Tooltip>
                        <TooltipTrigger asChild={true}>
                          <button
                            type="button"
                            className="text-foreground/70 hover:text-foreground"
                          >
                            <HugeiconsIcon
                              icon={InformationCircleIcon}
                              className="size-3"
                            />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          Last row index to include from the training split
                          (inclusive, 0-based). For example, set Start to 0 and
                          End to 99 to train on the first 100 rows. Leave empty
                          to use all remaining rows.
                        </TooltipContent>
                      </Tooltip>
                    </span>
                    <Input
                      type="number"
                      inputMode="numeric"
                      min={0}
                      step={1}
                      placeholder="End"
                      value={datasetSliceEnd ?? ""}
                      onChange={(e) =>
                        setDatasetSliceEnd(normalizeSliceInput(e.target.value))
                      }
                    />
                  </div>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          <div className="flex flex-col gap-4 pt-1">
            {selectedDatasetName ? (
              <div className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3.5 py-3">
                <div className="rounded-md bg-indigo-500/10 p-1.5">
                  <HugeiconsIcon
                    icon={FileAttachmentIcon}
                    className="size-4 text-indigo-500"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-mono text-sm font-medium truncate">
                    {datasetSource === "upload"
                      ? selectedLocalDataset?.label ??
                        deriveLocalDatasetName(selectedDatasetName)
                      : selectedDatasetName}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {datasetSource === "upload" ? (
                      uploadedFile ? (
                        <>
                          Local dataset
                          {selectedLocalRows != null
                            ? ` / ${selectedLocalRows.toLocaleString()} rows`
                            : ""}
                        </>
                      ) : (
                        "Local dataset"
                      )
                    ) : (
                      <>
                        Hugging Face Dataset
                        {datasetSubset && ` / ${datasetSubset}`}
                        {datasetSplit && ` / ${datasetSplit}`}
                      </>
                    )}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="shrink-0 text-xs"
                  onClick={() => clearSelectionForTab(activeSourceTab)}
                >
                  Clear
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-3 rounded-lg border border-dashed bg-muted/20 px-3.5 py-3">
                <HugeiconsIcon
                  icon={Database02Icon}
                  className="size-4 text-muted-foreground/40"
                />
                <span className="text-xs text-muted-foreground">
                  No dataset selected
                </span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                size="sm"
                className="cursor-pointer gap-1.5"
                disabled={isUploading}
                onClick={handleUploadButtonClick}
              >
                {isUploading ? (
                  <Spinner className="size-3.5" />
                ) : (
                  <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
                )}
                {isUploading ? "Uploading..." : "Upload"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="cursor-pointer gap-1.5"
                disabled={!selectedDatasetName}
                onClick={() => openPreview()}
              >
                <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
                View dataset
              </Button>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.jsonl,.csv,.parquet,.pdf,.docx,.txt"
            className="hidden"
            onChange={(event) => {
              void handleDatasetFileChange(event);
            }}
          />
          <input
            ref={evalFileInputRef}
            type="file"
            accept=".json,.jsonl,.csv,.parquet"
            className="hidden"
            onChange={(event) => {
              void handleEvalFileChange(event);
            }}
          />
          <DocumentUploadRedirectDialog
            open={documentRedirectOpen}
            onOpenChange={setDocumentRedirectOpen}
            fileName={redirectFileName}
            onOpenLearningRecipes={handleOpenLearningRecipes}
          />
      </div>
      </SectionCard>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-md bg-background/60 px-2 py-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </div>
  );
}
