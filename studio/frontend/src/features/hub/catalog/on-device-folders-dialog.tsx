// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useIsAccountOwner } from "@/features/auth";
import {
  type ScanFolderInfo,
  addScanFolderToStore,
  countModelsPerFolder,
  folderDisplayName,
  formatBytes,
  refreshScanFolders,
  removeScanFolderFromStore,
  scanFolderStatusCopy,
  setScanFolderRecursive,
  useDeviceInventorySources,
  useScanFolders,
} from "@/features/hub";
import { FolderBrowser } from "@/features/model-picker";
import {
  openModelsDir,
  pickHuggingFaceCacheDir,
  pickNativeModelFolder,
} from "@/features/native-intents";
import {
  type HuggingFaceCacheSettings,
  loadHuggingFaceCacheSettings,
  updateHuggingFaceCacheSettings,
} from "@/features/settings";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Download01Icon,
  InformationCircleIcon,
  PlusSignIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { FolderAddIcon, FolderExportIcon, FolderOpenIcon, FolderSearchIcon } from "@/lib/folder-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { setDownloadDir, useDownloadDir } from "../download-manager";

type Pending = "add" | `remove:${number}` | `recursive:${number}` | null;

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatFreeSpace(bytes: number | null): string | null {
  if (bytes === null || !Number.isFinite(bytes)) return null;
  return `${formatBytes(bytes)} free`;
}

function modelCountLabel(count: number): string {
  if (count === 0) return "No models found";
  return count === 1 ? "1 model" : `${count} models`;
}

export function OnDeviceFoldersDialog({
  open,
  onOpenChange,
  onInventoryChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** For lists outside the shared inventory (the Images LoRA list); the inventory itself refreshes on its own. */
  onInventoryChange?: () => void;
}) {
  const isOwner = useIsAccountOwner();
  const { folders, loaded, loading, error: listError } = useScanFolders(open);
  const { localModels } = useDeviceInventorySources(["localModels"], {
    enabled: open,
  });
  const [path, setPath] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [pending, setPending] = useState<Pending>(null);
  const [downloadCache, setDownloadCache] =
    useState<HuggingFaceCacheSettings | null>(null);
  const [downloadCacheLoaded, setDownloadCacheLoaded] = useState(false);
  const [downloadBrowserOpen, setDownloadBrowserOpen] = useState(false);
  const [downloadSaving, setDownloadSaving] = useState(false);
  const browserDownloadDir = useDownloadDir();

  const sortedFolders = useMemo(
    () =>
      [...folders].sort(
        (a, b) =>
          folderDisplayName(a.path).localeCompare(folderDisplayName(b.path)) ||
          a.path.localeCompare(b.path),
      ),
    [folders],
  );

  const modelCounts = useMemo(
    () =>
      localModels.ready
        ? countModelsPerFolder(
            folders.map((folder) => folder.path),
            localModels.rows.map((row) => row.path),
          )
        : null,
    [folders, localModels.ready, localModels.rows],
  );

  // A finished inventory scan is what records a folder as slow or unreadable.
  const scannedAt = localModels.refreshedAt;
  useEffect(() => {
    if (open && scannedAt !== null) {
      void refreshScanFolders().catch(() => undefined);
    }
  }, [open, scannedAt]);

  useEffect(() => {
    if (!open) return;
    setInputError(null);
    let cancelled = false;
    // The dialog stays mounted, so re-arm the flag or a reopen shows stale data.
    setDownloadCacheLoaded(false);
    loadHuggingFaceCacheSettings()
      // Null drops the stale path rather than offer Change against a location we could not confirm.
      .catch(() => null)
      .then((settings) => {
        if (cancelled) return;
        setDownloadCache(settings);
        setDownloadCacheLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  // updateHuggingFaceCacheSettings bumps the inventory version, which re-fetches every source;
  // refreshing here too would scan twice under two keys that cannot be deduplicated.
  const saveDownloadLocation = useCallback(async (nextPath: string | null) => {
    setDownloadSaving(true);
    try {
      const settings = await updateHuggingFaceCacheSettings(nextPath);
      setDownloadCache(settings);
      toast.success("Models folder updated", {
        description: settings.cacheHome,
      });
    } catch (err) {
      toast.error("Couldn't change the Models folder", {
        description: formatError(err),
      });
    } finally {
      setDownloadSaving(false);
    }
  }, []);

  const changeDownloadLocation = useCallback(async () => {
    if (!isTauri) {
      setDownloadBrowserOpen(true);
      return;
    }
    try {
      const picked = await pickHuggingFaceCacheDir();
      if (picked) await saveDownloadLocation(picked);
    } catch (err) {
      toast.error("Couldn't open the folder picker", {
        description: formatError(err),
      });
    }
  }, [saveDownloadLocation]);

  const handleAdd = useCallback(
    async (rawPath: string, source: "input" | "browse") => {
      const nextPath = rawPath.trim();
      if (!nextPath || pending) return;
      setPending("add");
      setInputError(null);
      try {
        const { folder, alreadyAdded } = await addScanFolderToStore(nextPath);
        setPath("");
        if (alreadyAdded) {
          toast.info("Already in your folders", {
            description: folderDisplayName(folder.path),
          });
          return;
        }
        toast.success("Folder added", {
          description: folderDisplayName(folder.path),
        });
        onInventoryChange?.();
      } catch (err) {
        const message = formatError(err);
        if (source === "input") setInputError(message);
        else toast.error("Couldn't add folder", { description: message });
      } finally {
        setPending(null);
      }
    },
    [onInventoryChange, pending],
  );

  const browseLocations = useCallback(async () => {
    try {
      const picked = await pickNativeModelFolder(isOwner);
      if (picked === undefined) setBrowserOpen(true);
      else if (picked) await handleAdd(picked, "browse");
    } catch (err) {
      toast.error("Couldn't open the folder picker", {
        description: formatError(err),
      });
    }
  }, [handleAdd, isOwner]);

  const handleOpen = useCallback(async (folder: ScanFolderInfo) => {
    try {
      await openModelsDir(folder.path);
    } catch (err) {
      toast.error("Couldn't open folder", { description: formatError(err) });
    }
  }, []);

  const handleRecursive = useCallback(
    async (folder: ScanFolderInfo, recursive: boolean) => {
      if (pending) return;
      setPending(`recursive:${folder.id}`);
      try {
        await setScanFolderRecursive(folder, recursive);
        onInventoryChange?.();
      } catch (err) {
        toast.error("Couldn't change the folder", {
          description: formatError(err),
        });
      } finally {
        setPending(null);
      }
    },
    [onInventoryChange, pending],
  );

  const handleRemove = useCallback(
    async (folder: ScanFolderInfo) => {
      if (pending) return;
      setPending(`remove:${folder.id}`);
      try {
        await removeScanFolderFromStore(folder.id);
        onInventoryChange?.();
        // Removing only stops the scan, so an undo is safer than a confirm step.
        toast.success("Folder removed", {
          description: folderDisplayName(folder.path),
          action: {
            label: "Undo",
            onClick: () => {
              addScanFolderToStore(folder.path, Boolean(folder.recursive))
                .then(() => onInventoryChange?.())
                .catch((err) =>
                  toast.error("Couldn't restore folder", {
                    description: formatError(err),
                  }),
                );
            },
          },
        });
      } catch (err) {
        toast.error("Couldn't remove folder", {
          description: formatError(err),
        });
      } finally {
        setPending(null);
      }
    },
    [onInventoryChange, pending],
  );

  const freeSpace = formatFreeSpace(downloadCache?.freeBytes ?? null);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent
          className="flex max-h-[90dvh] flex-col gap-0 overflow-hidden p-0 sm:max-w-[calc(620px*var(--ui-space-scale,1))] lg:max-w-[calc(660px*var(--ui-space-scale,1))] xl:max-w-[calc(680px*var(--ui-space-scale,1))] [&_[data-slot=dialog-close]]:right-3 [&_[data-slot=dialog-close]]:top-3"
          overlayClassName="bg-black/20 backdrop-blur-none"
        >
          <DialogHeader className="shrink-0 border-b border-border/60 px-5 py-4">
            <DialogTitle className="text-ui-15">Model folders</DialogTitle>
            <DialogDescription className="text-ui-12 text-muted-foreground">
              Unsloth lists models from your Models folder and from any custom
              folders you add. Nothing is moved or copied.
            </DialogDescription>
          </DialogHeader>

          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
            <section
              aria-labelledby="models-folder-heading"
              className="rounded-lg border border-border/70 bg-muted/20 p-3"
            >
              <div
                id="models-folder-heading"
                className="mb-2 flex items-center gap-2 text-ui-12 font-medium text-foreground"
              >
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-3.5 text-muted-foreground"
                />
                Models folder
                <span className="font-normal text-muted-foreground">
                  New downloads are saved here
                </span>
              </div>

              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <Input
                  readOnly={true}
                  aria-label="Models folder"
                  value={
                    downloadCache?.cacheHome ??
                    (downloadCacheLoaded ? "Unavailable" : "Loading...")
                  }
                  title={downloadCache?.cacheHome}
                  className="field-soft h-control min-w-0 flex-1 rounded-full px-3 font-mono text-ui-12"
                />
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void changeDownloadLocation()}
                    disabled={!downloadCache?.editable || downloadSaving}
                  >
                    {downloadSaving ? (
                      <Spinner className="size-3.5" />
                    ) : (
                      <HugeiconsIcon
                        icon={FolderSearchIcon}
                        strokeWidth={1.75}
                        data-icon="inline-start"
                        className="size-3.5"
                      />
                    )}
                    Change
                  </Button>
                  {downloadCache?.isCustom &&
                  downloadCache.source !== "environment" ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => void saveDownloadLocation(null)}
                      disabled={downloadSaving}
                      className="h-9 rounded-full px-3 text-ui-12p5 text-muted-foreground"
                    >
                      Use default
                    </Button>
                  ) : null}
                </div>
              </div>

              <p className="mt-2 text-ui-10p5 text-muted-foreground">
                {downloadCache?.source === "environment"
                  ? `Set by the ${
                      downloadCache.environmentVariable ?? "HF_HOME"
                    } environment variable. Unset it to choose a folder here.`
                  : [
                      "Models you already downloaded stay where they are.",
                      freeSpace,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
              </p>
              {isOwner && browserDownloadDir ? (
                <p className="mt-1.5 text-ui-10p5 text-muted-foreground">
                  Downloads from this browser go to{" "}
                  <span
                    className="font-mono text-foreground"
                    title={browserDownloadDir}
                  >
                    {folderDisplayName(browserDownloadDir)}
                  </span>{" "}
                  instead (Download to menu).{" "}
                  <button
                    type="button"
                    onClick={() => setDownloadDir(null)}
                    className="font-medium text-foreground underline-offset-2 hover:underline"
                  >
                    Use the Models folder
                  </button>
                </p>
              ) : null}
            </section>

            <section
              aria-labelledby="custom-folders-heading"
              className="overflow-hidden rounded-xl border border-border/70"
            >
              <div className="flex h-10 items-center justify-between gap-2 border-b border-border/60 px-3">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span
                    id="custom-folders-heading"
                    className="text-ui-12 font-medium text-foreground"
                  >
                    Custom folders
                  </span>
                  {loaded && folders.length > 0 ? (
                    <span className="text-ui-11 text-muted-foreground">
                      {folders.length}
                    </span>
                  ) : null}
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        aria-label="What can a custom folder hold?"
                        className="inline-flex size-6 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          strokeWidth={1.75}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent
                      side="bottom"
                      className="tooltip-compact max-w-xs"
                    >
                      GGUF files, Hugging Face model folders or caches, LM
                      Studio and Ollama libraries, LoRA adapters, and ComfyUI
                      installs (diffusion_models, unet, checkpoints and
                      extra_model_paths.yaml).
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label="Refresh folders"
                      onClick={() =>
                        void refreshScanFolders().catch(() => undefined)
                      }
                      disabled={loading}
                      className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                    >
                      <HugeiconsIcon
                        icon={Refresh01Icon}
                        strokeWidth={1.75}
                        className={cn("size-3.5", loading && "animate-spin")}
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="tooltip-compact">
                    Refresh
                  </TooltipContent>
                </Tooltip>
              </div>

              <div className="border-b border-border/60 bg-muted/20 p-3">
                <form
                  className="flex flex-col gap-2 sm:flex-row sm:items-center"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void handleAdd(path, "input");
                  }}
                >
                  <Input
                    value={path}
                    onChange={(event) => {
                      setPath(event.target.value);
                      if (inputError) setInputError(null);
                    }}
                    aria-label="Folder path"
                    aria-invalid={inputError ? true : undefined}
                    aria-describedby={
                      inputError ? "custom-folder-error" : undefined
                    }
                    placeholder="Paste a folder path"
                    spellCheck={false}
                    autoComplete="off"
                    className="field-soft h-control min-w-0 flex-1 rounded-full px-3 font-mono text-ui-12 placeholder:font-sans"
                  />
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => void browseLocations()}
                      disabled={pending !== null}
                    >
                      <HugeiconsIcon
                        icon={FolderSearchIcon}
                        strokeWidth={1.75}
                        data-icon="inline-start"
                        className="size-3.5"
                      />
                      Browse...
                    </Button>
                    <Button
                      type="submit"
                      size="sm"
                      disabled={!path.trim() || pending !== null}
                    >
                      {pending === "add" ? (
                        <Spinner className="size-3.5" />
                      ) : (
                        <HugeiconsIcon
                          icon={PlusSignIcon}
                          strokeWidth={1.75}
                          data-icon="inline-start"
                          className="size-3.5"
                        />
                      )}
                      Add
                    </Button>
                  </div>
                </form>
                {inputError ? (
                  <p
                    id="custom-folder-error"
                    role="alert"
                    className="mt-2 px-1 text-ui-11 text-destructive"
                  >
                    {inputError}
                  </p>
                ) : null}
              </div>

              <div className="max-h-72 overflow-y-auto">
                {!loaded && loading ? (
                  <div className="flex h-24 items-center justify-center gap-2 text-ui-12 text-muted-foreground">
                    <Spinner className="size-3.5" />
                    Loading folders...
                  </div>
                ) : !loaded && listError ? (
                  <div className="flex h-24 items-center justify-center px-4 text-center text-ui-12 text-destructive">
                    {listError}
                  </div>
                ) : sortedFolders.length === 0 ? (
                  <div className="flex flex-col items-center justify-center gap-1.5 px-6 py-6 text-center">
                    <HugeiconsIcon
                      icon={FolderAddIcon}
                      strokeWidth={1.75}
                      className="size-5 text-muted-foreground/60"
                    />
                    <p className="text-ui-12 font-medium text-foreground">
                      No custom folders yet
                    </p>
                    <p className="max-w-sm text-ui-11 text-muted-foreground">
                      Add a folder where you keep models, such as an LM Studio,
                      Ollama or ComfyUI folder, or an external drive.
                    </p>
                  </div>
                ) : (
                  <ul aria-labelledby="custom-folders-heading">
                    {sortedFolders.map((folder) => (
                      <FolderRow
                        key={folder.id}
                        folder={folder}
                        modelCount={modelCounts?.get(folder.path) ?? null}
                        pending={pending}
                        onOpen={handleOpen}
                        onRecursive={handleRecursive}
                        onRemove={handleRemove}
                      />
                    ))}
                  </ul>
                )}
              </div>
            </section>
          </div>
        </DialogContent>
      </Dialog>

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(selectedPath) => void handleAdd(selectedPath, "browse")}
      />

      <FolderBrowser
        open={!isTauri && downloadBrowserOpen}
        onOpenChange={setDownloadBrowserOpen}
        onSelect={(selectedPath) => void saveDownloadLocation(selectedPath)}
        initialPath={downloadCache?.cacheHome}
        title="Choose the Models folder"
        confirmLabel="Use for new downloads"
        showModelHints={false}
      />
    </>
  );
}

function FolderRow({
  folder,
  modelCount,
  pending,
  onOpen,
  onRecursive,
  onRemove,
}: {
  folder: ScanFolderInfo;
  modelCount: number | null;
  pending: Pending;
  onOpen: (folder: ScanFolderInfo) => void;
  onRecursive: (folder: ScanFolderInfo, recursive: boolean) => void;
  onRemove: (folder: ScanFolderInfo) => void;
}) {
  const name = folderDisplayName(folder.path);
  const problem = scanFolderStatusCopy(folder.status);
  const removing = pending === `remove:${folder.id}`;
  const toggling = pending === `recursive:${folder.id}`;
  const switchId = `scan-folder-recursive-${folder.id}`;
  return (
    <li className="flex min-h-14 items-center gap-3 border-b border-border/50 px-3 py-2.5 last:border-b-0">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-sm bg-muted text-muted-foreground">
        <HugeiconsIcon
          icon={FolderOpenIcon}
          strokeWidth={1.75}
          className="size-4"
        />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-baseline gap-2">
          <p
            className="truncate text-ui-12p5 font-medium text-foreground"
            title={name}
          >
            {name}
          </p>
          {modelCount !== null && !problem ? (
            <span className="shrink-0 text-ui-10p5 text-muted-foreground">
              {modelCountLabel(modelCount)}
            </span>
          ) : null}
        </div>
        <p
          className="truncate font-mono text-ui-10p5 text-muted-foreground"
          title={folder.path}
        >
          {folder.path}
        </p>
        {problem ? (
          <p
            data-testid={`scan-folder-problem-${folder.id}`}
            className="mt-1 text-ui-10p5 text-amber-600 dark:text-amber-500"
          >
            {problem.title}. {problem.hint}
          </p>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <label
              htmlFor={switchId}
              className="flex cursor-pointer items-center gap-1.5 rounded-full px-1.5 py-1 text-ui-11 text-muted-foreground"
            >
              <span className="hidden sm:inline">Sub-folders</span>
              {toggling ? (
                <Spinner className="size-3.5" />
              ) : (
                <Switch
                  id={switchId}
                  size="sm"
                  checked={Boolean(folder.recursive)}
                  onCheckedChange={(checked) => onRecursive(folder, checked)}
                  disabled={pending !== null}
                  aria-label={`Also scan sub-folders of ${name}`}
                />
              )}
            </label>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="tooltip-compact">
            Also scan sub-folders
          </TooltipContent>
        </Tooltip>
        {isTauri ? (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                aria-label={`Open ${name} in file manager`}
                onClick={() => onOpen(folder)}
                className="inline-flex size-8 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={FolderExportIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent side="left" className="tooltip-compact">
              Open in file manager
            </TooltipContent>
          </Tooltip>
        ) : null}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={`Remove ${name}`}
              onClick={() => onRemove(folder)}
              disabled={pending !== null}
              className="inline-flex size-8 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
            >
              {removing ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="left" className="tooltip-compact">
            Remove from list (files stay on disk)
          </TooltipContent>
        </Tooltip>
      </div>
    </li>
  );
}
