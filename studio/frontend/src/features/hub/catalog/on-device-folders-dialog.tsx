


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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type ScanFolderInfo,
  addScanFolder,
  listScanFolders,
  removeScanFolder,
} from "@/features/hub";
import { FolderBrowser } from "@/features/model-picker";
import {
  openModelsDir,
  pickHuggingFaceCacheDir,
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
  DownloadCircle01Icon,
  FileSearchIcon,
  FolderAddIcon,
  FolderExportIcon,
  FolderOpenIcon,
  FolderSearchIcon,
  PlusSignIcon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

function pathTail(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? path;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatFreeSpace(bytes: number | null): string | null {
  if (bytes === null || !Number.isFinite(bytes)) return null;
  const gb = bytes / 1024 ** 3;
  return gb >= 10 ? `${Math.round(gb)} GB free` : `${gb.toFixed(1)} GB free`;
}

export function OnDeviceFoldersDialog({
  open,
  onOpenChange,
  onInventoryChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onInventoryChange?: () => void;
}) {
  const [folders, setFolders] = useState<ScanFolderInfo[]>([]);
  const [path, setPath] = useState("");
  const [browserOpen, setBrowserOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<"add" | `remove:${number}` | null>(
    null,
  );
  const refreshIdRef = useRef(0);
  const mutationVersionRef = useRef(0);
  const [downloadCache, setDownloadCache] =
    useState<HuggingFaceCacheSettings | null>(null);
  const [downloadCacheLoaded, setDownloadCacheLoaded] = useState(false);
  const [downloadBrowserOpen, setDownloadBrowserOpen] = useState(false);
  const [downloadSaving, setDownloadSaving] = useState(false);

  const sortedFolders = useMemo(
    () => [...folders].sort((a, b) => a.path.localeCompare(b.path)),
    [folders],
  );

  const refreshFolders = useCallback(() => {
    const refreshId = refreshIdRef.current + 1;
    const mutationVersion = mutationVersionRef.current;
    refreshIdRef.current = refreshId;
    setLoading(true);
    setError(null);
    listScanFolders()
      .then((nextFolders) => {
        if (refreshIdRef.current !== refreshId) {
          return;
        }
        if (mutationVersionRef.current !== mutationVersion) {
          return;
        }
        setFolders(nextFolders);
      })
      .catch((err) => {
        if (refreshIdRef.current === refreshId) {
          setError(formatError(err));
        }
      })
      .finally(() => {
        if (refreshIdRef.current === refreshId) {
          setLoading(false);
        }
      });
  }, []);

  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(refreshFolders, 0);
    return () => window.clearTimeout(timer);
  }, [open, refreshFolders]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    // The dialog stays mounted between opens, so re-arm the flag or a reopen
    // shows the previous answer as if it were fresh.
    setDownloadCacheLoaded(false);
    loadHuggingFaceCacheSettings()
      // Indexed locations do not depend on this. Null drops the stale path
      // rather than offer Change against a location we could not confirm.
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

  const handleInventoryChanged = useCallback(() => {
    onInventoryChange?.();
  }, [onInventoryChange]);

  // Relocating the cache changes which repos are on disk, but
  // updateHuggingFaceCacheSettings already bumps the inventory version, which
  // re-fetches every source. Refreshing here too would scan twice, since the
  // two rounds carry different version keys and cannot be deduplicated.
  const saveDownloadLocation = useCallback(async (nextPath: string | null) => {
    setDownloadSaving(true);
    try {
      const settings = await updateHuggingFaceCacheSettings(nextPath);
      setDownloadCache(settings);
      toast.success("Download location updated", {
        description: settings.cacheHome,
      });
    } catch (err) {
      toast.error("Couldn't update the download location", {
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
    async (rawPath: string) => {
      const nextPath = rawPath.trim();
      if (!nextPath || pending) return;
      setPending("add");
      setError(null);
      try {
        const folder = await addScanFolder(nextPath);
        setPath("");
        mutationVersionRef.current += 1;
        setFolders((current) => {
          const withoutDuplicate = current.filter(
            (row) => row.id !== folder.id,
          );
          return [...withoutDuplicate, folder];
        });
        toast.success("Location added", {
          description: pathTail(folder.path),
        });
        handleInventoryChanged();
      } catch (err) {
        const message = formatError(err);
        setError(message);
        toast.error("Couldn't add location", { description: message });
      } finally {
        setPending(null);
      }
    },
    [handleInventoryChanged, pending],
  );

  // Scan folders are arbitrary paths that may be moved or deleted after they
  // were registered, so surface the command's failure as a toast.
  const handleOpen = useCallback(async (folder: ScanFolderInfo) => {
    try {
      await openModelsDir(folder.path);
    } catch (err) {
      toast.error("Couldn't open location", { description: formatError(err) });
    }
  }, []);

  const handleRemove = useCallback(
    async (folder: ScanFolderInfo) => {
      const key = `remove:${folder.id}` as const;
      if (pending) return;
      setPending(key);
      setError(null);
      try {
        await removeScanFolder(folder.id);
        mutationVersionRef.current += 1;
        setFolders((current) => current.filter((row) => row.id !== folder.id));
        toast.success("Location removed", {
          description: pathTail(folder.path),
        });
        handleInventoryChanged();
      } catch (err) {
        const message = formatError(err);
        setError(message);
        toast.error("Couldn't remove location", { description: message });
      } finally {
        setPending(null);
      }
    },
    [handleInventoryChanged, pending],
  );

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent
          className="flex max-h-[90dvh] flex-col gap-0 overflow-hidden p-0 sm:max-w-[620px] lg:max-w-[660px] xl:max-w-[680px] [&_[data-slot=dialog-close]]:right-3 [&_[data-slot=dialog-close]]:top-3"
          overlayClassName="bg-black/20 backdrop-blur-none"
        >
          <DialogHeader className="shrink-0 border-b border-border/60 px-5 py-4">
            <DialogTitle className="text-ui-15">
              On-device locations
            </DialogTitle>
            <DialogDescription className="sr-only">
              Hugging Face model folders, GGUF files, and adapters are indexed
              here.
            </DialogDescription>
          </DialogHeader>

          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 py-4">
            <div className="rounded-[14px] border border-border/70 bg-muted/20 p-3">
              <div className="mb-2 flex items-center gap-2 text-ui-12 font-medium text-foreground">
                <HugeiconsIcon
                  icon={DownloadCircle01Icon}
                  strokeWidth={1.75}
                  className="size-3.5 text-muted-foreground"
                />
                Download location
              </div>

              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <Input
                  readOnly={true}
                  aria-label="Model download location"
                  value={
                    downloadCache?.cacheHome ??
                    (downloadCacheLoaded ? "Unknown" : "Loading...")
                  }
                  title={downloadCache?.cacheHome}
                  className="field-soft h-9 min-w-0 flex-1 rounded-full px-3 font-mono text-ui-12"
                />
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void changeDownloadLocation()}
                    disabled={!downloadCache?.editable || downloadSaving}
                    className="h-9 rounded-full px-3 text-ui-12p5"
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
                  {downloadCache?.isCustom ? (
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
                  ? `Managed by the ${
                      downloadCache.environmentVariable ?? "HF_HOME"
                    } environment variable.`
                  : [
                      "New downloads only. Models already on disk stay where they are.",
                      formatFreeSpace(downloadCache?.freeBytes ?? null),
                    ]
                      .filter(Boolean)
                      .join(" · ")}
              </p>
            </div>

            <div className="rounded-[14px] border border-border/70 bg-muted/20 p-3">
              <div className="mb-2 flex items-center gap-2 text-ui-12 font-medium text-foreground">
                <HugeiconsIcon
                  icon={FolderAddIcon}
                  strokeWidth={1.75}
                  className="size-3.5 text-muted-foreground"
                />
                Add location
              </div>

              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <div className="relative min-w-0 flex-1">
                  <HugeiconsIcon
                    icon={FileSearchIcon}
                    strokeWidth={1.75}
                    className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                  />
                  <Input
                    value={path}
                    onChange={(event) => setPath(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter") return;
                      event.preventDefault();
                      void handleAdd(path);
                    }}
                    placeholder="Paste model folder or file path"
                    className="field-soft h-9 rounded-full pl-9 pr-3 font-mono text-ui-12 placeholder:font-sans"
                  />
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        onClick={() => setBrowserOpen(true)}
                        aria-label="Browse locations"
                        className="size-9 rounded-full"
                      >
                        <HugeiconsIcon
                          icon={FolderSearchIcon}
                          strokeWidth={1.75}
                          className="size-4"
                        />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="tooltip-compact">
                      Browse
                    </TooltipContent>
                  </Tooltip>
                  <Button
                    type="button"
                    size="sm"
                    onClick={() => void handleAdd(path)}
                    disabled={!path.trim() || pending !== null}
                    className="h-9 rounded-full px-3 text-ui-12p5"
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
              </div>
            </div>

            {error ? (
              <div className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-3 py-2 text-ui-12 text-destructive">
                {error}
              </div>
            ) : null}

            <div className="overflow-hidden rounded-[14px] border border-border/70">
              <div className="flex h-10 items-center justify-between border-b border-border/60 px-3">
                <span className="text-ui-12 font-medium text-foreground">
                  Indexed locations
                </span>
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label="Refresh locations"
                      onClick={refreshFolders}
                      disabled={loading}
                      className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:opacity-50"
                    >
                      <HugeiconsIcon
                        icon={RefreshIcon}
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

              <div className="max-h-64 overflow-y-auto">
                {loading ? (
                  <div className="flex h-24 items-center justify-center gap-2 text-ui-12 text-muted-foreground">
                    <Spinner className="size-3.5" />
                    Loading locations...
                  </div>
                ) : sortedFolders.length === 0 ? (
                  <div className="flex h-28 flex-col items-center justify-center gap-2 px-4 text-center text-ui-12 text-muted-foreground">
                    <HugeiconsIcon
                      icon={FolderOpenIcon}
                      strokeWidth={1.75}
                      className="size-5 text-muted-foreground/60"
                    />
                    No custom locations
                  </div>
                ) : (
                  sortedFolders.map((folder) => {
                    const removing = pending === `remove:${folder.id}`;
                    return (
                      <div
                        key={folder.id}
                        className={cn(
                          "grid min-h-12 w-full items-center gap-3 border-b border-border/50 px-3 py-2 last:border-b-0",
                          isTauri
                            ? "grid-cols-[2rem_minmax(0,1fr)_2rem_2rem]"
                            : "grid-cols-[2rem_minmax(0,1fr)_2rem]",
                        )}
                      >
                        <div className="flex size-8 shrink-0 items-center justify-center rounded-[9px] bg-muted text-muted-foreground">
                          <HugeiconsIcon
                            icon={FolderOpenIcon}
                            strokeWidth={1.75}
                            className="size-4"
                          />
                        </div>
                        <div className="min-w-0 overflow-hidden">
                          <p
                            className="block w-full truncate text-ui-12p5 font-medium text-foreground"
                            title={pathTail(folder.path)}
                          >
                            {pathTail(folder.path)}
                          </p>
                          <Tooltip>
                            <TooltipTrigger asChild={true}>
                              <p className="block w-full truncate font-mono text-ui-10p5 text-muted-foreground">
                                {folder.path}
                              </p>
                            </TooltipTrigger>
                            <TooltipContent
                              side="bottom"
                              className="tooltip-compact max-w-xs break-all"
                            >
                              {folder.path}
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        {isTauri ? (
                          <Tooltip>
                            <TooltipTrigger asChild={true}>
                              <button
                                type="button"
                                aria-label={`Open ${folder.path}`}
                                onClick={() => void handleOpen(folder)}
                                className="inline-flex size-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                              >
                                <HugeiconsIcon
                                  icon={FolderExportIcon}
                                  strokeWidth={1.75}
                                  className="size-4"
                                />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent
                              side="left"
                              className="tooltip-compact"
                            >
                              Open in file manager
                            </TooltipContent>
                          </Tooltip>
                        ) : null}
                        <Tooltip>
                          <TooltipTrigger asChild={true}>
                            <button
                              type="button"
                              aria-label={`Remove ${folder.path}`}
                              onClick={() => void handleRemove(folder)}
                              disabled={pending !== null}
                              className="inline-flex size-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive disabled:opacity-50"
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
                          <TooltipContent
                            side="left"
                            className="tooltip-compact"
                          >
                            Remove from list
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(selectedPath) => void handleAdd(selectedPath)}
      />

      <FolderBrowser
        open={!isTauri && downloadBrowserOpen}
        onOpenChange={setDownloadBrowserOpen}
        onSelect={(selectedPath) => void saveDownloadLocation(selectedPath)}
        initialPath={downloadCache?.cacheHome}
        title="Choose model download location"
        confirmLabel="Use for future downloads"
        showModelHints={false}
      />
    </>
  );
}
