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
import { openModelsDir } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
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

  const handleInventoryChanged = useCallback(() => {
    onInventoryChange?.();
  }, [onInventoryChange]);

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
          className="gap-0 overflow-hidden p-0 sm:max-w-[620px] lg:max-w-[660px] xl:max-w-[680px] [&_[data-slot=dialog-close]]:right-3 [&_[data-slot=dialog-close]]:top-3"
          overlayClassName="bg-black/20 backdrop-blur-none"
        >
          <DialogHeader className="border-b border-border/60 px-5 py-4">
            <DialogTitle className="text-[0.9375rem]">
              On-device locations
            </DialogTitle>
            <DialogDescription className="sr-only">
              Hugging Face model folders, GGUF files, and adapters are indexed
              here.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 px-5 py-4">
            <div className="rounded-[14px] border border-border/70 bg-muted/20 p-3">
              <div className="mb-2 flex items-center gap-2 text-[0.75rem] font-medium text-foreground">
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
                    className="field-soft h-9 rounded-full pl-9 pr-3 font-mono text-[0.75rem] placeholder:font-sans"
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
                    className="h-9 rounded-full px-3 text-[0.78125rem]"
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
              <div className="rounded-[10px] border border-destructive/20 bg-destructive/5 px-3 py-2 text-[0.75rem] text-destructive">
                {error}
              </div>
            ) : null}

            <div className="overflow-hidden rounded-[14px] border border-border/70">
              <div className="flex h-10 items-center justify-between border-b border-border/60 px-3">
                <span className="text-[0.75rem] font-medium text-foreground">
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
                  <div className="flex h-24 items-center justify-center gap-2 text-[0.75rem] text-muted-foreground">
                    <Spinner className="size-3.5" />
                    Loading locations...
                  </div>
                ) : sortedFolders.length === 0 ? (
                  <div className="flex h-28 flex-col items-center justify-center gap-2 px-4 text-center text-[0.75rem] text-muted-foreground">
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
                            ? "grid-cols-[32px_minmax(0,1fr)_32px_32px]"
                            : "grid-cols-[32px_minmax(0,1fr)_32px]",
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
                            className="block w-full truncate text-[0.78125rem] font-medium text-foreground"
                            title={pathTail(folder.path)}
                          >
                            {pathTail(folder.path)}
                          </p>
                          <Tooltip>
                            <TooltipTrigger asChild={true}>
                              <p className="block w-full truncate font-mono text-[0.65625rem] text-muted-foreground">
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
    </>
  );
}
