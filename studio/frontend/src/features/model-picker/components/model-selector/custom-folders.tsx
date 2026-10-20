// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model picker's Custom Folders section: the scan folders, the add-by-path and browse
// controls, and the folders worth suggesting. Shared by the chat picker and the voice Decide list.

import { useIsAccountOwner } from "@/features/auth";
import {
  type ScanFolderInfo,
  addScanFolder,
  listRecommendedFolders,
  listScanFolders,
  removeScanFolder,
} from "@/features/chat";
import { scanFolderStatusCopy } from "@/features/hub";
import { pickNativeModelFolder } from "@/features/native-intents";
import { toast } from "@/lib/toast";
import {
  Add01Icon,
  Cancel01Icon,
  Folder02Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";
import { type Ref, useCallback, useEffect, useState } from "react";
import { FolderBrowser } from "./folder-browser";

// Module-level so re-mounting the popover shows the folders instantly.
let _scanFoldersCache: ScanFolderInfo[] = [];

export function useScanFolders(onChanged?: () => void) {
  const [scanFolders, setScanFolders] =
    useState<ScanFolderInfo[]>(_scanFoldersCache);
  const [folderInput, setFolderInput] = useState("");
  const [folderError, setFolderError] = useState<string | null>(null);
  const [showFolderInput, setShowFolderInput] = useState(false);
  const [folderLoading, setFolderLoading] = useState(false);
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);
  const [recommendedFolders, setRecommendedFolders] = useState<string[]>([]);

  const refreshScanFolders = useCallback(() => {
    listScanFolders()
      .then((v) => {
        _scanFoldersCache = v;
        setScanFolders(v);
      })
      .catch(() => {});
  }, []);

  const handleAddFolder = useCallback(
    async (overridePath?: string) => {
      // An explicit path lets the folder browser submit in the same tick it calls `setFolderInput`;
      // reading `folderInput` would race the update.
      const raw = overridePath !== undefined ? overridePath : folderInput;
      const trimmed = raw.trim();
      if (!trimmed || folderLoading) return;
      setFolderError(null);
      setFolderLoading(true);
      // From the folder browser's "Use this folder": the typed-input panel is closed, so surface failures via toast.
      const fromBrowser = overridePath !== undefined;
      try {
        const created = await addScanFolder(trimmed);
        // Backend returns the existing row for duplicates, so dedupe.
        const next = _scanFoldersCache.some(
          (f) => f.id === created.id || f.path === created.path,
        )
          ? _scanFoldersCache
          : [..._scanFoldersCache, created];
        _scanFoldersCache = next;
        setScanFolders(next);
        setFolderInput("");
        setShowFolderInput(false);
        onChanged?.();
        void refreshScanFolders();
      } catch (e) {
        const message = e instanceof Error ? e.message : "Failed to add folder";
        setFolderError(message);
        if (fromBrowser) {
          toast.error("Couldn't add folder", { description: message });
        }
      } finally {
        setFolderLoading(false);
      }
    },
    [folderInput, folderLoading, refreshScanFolders, onChanged],
  );

  const isOwner = useIsAccountOwner();
  const browseForFolder = useCallback(async () => {
    const picked = await pickNativeModelFolder(isOwner);
    if (picked === undefined) {
      setShowFolderBrowser(true);
    } else if (picked) {
      setFolderInput(picked);
      setFolderError(null);
      void handleAddFolder(picked);
    }
  }, [handleAddFolder, isOwner]);

  const handleRemoveFolder = useCallback(
    async (id: number) => {
      try {
        await removeScanFolder(id);
        // Optimistic: drop it immediately.
        const next = _scanFoldersCache.filter((f) => f.id !== id);
        _scanFoldersCache = next;
        setScanFolders(next);
        refreshScanFolders();
        onChanged?.();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed to remove folder");
        refreshScanFolders();
      }
    },
    [refreshScanFolders, onChanged],
  );

  useEffect(() => {
    refreshScanFolders();
    listRecommendedFolders()
      .then(setRecommendedFolders)
      .catch(() => {});
  }, [refreshScanFolders]);

  return {
    scanFolders,
    folderInput,
    setFolderInput,
    folderError,
    setFolderError,
    showFolderInput,
    setShowFolderInput,
    folderLoading,
    showFolderBrowser,
    setShowFolderBrowser,
    recommendedFolders,
    handleAddFolder,
    handleRemoveFolder,
    browseForFolder,
  };
}

export type ScanFolders = ReturnType<typeof useScanFolders>;

/** The heading with its add, browse and collapse controls, the folder list, suggestions and the path input. */
export function CustomFoldersControls({
  folders,
  collapsed,
  onCollapsedChange,
  sectionRef,
}: {
  folders: ScanFolders;
  collapsed: boolean;
  onCollapsedChange: (update: (collapsed: boolean) => boolean) => void;
  sectionRef?: Ref<HTMLDivElement>;
}) {
  const {
    scanFolders,
    folderInput,
    setFolderInput,
    folderError,
    setFolderError,
    showFolderInput,
    setShowFolderInput,
    folderLoading,
    showFolderBrowser,
    setShowFolderBrowser,
    recommendedFolders,
    handleAddFolder,
    handleRemoveFolder,
    browseForFolder,
  } = folders;
  const customFoldersCollapsed = collapsed;
  return (
    <>
      <div
        ref={sectionRef}
        className="mt-3 flex items-center gap-1 border-t border-border px-2.5 pb-1 pt-3"
      >
        <button
          type="button"
          onClick={() => void browseForFolder()}
          title="Browse folders on the server"
          className="flex items-center gap-1.5 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground"
        >
          <HugeiconsIcon icon={Folder02Icon} className="size-3.5" />
          Custom Folders
        </button>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            aria-label={
              showFolderInput
                ? "Cancel adding folder"
                : "Add scan folder by path"
            }
            title={showFolderInput ? "Cancel" : "Add by typing a path"}
            onClick={() => {
              setShowFolderInput((open) => {
                if (open) {
                  setFolderInput("");
                  setFolderError(null);
                }
                return !open;
              });
            }}
            className="shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
          >
            <HugeiconsIcon
              icon={showFolderInput ? Cancel01Icon : Add01Icon}
              className="size-3"
            />
          </button>
          <button
            type="button"
            aria-label="Browse for a folder on the server"
            title="Browse folders on the server"
            onClick={() => void browseForFolder()}
            className="shrink-0 rounded p-0.5 text-muted-foreground/80 transition-colors hover:text-foreground"
          >
            <HugeiconsIcon icon={Search01Icon} className="size-2.5" />
          </button>
        </div>
        <div className="ml-auto">
          <button
            type="button"
            aria-label={
              customFoldersCollapsed
                ? "Expand custom folders"
                : "Collapse custom folders"
            }
            title={customFoldersCollapsed ? "Expand" : "Collapse"}
            onClick={() => onCollapsedChange((v) => !v)}
            className="-mr-0.5 shrink-0 rounded p-1 text-muted-foreground/80 transition-colors hover:text-foreground"
          >
            {customFoldersCollapsed ? (
              <ChevronRightIcon className="size-3" />
            ) : (
              <ChevronDownIcon className="size-3" />
            )}
          </button>
        </div>
      </div>

      {/* Folder paths */}
      {!customFoldersCollapsed &&
        scanFolders.map((f) => {
          const problem = scanFolderStatusCopy(f.status);
          return (
            <div
              key={f.id}
              className="group flex items-center gap-1.5 px-2.5 py-0.5"
            >
              <HugeiconsIcon
                icon={Folder02Icon}
                className="size-3 shrink-0 text-muted-foreground/80"
              />
              <div className="min-w-0 flex-1">
                <span
                  className="block truncate font-mono text-ui-10 text-muted-foreground/80"
                  title={f.path}
                >
                  {f.path}
                </span>
                {problem ? (
                  <span
                    className="block truncate text-ui-10 text-amber-600 dark:text-amber-500"
                    title={problem.hint}
                  >
                    {problem.title}
                  </span>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => handleRemoveFolder(f.id)}
                aria-label={`Remove folder ${f.path}`}
                className="shrink-0 rounded p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:bg-destructive/10 focus-visible:text-destructive"
              >
                <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
              </button>
            </div>
          );
        })}

      {/* Recommended folders */}
      {!customFoldersCollapsed &&
        (() => {
          const registered = new Set(scanFolders.map((f) => f.path));
          const unregistered = recommendedFolders.filter(
            (p) => !registered.has(p),
          );
          if (unregistered.length === 0) return null;
          return (
            <div className="flex flex-wrap gap-1 px-2.5 pb-0.5">
              {unregistered.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => void handleAddFolder(p)}
                  disabled={folderLoading}
                  title={`Add ${p}`}
                  className="rounded-full border border-dashed border-border px-2 py-0.5 font-mono text-ui-10 text-muted-foreground/80 transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:bg-accent hover:text-foreground disabled:opacity-40"
                >
                  <span className="text-ui-11 font-semibold">+</span>{" "}
                  {p.length > 30 ? `...${p.slice(-27)}` : p}
                </button>
              ))}
            </div>
          );
        })()}

      {/* Add folder input */}
      {!customFoldersCollapsed && showFolderInput && (
        <div className="px-2.5 pb-1 pt-0.5">
          <div className="flex items-center gap-1">
            <HugeiconsIcon
              icon={Folder02Icon}
              className="size-3 shrink-0 text-muted-foreground/80"
            />
            <input
              value={folderInput}
              onChange={(e) => {
                setFolderInput(e.target.value);
                setFolderError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleAddFolder();
                }
                if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  setShowFolderInput(false);
                  setFolderInput("");
                  setFolderError(null);
                }
              }}
              placeholder="/path/to/models"
              className="h-6 min-w-0 flex-1 rounded border border-border bg-transparent px-1.5 font-mono text-ui-10 text-foreground outline-none placeholder:text-muted-foreground/80 focus:border-[color-mix(in_oklab,var(--foreground)_calc(20%*var(--contrast-edge-gain,1)),transparent)]"
              disabled={folderLoading}
              autoFocus={true}
            />
            <button
              type="button"
              onClick={() => void browseForFolder()}
              disabled={folderLoading}
              aria-label="Browse for folder"
              title="Browse folders on the server"
              className="flex h-6 shrink-0 items-center justify-center rounded border border-border px-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
            >
              <HugeiconsIcon icon={Search01Icon} className="size-3" />
            </button>
            <button
              type="button"
              onClick={() => {
                void handleAddFolder();
              }}
              disabled={folderLoading || !folderInput.trim()}
              className="h-6 shrink-0 rounded border border-border px-1.5 text-ui-10 text-muted-foreground transition-colors hover:bg-accent disabled:opacity-40"
            >
              Add
            </button>
          </div>
          {folderError && (
            <p className="px-0.5 pt-0.5 text-ui-10 text-destructive">
              {folderError}
            </p>
          )}
        </div>
      )}

      <FolderBrowser
        open={showFolderBrowser}
        onOpenChange={setShowFolderBrowser}
        initialPath={folderInput.trim() || undefined}
        onSelect={(picked) => {
          setFolderInput(picked);
          setFolderError(null);
          // Pass the path explicitly: `folderInput` state has not flushed when "Use this folder" submits.
          void handleAddFolder(picked);
        }}
      />
    </>
  );
}
