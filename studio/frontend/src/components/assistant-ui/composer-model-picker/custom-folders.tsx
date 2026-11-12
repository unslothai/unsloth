// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useIsAccountOwner } from "@/features/auth";
import { listRecommendedFolders } from "@/features/chat";
import {
  type ScanFolderInfo,
  addScanFolderToStore,
  folderDisplayName,
  removeScanFolderFromStore,
  scanFolderStatusCopy,
  useScanFolders,
} from "@/features/hub";
import { FolderBrowser } from "@/features/model-picker";
import { pickNativeModelFolder } from "@/features/native-intents";
import { Folder02Icon } from "@/lib/folder-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  InformationCircleIcon,
  PlusSignIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useState } from "react";

const ICON_BUTTON =
  "grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground";

function Hint({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild={true}>{children}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

export function useCustomFolders({
  enabled,
  onFoldersChange,
}: {
  enabled: boolean;
  onFoldersChange?: () => void;
}): { actions: ReactNode; body: ReactNode } {
  const { folders } = useScanFolders(enabled);
  const [recommended, setRecommended] = useState<string[]>([]);
  const [typing, setTyping] = useState(false);
  const [path, setPath] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const isOwner = useIsAccountOwner();

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let live = true;
    listRecommendedFolders()
      .then((found) => {
        if (live) {
          setRecommended(found);
        }
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [enabled]);

  const add = useCallback(
    async (raw: string, fromBrowser: boolean) => {
      const trimmed = raw.trim();
      if (!trimmed || busy) {
        return;
      }
      setError(null);
      setBusy(true);
      try {
        const { folder, alreadyAdded } = await addScanFolderToStore(trimmed);
        setPath("");
        setTyping(false);
        if (alreadyAdded) {
          toast.info("Already in your folders", {
            description: folderDisplayName(folder.path),
          });
        }
        onFoldersChange?.();
      } catch (e) {
        const message = e instanceof Error ? e.message : "Failed to add folder";
        setError(message);
        if (fromBrowser) {
          toast.error("Couldn't add folder", { description: message });
        }
      } finally {
        setBusy(false);
      }
    },
    [busy, onFoldersChange],
  );

  const browse = useCallback(async () => {
    const picked = await pickNativeModelFolder(isOwner);
    if (picked === undefined) {
      setBrowsing(true);
    } else if (picked) {
      void add(picked, true);
    }
  }, [add, isOwner]);

  const remove = useCallback(
    async (folder: ScanFolderInfo) => {
      try {
        await removeScanFolderFromStore(folder.id);
        onFoldersChange?.();
        toast.success("Folder removed", {
          description: folderDisplayName(folder.path),
          action: {
            label: "Undo",
            onClick: () => {
              addScanFolderToStore(folder.path, Boolean(folder.recursive))
                .then(() => onFoldersChange?.())
                .catch((e) =>
                  toast.error("Couldn't restore folder", {
                    description: e instanceof Error ? e.message : undefined,
                  }),
                );
            },
          },
        });
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Failed to remove folder");
      }
    },
    [onFoldersChange],
  );

  const actions = (
    <>
      <Hint label={typing ? "Cancel" : "Add by typing a path"}>
        <button
          type="button"
          aria-label={
            typing ? "Cancel adding folder" : "Add scan folder by path"
          }
          onClick={() => {
            setTyping((open) => !open);
            setPath("");
            setError(null);
          }}
          className={ICON_BUTTON}
        >
          <HugeiconsIcon
            icon={typing ? Cancel01Icon : PlusSignIcon}
            strokeWidth={1.75}
            className="size-3"
          />
        </button>
      </Hint>
      <Hint label="Browse folders on the server">
        <button
          type="button"
          aria-label="Browse for a folder on the server"
          onClick={() => void browse()}
          className={ICON_BUTTON}
        >
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={1.75}
            className="size-3"
          />
        </button>
      </Hint>
    </>
  );

  const registered = new Set(folders.map((f) => f.path));
  const suggestions = recommended.filter((p) => !registered.has(p));
  const body = (
    <div className="flex shrink-0 flex-col gap-0.5">
      {folders.map((folder) => {
        const problem = scanFolderStatusCopy(folder.status);
        return (
          <div
            key={folder.id}
            className="group/folder flex h-[calc(28px*var(--ui-space-scale,1))] shrink-0 items-center gap-2 rounded-full pr-[calc(6px*var(--ui-space-scale,1))] pl-3 hover:bg-sidebar-accent"
          >
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
            />
            <span className="min-w-0 flex-1 truncate font-mono text-ui-12 text-muted-foreground">
              {folder.path}
            </span>
            {problem ? (
              <Hint
                label={
                  problem.hint
                    ? `${problem.title}. ${problem.hint}`
                    : problem.title
                }
              >
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  strokeWidth={1.75}
                  aria-label={problem.title}
                  className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-amber-600 dark:text-amber-500"
                />
              </Hint>
            ) : null}
            <Hint label="Remove folder">
              <button
                type="button"
                onClick={() => void remove(folder)}
                aria-label={`Remove folder ${folder.path}`}
                className={cn(
                  ICON_BUTTON,
                  "opacity-0 group-hover/folder:opacity-100 focus-visible:opacity-100",
                )}
              >
                <HugeiconsIcon
                  icon={Cancel01Icon}
                  strokeWidth={1.75}
                  className="size-3"
                />
              </button>
            </Hint>
          </div>
        );
      })}
      {suggestions.length > 0 ? (
        <div className="flex flex-wrap gap-1 px-3 py-1">
          {suggestions.map((p) => (
            <Hint key={p} label={`Add ${p}`}>
              <button
                type="button"
                onClick={() => void add(p, true)}
                disabled={busy}
                className="flex h-[calc(24px*var(--ui-space-scale,1))] max-w-full items-center gap-1 rounded-full border border-dashed border-border px-2.5 font-mono text-ui-11 text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground disabled:opacity-40"
              >
                <HugeiconsIcon
                  icon={PlusSignIcon}
                  strokeWidth={1.75}
                  className="size-3 shrink-0"
                />
                <span className="truncate">{folderDisplayName(p)}</span>
              </button>
            </Hint>
          ))}
        </div>
      ) : null}
      {typing ? (
        <div className="flex flex-col gap-1 px-1 py-1">
          <div className="flex h-[calc(30px*var(--ui-space-scale,1))] items-center gap-2 rounded-full bg-sidebar-accent pr-1 pl-3">
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
            />
            <input
              value={path}
              onChange={(event) => {
                setPath(event.target.value);
                setError(null);
              }}
              onKeyDown={(event) => {
                // The list's arrow and Enter handling must not take the typed path.
                event.stopPropagation();
                if (event.key === "Enter") {
                  event.preventDefault();
                  void add(path, false);
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  setTyping(false);
                  setPath("");
                  setError(null);
                }
              }}
              placeholder="/path/to/models"
              aria-label="Folder path"
              disabled={busy}
              // biome-ignore lint/a11y/noAutofocus: opened by the user to type a path.
              autoFocus={true}
              className="h-full min-w-0 flex-1 bg-transparent font-mono text-ui-12 text-foreground outline-none placeholder:text-muted-foreground"
            />
            <button
              type="button"
              onClick={() => void add(path, false)}
              disabled={busy || !path.trim()}
              className="h-[calc(24px*var(--ui-space-scale,1))] shrink-0 rounded-full px-2.5 text-ui-12 font-medium text-foreground transition-colors hover:bg-background disabled:opacity-40"
            >
              Add
            </button>
          </div>
          {error ? (
            <p className="px-3 text-ui-12 text-destructive">{error}</p>
          ) : null}
        </div>
      ) : null}
      <FolderBrowser
        open={browsing}
        onOpenChange={setBrowsing}
        initialPath={path.trim() || undefined}
        onSelect={(picked) => {
          void add(picked, true);
        }}
      />
    </div>
  );

  return { actions, body };
}
