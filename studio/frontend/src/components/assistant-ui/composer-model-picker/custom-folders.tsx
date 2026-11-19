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
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useEffect, useState } from "react";

const ICON_BUTTON =
  "grid size-[calc(20px*var(--ui-space-scale,1))] shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground max-md:size-8";

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
    async (raw: string) => {
      const trimmed = raw.trim();
      if (!trimmed || busy) {
        return;
      }
      setBusy(true);
      try {
        const { folder, alreadyAdded } = await addScanFolderToStore(trimmed);
        if (alreadyAdded) {
          toast.info("Already in your folders", {
            description: folderDisplayName(folder.path),
          });
        }
        onFoldersChange?.();
      } catch (e) {
        toast.error("Couldn't add folder", {
          description: e instanceof Error ? e.message : "Failed to add folder",
        });
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
      void add(picked);
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

  // One add control: the empty state's row, or this once a folder exists.
  const actions =
    folders.length === 0 ? null : (
      <Hint label="Add a folder to scan for models">
        <button
          type="button"
          aria-label="Add a custom folder"
          onClick={() => void browse()}
          disabled={busy}
          className={cn(ICON_BUTTON, "disabled:opacity-40")}
        >
          <HugeiconsIcon
            icon={PlusSignIcon}
            strokeWidth={1.75}
            className="size-3"
          />
        </button>
      </Hint>
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
            className="group/folder flex h-[calc(28px*var(--ui-space-scale,1))] shrink-0 max-md:h-9 items-center gap-2 rounded-full pr-[calc(6px*var(--ui-space-scale,1))] pl-3 hover:bg-sidebar-accent"
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
                  "opacity-0 group-hover/folder:opacity-100 focus-visible:opacity-100 max-md:opacity-100",
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
                onClick={() => void add(p)}
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
      {folders.length === 0 ? (
        <button
          type="button"
          onClick={() => void browse()}
          disabled={busy}
          className="mx-1 flex h-[calc(30px*var(--ui-space-scale,1))] shrink-0 items-center gap-2 rounded-full border border-dashed border-border pl-3 text-left text-ui-12 text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-foreground disabled:opacity-40"
        >
          <HugeiconsIcon
            icon={PlusSignIcon}
            strokeWidth={1.75}
            className="size-[calc(14px*var(--ui-space-scale,1))] shrink-0"
          />
          Add a folder to scan for models
        </button>
      ) : null}
      <FolderBrowser
        open={browsing}
        onOpenChange={setBrowsing}
        acceptUnbrowsable={true}
        onSelect={(picked) => {
          void add(picked);
        }}
      />
    </div>
  );

  return { actions, body };
}
