// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  pickNativeDocumentFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  FolderAddIcon,
  FolderSyncIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type Dispatch,
  type SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";
import { FOLDER_CARD_CLASS, FOLDER_ROW_CLASS } from "./linked-folders-manager";
import { EXPIRY_GRACE_MS } from "./staged-source";
import {
  isFolderExpired,
  type StagedFolder,
  stageFolder,
} from "./link-staged-folders";

export function ProjectFolderPicker({
  folders,
  onChange,
  disabled = false,
  onPendingChange,
}: {
  folders: StagedFolder[];
  onChange: Dispatch<SetStateAction<StagedFolder[]>>;
  disabled?: boolean;
  onPendingChange?: (pending: boolean) => void;
}) {
  const leasesSupported = useNativePathLeasesSupported();
  const supported = isTauri && leasesSupported;
  const [picking, setPicking] = useState(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Leases are short-lived: drop folders before they expire.
  useEffect(() => {
    if (folders.length === 0) return;
    const soonest = Math.min(...folders.map((f) => f.expiresAtMs));
    const timer = setTimeout(
      () => {
        const expired = folders.filter((f) => isFolderExpired(f, Date.now()));
        if (expired.length === 0) return;
        onChange((current) => current.filter((f) => !expired.includes(f)));
        toast.info(
          expired.length === 1
            ? "A picked folder expired"
            : `${expired.length} picked folders expired`,
          { description: "Add them again to link them." },
        );
      },
      Math.max(0, soonest - EXPIRY_GRACE_MS - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [folders, onChange]);

  async function add() {
    if (!supported || picking || disabled) return;
    setPicking(true);
    onPendingChange?.(true);
    try {
      const selected = await pickNativeDocumentFolder();
      if (!selected || !mounted.current) return;
      onChange((current) => [...current, stageFolder(selected)]);
    } catch (error) {
      toast.error("Could not add folder", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (mounted.current) {
        setPicking(false);
        onPendingChange?.(false);
      }
    }
  }

  return (
    <div className="space-y-2.5">
      <p className="text-ui-15 font-medium text-foreground">Linked folders</p>
      <div className={FOLDER_CARD_CLASS}>
        {folders.map((folder) => (
          <div key={folder.token} className={FOLDER_ROW_CLASS}>
            <HugeiconsIcon
              icon={FolderSyncIcon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
            <span className="min-w-0 flex-1 truncate text-sm">
              {folder.displayName}
            </span>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="shrink-0 rounded-full"
              disabled={disabled}
              aria-label={`Remove ${folder.displayName}`}
              onClick={() =>
                onChange((current) =>
                  current.filter((f) => f.token !== folder.token),
                )
              }
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            </Button>
          </div>
        ))}
        <button
          type="button"
          disabled={!supported || picking || disabled}
          onClick={() => void add()}
          aria-label="Add folder"
          title={
            supported
              ? "Choose a local folder"
              : "Requires the managed desktop backend"
          }
          className={cn(
            FOLDER_ROW_CLASS,
            "w-full cursor-pointer py-3 text-left text-sm transition-colors hover:bg-muted/60 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent",
          )}
        >
          {picking ? (
            <Spinner className="size-4 shrink-0" />
          ) : (
            <HugeiconsIcon
              icon={FolderAddIcon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
          )}
          <span>Add folder</span>
        </button>
      </div>
      <p className="px-1 text-ui-11 text-muted-foreground">
        {supported
          ? "Indexed and kept in sync, for every chat in this project to search."
          : "Linking a folder needs the managed desktop backend."}
      </p>
    </div>
  );
}
