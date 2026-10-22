// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useIsAccountOwner } from "@/features/auth";
import { FolderBrowser } from "@/features/model-picker";
import { pickHuggingFaceCacheDir } from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { Folder02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { setDownloadDir, useDownloadDir } from "../download-manager";
import { useScanFolders } from "../inventory/scan-folders-store";

const MODELS_FOLDER = "";
const PATH_SEPARATORS = /[\\/]+/;

function folderName(path: string): string {
  const parts = path.split(PATH_SEPARATORS).filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

export function DownloadLocationPicker() {
  const isOwner = useIsAccountOwner();
  const dir = useDownloadDir();
  const [menuOpen, setMenuOpen] = useState(false);
  const { folders } = useScanFolders(menuOpen && isOwner);
  const [browserOpen, setBrowserOpen] = useState(false);

  if (!isOwner) {
    return null;
  }

  const choose = async () => {
    if (!isTauri) {
      setBrowserOpen(true);
      return;
    }
    try {
      const path = await pickHuggingFaceCacheDir();
      if (path) {
        setDownloadDir(path);
      }
    } catch (error) {
      toast.error("Couldn't open the folder picker.", {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const options = folders.map((f) => f.path);
  if (dir && !options.includes(dir)) {
    options.unshift(dir);
  }
  const label = dir ? folderName(dir) : null;

  return (
    <>
      <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            aria-label={`Download to: ${dir ?? "Models folder"}`}
            title={`Download to: ${dir ?? "Models folder"}`}
            className="hub-tag-soft inline-flex h-[calc(26px*var(--ui-space-scale,1))] max-w-[calc(160px*var(--ui-space-scale,1))] min-w-0 cursor-pointer items-center gap-1 rounded-full border-0 px-2 text-ui-11 font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            <HugeiconsIcon
              icon={Folder02Icon}
              strokeWidth={1.75}
              className="size-3.5 shrink-0"
            />
            {dir ? <span className="truncate">{label}</span> : null}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="max-w-[calc(360px*var(--ui-space-scale,1))]"
        >
          <DropdownMenuLabel>Download to</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={dir ?? MODELS_FOLDER}
            onValueChange={(value) => setDownloadDir(value || null)}
          >
            <DropdownMenuRadioItem value={MODELS_FOLDER}>
              Models folder
            </DropdownMenuRadioItem>
            {options.map((path) => (
              <DropdownMenuRadioItem key={path} value={path} title={path}>
                <span className="truncate font-mono text-xs">{path}</span>
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={() => {
              choose().catch(() => undefined);
            }}
          >
            Choose folder...
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <FolderBrowser
        open={!isTauri && browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(path) => setDownloadDir(path)}
        initialPath={dir ?? undefined}
        title="Choose download folder"
        confirmLabel="Download here"
        showModelHints={false}
      />
    </>
  );
}
