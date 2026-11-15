// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useIsAccountOwner } from "@/features/auth";
import {
  type NativeDocumentFolderSelection,
  pickNativeDocumentFolder,
  useNativePathLeasesSupported,
} from "@/features/native-intents";
import { isTauri } from "@/lib/api-base";
import { Copy01Icon } from "@/lib/copy-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { GitBranchIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { MANAGED_FOLDER_LABEL } from "../utils/project-folder-state";

/** Live facts about a saved folder: what the row names, and why tools may not be using it. */
export type WorkingFolderDetail = {
  path?: string | null;
  branch?: string | null;
  problem?: string | null;
};

export function WorkingFolderPicker({
  value,
  onChange,
  disabled,
  needsReselect = false,
  onPendingChange,
  detail,
}: {
  value: string | null;
  onChange: (
    value: NativeDocumentFolderSelection | null,
  ) => void | Promise<void>;
  disabled?: boolean;
  needsReselect?: boolean;
  onPendingChange?: (pending: boolean) => void;
  detail?: WorkingFolderDetail;
}) {
  const isOwner = useIsAccountOwner();
  const supported = useNativePathLeasesSupported();
  const mounted = useRef(true);
  const pending = useRef(false);
  const [picking, setPicking] = useState(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const available = isTauri && supported && isOwner;

  async function change(choose: boolean) {
    if ((choose && !available) || disabled || pending.current) return;
    pending.current = true;
    setPicking(true);
    onPendingChange?.(true);
    try {
      const selected = choose ? await pickNativeDocumentFolder() : null;
      if (mounted.current && (!choose || selected)) await onChange(selected);
    } catch (error) {
      if (mounted.current) {
        toast.error("Could not change the working folder", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      pending.current = false;
      if (mounted.current) {
        setPicking(false);
        onPendingChange?.(false);
      }
    }
  }

  return (
    <section className="min-w-0 space-y-2" aria-label="Working folder">
      <h3 className="text-ui-15 font-medium text-foreground">Working folder</h3>
      <p className="text-ui-13 text-muted-foreground">
        Where code and file tools run for every chat in this project.
      </p>
      <div
        className={cn(
          "flex min-w-0 items-center gap-2 rounded-full border border-border py-2 pr-2 pl-5",
          detail?.problem && "border-amber-500/60",
        )}
        data-folder-problem={detail?.problem ? "true" : undefined}
      >
        <span
          className={cn(
            "min-w-0 flex-1 truncate text-left text-sm",
            value && detail?.path && "font-mono text-ui-13",
          )}
          title={detail?.path ?? value ?? undefined}
          // A path is cut from its start, so the folder's own name stays in view.
          dir={value && detail?.path ? "rtl" : undefined}
        >
          <bdi dir="ltr">{value ?? MANAGED_FOLDER_LABEL}</bdi>
        </span>
        {value && detail?.branch && (
          <span
            className="hidden min-w-0 shrink items-center gap-1 text-muted-foreground sm:flex"
            title={`Branch ${detail.branch}`}
          >
            <HugeiconsIcon
              icon={GitBranchIcon}
              strokeWidth={1.75}
              className="size-3 shrink-0"
            />
            <span className="max-w-28 truncate font-mono text-ui-12">
              {detail.branch}
            </span>
          </span>
        )}
        {value && detail?.path && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="shrink-0 rounded-full"
            aria-label="Copy folder path"
            title="Copy folder path"
            onClick={() => {
              const path = detail.path;
              if (!path) return;
              void copyToClipboard(path).then((copied) => {
                if (copied) toast.success("Folder path copied");
                else toast.error("Could not copy the folder path");
              });
            }}
          >
            <HugeiconsIcon
              icon={Copy01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </Button>
        )}
        {value && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="shrink-0"
            disabled={disabled || picking}
            onClick={() => void change(false)}
          >
            Use managed folder
          </Button>
        )}
        <Button
          type="button"
          variant="dark"
          size="sm"
          className="shrink-0"
          disabled={!available || disabled || picking}
          onClick={() => void change(true)}
        >
          {picking ? "Choosing…" : value ? "Change folder" : "Choose folder"}
        </Button>
      </div>
      {detail?.problem && !needsReselect && (
        <p
          className="text-ui-11 text-amber-600 dark:text-amber-400"
          role="alert"
        >
          {detail.problem}
        </p>
      )}
      {(needsReselect || !available) && (
        <p
          className="text-ui-11 text-muted-foreground"
          role={needsReselect ? "alert" : undefined}
        >
          {needsReselect
            ? "Choose the folder again to retry, or use the managed folder."
            : !isTauri
              ? "Choose an existing folder in Unsloth Desktop, or use the managed folder."
              : !isOwner
                ? "Only the owner can choose a local working folder."
                : "Folder selection is unavailable with this connection."}
        </p>
      )}
    </section>
  );
}
