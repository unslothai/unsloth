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
import {
  FOLDER_CARD_CLASS,
  FOLDER_ROW_CLASS,
} from "@/features/rag/components/linked-folders-manager";
import { Folder02Icon } from "@/lib/folder-icons";
import { Alert02Icon, GitBranchIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { getProjectFolderAccess } from "../api/chat-api";
import { MANAGED_FOLDER_LABEL } from "../utils/project-folder-state";

const FolderBrowser = lazy(() =>
  import(
    "@/features/model-picker/components/model-selector/folder-browser"
  ).then((module) => ({ default: module.FolderBrowser })),
);

export type ServerFolderSelection = { serverPath: string; displayName: string };
export type WorkingFolderSelection =
  | NativeDocumentFolderSelection
  | ServerFolderSelection;

let serverPickAllowed: Promise<boolean> | null = null;

function useServerFolderPick(enabled: boolean): boolean {
  const [allowed, setAllowed] = useState(false);
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    serverPickAllowed ??= getProjectFolderAccess().then(
      (access) => access.serverPath,
      () => {
        serverPickAllowed = null;
        return false;
      },
    );
    void serverPickAllowed.then((value) => {
      if (live) setAllowed(value);
    });
    return () => {
      live = false;
    };
  }, [enabled]);
  return enabled && allowed;
}

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
  onChange: (value: WorkingFolderSelection | null) => void | Promise<void>;
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
  const native = isTauri && supported && isOwner;
  const browse = useServerFolderPick(!isTauri && isOwner);
  const available = native || browse;
  const [browsing, setBrowsing] = useState(false);

  async function change(choose: boolean, picked?: ServerFolderSelection) {
    if ((choose && !available) || disabled || pending.current) return;
    if (choose && !native && !picked) {
      setBrowsing(true);
      return;
    }
    pending.current = true;
    setPicking(true);
    onPendingChange?.(true);
    try {
      const selected = !choose
        ? null
        : (picked ?? (await pickNativeDocumentFolder()));
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

  const chosen = Boolean(value);
  const problem = detail?.problem && !needsReselect ? detail.problem : null;

  return (
    <section className="min-w-0 space-y-2" aria-label="Working folder">
      <h3 className="text-ui-15 font-medium text-foreground">Working folder</h3>
      <p className="text-ui-13 text-muted-foreground">
        Where code and file tools run for every chat in this project.
      </p>
      <div
        className={cn(FOLDER_CARD_CLASS, problem && "border-amber-500/60")}
        data-folder-problem={problem ? "true" : undefined}
      >
        <div className={FOLDER_ROW_CLASS}>
          <HugeiconsIcon
            icon={problem ? Alert02Icon : Folder02Icon}
            strokeWidth={1.75}
            className={cn(
              "size-4 shrink-0",
              problem
                ? "text-amber-600 dark:text-amber-400"
                : "text-muted-foreground",
            )}
          />
          <div className="flex min-w-0 flex-1 flex-col">
            <span
              className={cn(
                "truncate text-left text-sm",
                chosen && detail?.path && "font-mono text-ui-13",
              )}
              title={detail?.path ?? value ?? undefined}
              dir={chosen && detail?.path ? "rtl" : undefined}
            >
              <bdi dir="ltr">{value ?? MANAGED_FOLDER_LABEL}</bdi>
            </span>
            {problem ? (
              <span
                className="text-ui-12 text-amber-600 dark:text-amber-400"
                role="alert"
              >
                {problem}
              </span>
            ) : (
              <span className="truncate text-ui-12 text-muted-foreground">
                {chosen
                  ? "Tools read and write files here."
                  : "Unsloth keeps this project's files in a folder of its own."}
              </span>
            )}
          </div>
          {chosen && detail?.branch && (
            <span
              className="hidden max-w-32 min-w-0 shrink-0 items-center gap-1 text-muted-foreground sm:flex"
              title={`Branch ${detail.branch}`}
            >
              <HugeiconsIcon
                icon={GitBranchIcon}
                strokeWidth={1.75}
                className="size-3 shrink-0"
              />
              <span className="truncate font-mono text-ui-12">
                {detail.branch}
              </span>
            </span>
          )}
          {chosen && detail?.path && (
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
        </div>
        <div
          className={cn(FOLDER_ROW_CLASS, "flex-wrap justify-end gap-2 py-2")}
        >
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
      </div>
      {(needsReselect || !available) && (
        <p
          className="px-1 text-ui-11 text-muted-foreground"
          role={needsReselect ? "alert" : undefined}
        >
          {needsReselect
            ? "Choose the folder again to retry, or use the managed folder."
            : !isTauri
              ? isOwner
                ? "Choosing a folder here works on this computer with remote access off. Otherwise use Unsloth Desktop, or the managed folder."
                : "Only the owner can choose a local working folder."
              : !isOwner
                ? "Only the owner can choose a local working folder."
                : "Folder selection is unavailable with this connection."}
        </p>
      )}
      {browsing && (
        <Suspense fallback={null}>
          <FolderBrowser
            open={true}
            onOpenChange={setBrowsing}
            onSelect={(path) =>
              void change(true, {
                serverPath: path,
                displayName:
                  path
                    .replace(/[\\/]+$/, "")
                    .split(/[\\/]/)
                    .pop() || path,
              })
            }
            initialPath={(value && detail?.path) || undefined}
            title="Choose a working folder"
            description="Code and file tools will read and write in the folder you choose"
            confirmLabel="Use this folder"
            showModelHints={false}
          />
        </Suspense>
      )}
    </section>
  );
}
