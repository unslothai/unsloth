// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TrainIcon } from "@/components/icons/train-icon";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { deleteCachedModel, type LocalModelInfo } from "@/features/chat";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  Alert02Icon,
  Delete02Icon,
  PencilEdit02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { toast } from "sonner";
import { DeleteConfirmDialog } from "./download-card";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";

interface LocalOnDeviceCardProps {
  repoId: string | null;
  sourceLabel: string;
  source: LocalModelInfo["source"];
  path: string;
  isGguf: boolean;
  isActive: boolean;
  isLoading: boolean;
  loadingPhase?: "downloading" | "starting";
  unsupportedReason?: string | null;
  onLoad: () => void;
  onUseInChat: () => void;
  onTrain?: () => void;
  onChange?: () => void;
}

export function LocalOnDeviceCard({
  repoId,
  sourceLabel,
  source,
  path,
  isGguf,
  isActive,
  isLoading,
  loadingPhase,
  unsupportedReason,
  onLoad,
  onUseInChat,
  onTrain,
  onChange,
}: LocalOnDeviceCardProps) {
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);

  const canDelete =
    source === "hf_cache" && !!repoId && !isActive && !isLoading;

  async function handleDelete() {
    if (!repoId) return;
    setDeleting(true);
    try {
      await deleteCachedModel(repoId, undefined, hfToken || undefined);
      toast.success(`Deleted ${repoId}`);
      setDeleteOpen(false);
      onChange?.();
    } catch (err) {
      toast.error("Failed to delete model", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setDeleting(false);
    }
  }

  const formatLabel = isGguf ? "GGUF" : "Safetensors";
  const showOldCacheHint = source === "hf_cache" && !!unsupportedReason;

  return (
    <div className="flex w-full flex-col gap-2">
      {showOldCacheHint && (
        <div className="flex items-start gap-2 rounded-[12px] border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-[12px] leading-5 text-amber-700 dark:text-amber-300">
          <HugeiconsIcon
            icon={Alert02Icon}
            strokeWidth={1.75}
            className="mt-[1px] size-3.5 shrink-0"
          />
          <span>
            This looks like an older Hub cache. It may use a format that Unsloth
            no longer loads (missing config or unsupported quantization). You
            can still keep it on disk, or delete it to free space.
          </span>
        </div>
      )}
      <div className="download-card">
        <div className="group/dl flex items-center">
          <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
            <span className="flex min-w-0 items-center gap-1.5 text-[12px] text-muted-foreground">
              <DotTag
                tone="success"
                label={isActive ? "Loaded" : "On device"}
              />
              <DotTag
                tone={isGguf ? "gguf" : "checkpoint"}
                label={formatLabel}
              />
              {source !== "hf_cache" && (
                <span className="truncate text-muted-foreground/85">
                  {sourceLabel}
                </span>
              )}
            </span>
            <div className="ml-auto flex items-center gap-0.5">
              {canDelete && (
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <button
                      type="button"
                      aria-label={`Delete ${repoId}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteOpen(true);
                      }}
                      className="inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] text-muted-foreground opacity-0 transition-[opacity,background-color,color] duration-150 hover:bg-rose-500/10 hover:text-rose-600 focus-visible:opacity-100 group-hover/dl:opacity-100 dark:hover:bg-rose-500/15 dark:hover:text-rose-400"
                    >
                      <HugeiconsIcon
                        icon={Delete02Icon}
                        strokeWidth={1.75}
                        className="size-4"
                      />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={4}>
                    Delete from device
                  </TooltipContent>
                </Tooltip>
              )}
              <PathInfoButton
                path={path}
                title={sourceLabel}
                description="Where this model lives on disk."
              />
            </div>
          </div>
          {onTrain && (
            <div
              aria-hidden="true"
              className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] opacity-100 transition-opacity duration-150 group-hover/dl:opacity-0 dark:bg-white/[0.04]"
            />
          )}
          <div className="group/pair flex h-9 shrink-0 items-stretch gap-1.5">
            {onTrain && (
              <button
                type="button"
                onClick={() => onTrain()}
                className="hub-action-btn w-24"
              >
                <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
                Train
              </button>
            )}
            <button
              type="button"
              disabled={isLoading}
              onClick={() => {
                if (isActive) {
                  onUseInChat();
                  return;
                }
                onLoad();
              }}
              className={cn(
                isLoading || isActive
                  ? "hub-action-btn w-24"
                  : "run-action-btn w-24",
                isLoading && "opacity-70",
              )}
            >
              {isLoading ? (
                <>
                  <Spinner />
                  {loadingPhase === "downloading" ? "Preparing…" : "Loading…"}
                </>
              ) : isActive ? (
                <>
                  <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} />
                  Chat
                </>
              ) : (
                <>
                  <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} />
                  Run
                </>
              )}
            </button>
          </div>
        </div>
      </div>
      <DeleteConfirmDialog
        open={deleteOpen}
        onOpenChange={(o) => {
          if (!o && !deleting) setDeleteOpen(false);
        }}
        title="Delete cached model?"
        deleting={deleting}
        onConfirm={() => void handleDelete()}
        description={
          <>
            This will remove{" "}
            <span className="font-medium text-foreground">{repoId}</span> and
            its downloaded files from disk. You can re-download it later.
          </>
        }
      />
    </div>
  );
}
