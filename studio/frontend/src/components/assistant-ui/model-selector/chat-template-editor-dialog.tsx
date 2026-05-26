// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ArrowTurnBackwardIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

interface ChatTemplateEditorDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  hasOverride: boolean;
  defaultTemplate: string | null;
  draft: string;
  draftBytes: number;
  draftByteLimit: number;
  maxLength: number;
  onDraftChange: (value: string) => void;
  loading: boolean;
  draftDirty: boolean;
  draftIsDefault: boolean;
  onResetToDefault: () => void;
  onSave: () => void;
  saving: boolean;
}

export function ChatTemplateEditorDialog({
  open,
  onOpenChange,
  hasOverride,
  defaultTemplate,
  draft,
  draftBytes,
  draftByteLimit,
  maxLength,
  onDraftChange,
  loading,
  draftDirty,
  draftIsDefault,
  onResetToDefault,
  onSave,
  saving,
}: ChatTemplateEditorDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-3xl"
        overlayClassName="bg-background/35 supports-backdrop-filter:backdrop-blur-[1px]"
      >
        <DialogHeader>
          <DialogTitle>Edit Chat Template</DialogTitle>
          <DialogDescription>
            {hasOverride
              ? "Editing your saved override. The model's original template is preserved and can be restored with Reset Default."
              : defaultTemplate
                ? "Edits to this copy save as an override. The model's original template is preserved and can be restored with Reset Default."
                : "Paste a Jinja chat template. Leave empty to keep the model's built-in default."}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3 px-0.5">
            <div className="min-w-0 space-y-0.5">
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-medium text-foreground">
                  Template editor
                </span>
                {hasOverride && (
                  <span className="inline-flex h-[18px] items-center rounded-[6px] border border-amber-500/40 bg-amber-500/[0.06] px-1.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                    Custom override saved
                  </span>
                )}
              </div>
              <p className="text-[11px] text-muted-foreground">
                Jinja syntax. Saving an unchanged copy of the default clears
                any active override.
              </p>
            </div>
            <span
              className={cn(
                "shrink-0 text-[11px] tabular-nums",
                draftBytes > draftByteLimit
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
            >
              {draftBytes.toLocaleString()} /{" "}
              {draftByteLimit.toLocaleString()} bytes
            </span>
            {defaultTemplate != null && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={onResetToDefault}
                    disabled={draftIsDefault}
                    className={cn(
                      "inline-flex h-7 shrink-0 items-center gap-1 rounded-[8px] border px-2.5 text-[11.5px] font-medium transition-colors",
                      draftIsDefault
                        ? "border-border/40 text-muted-foreground/50"
                        : "border-border/60 text-foreground hover:bg-black/[0.04] dark:hover:bg-white/[0.05]",
                    )}
                  >
                    <HugeiconsIcon
                      icon={ArrowTurnBackwardIcon}
                      strokeWidth={1.75}
                      className="size-3.5"
                    />
                    Reset Default
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  side="top"
                  sideOffset={6}
                  className="tooltip-compact"
                >
                  Restore the model's original chat template
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="relative">
            <Textarea
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              maxLength={maxLength}
              fieldSizing="fixed"
              className="min-h-[24rem] max-h-[50vh] overflow-y-auto corner-squircle font-mono text-xs leading-5 focus-visible:border-input focus-visible:ring-0"
              rows={14}
              spellCheck={false}
              placeholder={
                loading
                  ? "Loading model's original chat template…"
                  : defaultTemplate
                    ? undefined
                    : "Paste your Jinja chat template here…"
              }
            />
            {loading && draft.length === 0 && (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center gap-2 rounded-[inherit] bg-background/40">
                <Spinner className="size-4 text-muted-foreground" />
                <span className="text-[12px] text-muted-foreground">
                  Reading template from model
                </span>
              </div>
            )}
          </div>
        </div>
        <DialogFooter className="flex-wrap gap-2 sm:justify-between">
          <Button
            type="button"
            variant="ghost"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button type="button" onClick={onSave} disabled={!draftDirty || saving}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
