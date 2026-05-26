// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { deleteFineTunedModel } from "@/features/chat";
import { notifyInventoryEntryDeleted } from "@/features/models/delete-notifications";
import {
  emitTrainingRunDeleted,
  emitTrainingRunsChanged,
} from "@/features/training";
import { useId, useState } from "react";
import { toast } from "sonner";

export interface DeleteFineTuneTarget {
  id: string;
  displayName: string;
  source: "training" | "exported";
  exportType?: "lora" | "merged" | "gguf";
  ggufVariant?: string;
}

interface DeleteFineTuneDialogProps {
  target: DeleteFineTuneTarget | null;
  onOpenChange: (open: boolean) => void;
  onDeleted: (target: DeleteFineTuneTarget, deletedRunIds: string[]) => void;
}

export function DeleteFineTuneDialog({
  target,
  onOpenChange,
  onDeleted,
}: DeleteFineTuneDialogProps) {
  const targetKey = target?.id ?? "";
  const [deleteRunRecordState, setDeleteRunRecordState] = useState({
    targetKey,
    value: false,
  });
  const deleteRunRecord =
    deleteRunRecordState.targetKey === targetKey
      ? deleteRunRecordState.value
      : false;
  const [inFlight, setInFlight] = useState(false);
  const deleteRunRecordId = useId();
  const isGgufTarget = target?.exportType === "gguf" && !!target.ggufVariant;

  async function confirm() {
    if (!target) return;
    const alsoDeleteRunRecord =
      target.source === "training" && deleteRunRecord;
    setInFlight(true);
    try {
      const result = await deleteFineTunedModel({
        modelPath: target.id,
        source: target.source,
        exportType: target.exportType,
        ggufVariant: target.ggufVariant,
        deleteRunRecord: alsoDeleteRunRecord,
      });
      const deletedRunIds = result.deleted_run_ids ?? [];
      notifyInventoryEntryDeleted({
        kind: "model",
        id: target.id,
        ggufVariant: target.ggufVariant,
      });
      for (const runId of deletedRunIds) {
        emitTrainingRunDeleted(runId);
      }
      emitTrainingRunsChanged();
      onDeleted(target, deletedRunIds);
    } catch (err) {
      toast.error("Failed to delete model", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setInFlight(false);
    }
  }

  return (
    <Dialog
      open={target !== null}
      onOpenChange={(open) => {
        if (!open && inFlight) return;
        onOpenChange(open);
      }}
    >
      <DialogContent className="menu-flat-destructive corner-squircle border border-border/60 bg-background/98 shadow-none sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {isGgufTarget ? "Delete GGUF quantization" : "Delete fine-tuned model"}
          </DialogTitle>
          <DialogDescription>
            {target && isGgufTarget ? (
              <>
                Delete <em>{target.ggufVariant}</em> for{" "}
                <em>{target.displayName}</em>? Other quantizations stay on disk.
              </>
            ) : target ? (
              <>
                Delete the model files for <em>{target.displayName}</em>? The
                model will disappear from the chat picker.
              </>
            ) : null}
          </DialogDescription>
        </DialogHeader>
        {target?.source === "training" && (
          <label
            htmlFor={deleteRunRecordId}
            className="mt-1 flex cursor-pointer items-start gap-2.5 rounded-md border border-border/60 bg-muted/40 p-3 text-xs leading-relaxed"
          >
            <Checkbox
              id={deleteRunRecordId}
              checked={deleteRunRecord}
              onCheckedChange={(checked) =>
                setDeleteRunRecordState({
                  targetKey,
                  value: checked === true,
                })
              }
              disabled={inFlight}
              className="mt-0.5"
            />
            <span className="flex flex-col gap-0.5">
              <span className="font-medium text-foreground">
                Also delete the training run from recents
              </span>
              <span className="text-muted-foreground">
                Removes the run and its stats from history. Leave unchecked to
                keep the stats. Chat and Continue training will be disabled
                for it.
              </span>
            </span>
          </label>
        )}
        <DialogFooter className="flex-wrap gap-2 sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            disabled={inFlight}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            disabled={inFlight}
            onClick={() => void confirm()}
          >
            {inFlight ? "Deleting..." : "Delete files"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
