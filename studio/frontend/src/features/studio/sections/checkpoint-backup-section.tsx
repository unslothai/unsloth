// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useTrainingConfigStore } from "@/features/training";
import type { CheckpointBackupConfig } from "@/features/training/types/config";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";

function cadenceError(saveSteps: number, intervalSteps: number): string | null {
  if (saveSteps <= 0) {
    return "Automatic interval backups require local checkpoint saves.";
  }
  if (intervalSteps <= 0 || intervalSteps % saveSteps !== 0) {
    return `Backup interval must be a multiple of local checkpoint save steps. With save_steps=${saveSteps}, use ${saveSteps}, ${saveSteps * 2}, ${saveSteps * 3}, ...`;
  }
  return null;
}

export function CheckpointBackupSection(): ReactElement {
  const [open, setOpen] = useState(false);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      saveSteps: state.saveSteps,
      backup: state.checkpointBackup,
      setSaveSteps: state.setSaveSteps,
      setBackup: state.setCheckpointBackup,
    })),
  );
  const update = (patch: Partial<CheckpointBackupConfig>) =>
    store.setBackup({ ...store.backup, ...patch });
  const multiplier =
    store.saveSteps > 0 && store.backup.intervalSteps % store.saveSteps === 0
      ? store.backup.intervalSteps / store.saveSteps
      : null;
  const error = store.backup.enabled
    ? cadenceError(store.saveSteps, store.backup.intervalSteps)
    : null;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
        Automatic Hugging Face checkpoint backup
        {store.backup.enabled && (
          <span className="ml-1 rounded-full bg-primary/10 px-2 py-0.5 text-primary">
            Enabled
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-4 space-y-4 rounded-xl border border-border/60 p-4">
        <label className="flex items-center justify-between gap-4 text-sm font-medium">
          Enable automatic Hugging Face backups
          <Switch
            checked={store.backup.enabled}
            onCheckedChange={(enabled) => update({ enabled })}
            aria-label="Enable automatic Hugging Face backups"
          />
        </label>

        {store.backup.enabled && (
          <>
            <label className="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Repository ID
              <Input
                value={store.backup.repoId ?? ""}
                onChange={(event) =>
                  update({ repoId: event.target.value || null })
                }
                placeholder="username/unsloth-checkpoint-backups"
                aria-invalid={!store.backup.repoId}
              />
              {!store.backup.repoId && (
                <span className="text-destructive">
                  Repository ID is required.
                </span>
              )}
            </label>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="grid gap-1.5 text-xs font-medium text-muted-foreground">
                Local checkpoint save interval
                <Input
                  type="number"
                  min={0}
                  step={1}
                  value={store.saveSteps}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    store.setSaveSteps(next);
                    if (multiplier && next > 0) {
                      update({ intervalSteps: next * multiplier });
                    }
                  }}
                />
              </label>
              <label className="grid gap-1.5 text-xs font-medium text-muted-foreground">
                Backup upload interval
                <Select
                  value={
                    multiplier && multiplier <= 3
                      ? String(multiplier)
                      : "custom"
                  }
                  onValueChange={(value) => {
                    if (value !== "custom" && store.saveSteps > 0) {
                      update({
                        intervalSteps: store.saveSteps * Number(value),
                      });
                    }
                  }}
                >
                  <SelectTrigger aria-label="Backup upload interval">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[1, 2, 3].map((value) => (
                      <SelectItem key={value} value={String(value)}>
                        {value === 1
                          ? "Every checkpoint"
                          : `Every ${value} checkpoints`}{" "}
                        — {store.saveSteps * value} steps
                      </SelectItem>
                    ))}
                    <SelectItem value="custom">Custom…</SelectItem>
                  </SelectContent>
                </Select>
                {(multiplier === null || multiplier > 3) && (
                  <Input
                    type="number"
                    min={1}
                    step={1}
                    value={store.backup.intervalSteps}
                    onChange={(event) =>
                      update({ intervalSteps: Number(event.target.value) })
                    }
                    aria-invalid={!!error}
                    aria-label="Custom backup interval steps"
                  />
                )}
              </label>
            </div>

            {error ? (
              <p className="text-xs text-destructive" role="alert">
                {error}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                Local checkpoints will be saved every {store.saveSteps} steps.
                Hugging Face backup will run every {store.backup.intervalSteps}{" "}
                steps: {store.backup.intervalSteps},{" "}
                {store.backup.intervalSteps * 2},{" "}
                {store.backup.intervalSteps * 3}, ...
              </p>
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="flex items-center justify-between gap-3 text-xs">
                Private repository
                <Switch
                  checked={store.backup.private}
                  onCheckedChange={(value) => update({ private: value })}
                  aria-label="Private backup repository"
                />
              </label>
              <label className="flex items-center gap-3 text-xs">
                Remote checkpoints to keep
                <Input
                  className="w-20"
                  type="number"
                  min={1}
                  value={store.backup.keepRemote}
                  onChange={(event) =>
                    update({ keepRemote: Number(event.target.value) })
                  }
                />
              </label>
              <label className="flex items-center justify-between gap-3 text-xs">
                Upload when stopped
                <Switch
                  checked={store.backup.uploadOnStop}
                  onCheckedChange={(value) => update({ uploadOnStop: value })}
                  aria-label="Upload checkpoint when training is stopped"
                />
              </label>
              <label className="flex items-center justify-between gap-3 text-xs">
                Upload when completed
                <Switch
                  checked={store.backup.uploadOnComplete}
                  onCheckedChange={(value) =>
                    update({ uploadOnComplete: value })
                  }
                  aria-label="Upload checkpoint when training completes"
                />
              </label>
            </div>
            <p className="text-xs text-muted-foreground">
              Backups run asynchronously. A hard runtime termination cannot
              guarantee the final upload.
            </p>
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
