// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
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
import {
  BACKUP_INTERVAL_OPTIONS,
  backupIntervalError,
  effectiveBackupSteps,
} from "./checkpoint-backup-cadence";

const REPO_ID_PATTERN = /^[\w.-]+\/[\w.-]+$/;

const intervalLabel = (value: number) =>
  value === 1 ? "Every checkpoint" : `Every ${value} checkpoints`;

export function CheckpointBackupSection(): ReactElement {
  const [open, setOpen] = useState(false);
  const [repoTouched, setRepoTouched] = useState(false);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      saveSteps: state.saveSteps,
      backup: state.checkpointBackup,
      setBackup: state.setCheckpointBackup,
    })),
  );
  const update = (patch: Partial<CheckpointBackupConfig>) =>
    store.setBackup({ ...store.backup, ...patch });
  const count = store.backup.intervalCheckpoints;
  const isPreset = BACKUP_INTERVAL_OPTIONS.includes(
    count as (typeof BACKUP_INTERVAL_OPTIONS)[number],
  );
  const countError = backupIntervalError(count);
  const repoError =
    repoTouched && !store.backup.repoId
      ? "Enter a Hugging Face repository ID."
      : repoTouched && !REPO_ID_PATTERN.test(store.backup.repoId ?? "")
        ? "Use the format owner/repository."
        : null;
  const effective = effectiveBackupSteps(store.saveSteps, count);

  const configureSaveSteps = () => {
    const input = document.getElementById(
      "training-save-steps",
    ) as HTMLInputElement | null;
    input?.scrollIntoView({ behavior: "smooth", block: "center" });
    input?.focus({ preventScroll: true });
  };

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="border-t border-border/60 pt-4"
    >
      <div className="flex items-center gap-3">
        <CollapsibleTrigger className="flex flex-1 cursor-pointer items-center gap-1.5 text-left text-xs text-muted-foreground">
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          />
          <span>Automatic Hugging Face backups</span>
          {store.backup.enabled && !open && (
            <span className="ml-1 flex min-w-0 text-muted-foreground/80">
              <span className="shrink-0">· {intervalLabel(count)}</span>
              {store.backup.repoId && (
                <span className="truncate" title={store.backup.repoId}>
                  &nbsp;· {store.backup.repoId}
                </span>
              )}
            </span>
          )}
        </CollapsibleTrigger>
        <Switch
          checked={store.backup.enabled}
          onCheckedChange={(enabled) => update({ enabled })}
          aria-label="Automatic Hugging Face backups"
        />
      </div>

      <CollapsibleContent className="mt-4 space-y-4">
        {store.backup.enabled && (
          <>
            <div className="grid gap-1.5 text-xs font-medium text-muted-foreground">
              Repository
              <Input
                value={store.backup.repoId ?? ""}
                onChange={(event) =>
                  update({ repoId: event.target.value || null })
                }
                onBlur={() => setRepoTouched(true)}
                placeholder="username/unsloth-checkpoint-backups"
                aria-invalid={!!repoError}
                aria-describedby={
                  repoError ? "backup-repo-error" : "backup-repo-help"
                }
              />
              <span
                id={repoError ? "backup-repo-error" : "backup-repo-help"}
                className={
                  repoError ? "text-destructive" : "text-muted-foreground"
                }
              >
                {repoError ?? "Example: username/my-training-backups"}
              </span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={!store.backup.repoId || !!repoError}
              onClick={() => setRepoTouched(true)}
            >
              Test access
            </Button>

            <div className="grid max-w-sm gap-1.5 text-xs font-medium text-muted-foreground">
              Upload backup every
              <Select
                value={isPreset ? String(count) : "custom"}
                onValueChange={(value) =>
                  value !== "custom" &&
                  update({ intervalCheckpoints: Number(value) })
                }
              >
                <SelectTrigger aria-label="Upload backup every">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BACKUP_INTERVAL_OPTIONS.map((value) => (
                    <SelectItem key={value} value={String(value)}>
                      <span className="flex w-full justify-between gap-6">
                        <span>{intervalLabel(value)}</span>
                        <span className="text-muted-foreground">
                          {store.saveSteps * value} steps
                        </span>
                      </span>
                    </SelectItem>
                  ))}
                  <SelectItem value="custom">Custom…</SelectItem>
                </SelectContent>
              </Select>
              {!isPreset && (
                <span className="flex items-center gap-2">
                  Upload every{" "}
                  <Input
                    className="h-8 w-20"
                    type="number"
                    min={1}
                    max={1000}
                    step={1}
                    value={count}
                    onChange={(event) =>
                      update({
                        intervalCheckpoints: Number(event.target.value),
                      })
                    }
                    aria-label="Custom backup checkpoint count"
                    aria-invalid={!!countError}
                    aria-describedby={
                      countError ? "backup-count-error" : undefined
                    }
                  />{" "}
                  checkpoints
                </span>
              )}
              {countError && (
                <span
                  id="backup-count-error"
                  role="alert"
                  className="text-destructive"
                >
                  {countError}
                </span>
              )}
            </div>

            {store.saveSteps <= 0 ? (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Periodic backups require checkpoint saving. Set Save Steps above
                0.{" "}
                <Button
                  type="button"
                  variant="link"
                  size="xs"
                  onClick={configureSaveSteps}
                >
                  Configure Save Steps
                </Button>
              </p>
            ) : (
              !countError && (
                <p className="text-xs text-muted-foreground">
                  Effective upload interval: {effective} steps. Backups upload
                  at steps {effective}, {effective * 2}, {effective * 3}, …
                </p>
              )
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="flex items-center justify-between gap-3 text-xs">
                Keep backups{" "}
                <Input
                  className="w-20"
                  type="number"
                  min={1}
                  value={store.backup.keepRemote}
                  onChange={(event) =>
                    update({ keepRemote: Number(event.target.value) })
                  }
                  aria-label="Remote backups to keep"
                />
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                Upload when stopped{" "}
                <Switch
                  checked={store.backup.uploadOnStop}
                  onCheckedChange={(value) => update({ uploadOnStop: value })}
                />
              </div>
              <div className="flex items-center justify-between gap-3 text-xs">
                Upload when completed{" "}
                <Switch
                  checked={store.backup.uploadOnComplete}
                  onCheckedChange={(value) =>
                    update({ uploadOnComplete: value })
                  }
                />
              </div>
            </div>
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
