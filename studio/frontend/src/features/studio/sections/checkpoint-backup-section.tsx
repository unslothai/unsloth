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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { hfApiToken, hubTokenHeader, useHfTokenStore } from "@/features/hub";
import { useTrainingConfigStore } from "@/features/training";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  BACKUP_INTERVAL_OPTIONS,
  MAX_REMOTE_CHECKPOINTS,
  backupIntervalError,
  effectiveBackupSteps,
  remoteCheckpointRetentionError,
} from "./checkpoint-backup-cadence";
import { FieldHint } from "./field-hint";

const REPO_ID_PATTERN = /^[\w.-]+\/[\w.-]+$/;

type AccessState =
  | "Not checked"
  | "Checking…"
  | "Ready to upload"
  | "Repository not found"
  | "Authentication required"
  | "No write permission";

const intervalLabel = (value: number) =>
  value === 1 ? "Every checkpoint" : `Every ${value} checkpoints`;

function LabelWithHint({
  children,
  hint,
  hintLabel,
}: {
  children: string;
  hint: string;
  hintLabel: string;
}) {
  return (
    <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
      {children}
      <FieldHint text={hint} label={hintLabel} />
    </span>
  );
}

export function CheckpointBackupSection(): ReactElement {
  const [open, setOpen] = useState(false);
  const [repoTouched, setRepoTouched] = useState(false);
  const [retentionTouched, setRetentionTouched] = useState(false);
  const [accessState, setAccessState] = useState<AccessState>("Not checked");
  const hfToken = useHfTokenStore((state) => state.token);
  const store = useTrainingConfigStore(
    useShallow((state) => ({
      saveSteps: state.saveSteps,
      backup: state.checkpointBackup,
      setBackup: state.setCheckpointBackup,
    })),
  );
  const update = (patch: Partial<typeof store.backup>) =>
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
  const retentionError = retentionTouched
    ? remoteCheckpointRetentionError(store.backup.keepRemote)
    : null;
  const effective = effectiveBackupSteps(store.saveSteps, count);
  const canTest = !!store.backup.repoId && !repoError;

  const configureSaveSteps = () => {
    const input = document.getElementById(
      "training-save-steps",
    ) as HTMLInputElement | null;
    input?.scrollIntoView({ behavior: "smooth", block: "center" });
    input?.focus({ preventScroll: true });
  };

  const testAccess = async () => {
    setRepoTouched(true);
    if (!(store.backup.repoId && REPO_ID_PATTERN.test(store.backup.repoId))) {
      return;
    }
    setAccessState("Checking…");
    try {
      const response = await fetch(
        `https://huggingface.co/api/models/${encodeURIComponent(store.backup.repoId)}`,
        { headers: hubTokenHeader(hfApiToken(hfToken) ?? null) },
      );
      setAccessState(
        response.ok
          ? "Ready to upload"
          : response.status === 401
            ? "Authentication required"
            : response.status === 403
              ? "No write permission"
              : "Repository not found",
      );
    } catch {
      setAccessState("Repository not found");
    }
  };

  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="min-w-0 border-t border-border/60 pt-4"
    >
      <div className="flex min-w-0 items-center gap-3">
        <CollapsibleTrigger className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 text-left text-xs text-muted-foreground">
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={`size-3.5 shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          />
          <span>Automatic Hugging Face backups</span>
        </CollapsibleTrigger>
        <FieldHint
          label="More information about automatic Hugging Face backups"
          text="Automatically uploads selected training checkpoints to a Hugging Face model repository. Uploads run in the background and do not stop training if they fail."
        />
        <Switch
          checked={store.backup.enabled}
          onCheckedChange={(enabled) => update({ enabled })}
          aria-label="Automatic Hugging Face backups"
        />
      </div>

      <CollapsibleContent className="mt-5 min-w-0 space-y-5">
        {store.backup.enabled && (
          <>
            <section className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-3">
              <h3 className="text-xs font-medium text-foreground">
                Destination
              </h3>
              <label
                htmlFor="checkpoint-backup-repo"
                className="grid min-w-0 gap-1.5"
              >
                <LabelWithHint
                  hintLabel="More information about Repository ID"
                  hint="The Hugging Face model repository that will receive checkpoint backups. Use the format owner/repository. The repository must already exist and your saved Hugging Face token must have write access."
                >
                  Repository ID
                </LabelWithHint>
                <Input
                  id="checkpoint-backup-repo"
                  className="box-border w-full min-w-0 max-w-full"
                  value={store.backup.repoId ?? ""}
                  onChange={(event) => {
                    update({ repoId: event.target.value || null });
                    setAccessState("Not checked");
                  }}
                  onBlur={() => setRepoTouched(true)}
                  placeholder="username/unsloth-checkpoint-backups"
                  aria-invalid={!!repoError}
                  aria-describedby={
                    repoError ? "backup-repo-error" : "backup-repo-help"
                  }
                />
                <span
                  id={repoError ? "backup-repo-error" : "backup-repo-help"}
                  className={`text-xs ${repoError ? "text-destructive" : "text-muted-foreground"}`}
                >
                  {repoError ?? "Example: username/my-training-backups"}
                </span>
              </label>
              <div className="flex flex-wrap items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <span tabIndex={canTest ? undefined : 0}>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        disabled={!canTest || accessState === "Checking…"}
                        onClick={testAccess}
                      >
                        Test access
                      </Button>
                    </span>
                  </TooltipTrigger>
                  {!canTest && (
                    <TooltipContent>
                      Enter a repository ID before testing access.
                    </TooltipContent>
                  )}
                </Tooltip>
                <output className="text-xs text-muted-foreground">
                  {accessState}
                </output>
              </div>
            </section>

            <section className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-3 border-t border-border/60 pt-5">
              <h3 className="text-xs font-medium text-foreground">
                Backup schedule
              </h3>
              <div className="grid max-w-sm min-w-0 gap-1.5">
                <LabelWithHint
                  hintLabel="More information about backup frequency"
                  hint="Controls how often completed local checkpoints are uploaded. The local checkpoint frequency is configured by Save Steps above. Final uploads can still run when periodic backups are off."
                >
                  Upload backup every
                </LabelWithHint>
                <div className="flex min-w-0 flex-wrap items-center gap-2">
                  <Select
                    disabled={store.saveSteps <= 0}
                    value={isPreset ? String(count) : "custom"}
                    onValueChange={(value) =>
                      value !== "custom" &&
                      update({ intervalCheckpoints: Number(value) })
                    }
                  >
                    <SelectTrigger
                      className="min-w-0 flex-1"
                      aria-label="Upload backup every"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {BACKUP_INTERVAL_OPTIONS.map((value) => (
                        <SelectItem key={value} value={String(value)}>
                          <span className="flex w-full justify-between gap-6">
                            <span>{intervalLabel(value)}</span>
                            {store.saveSteps > 0 && (
                              <span className="text-muted-foreground">
                                Every {store.saveSteps * value} steps
                              </span>
                            )}
                          </span>
                        </SelectItem>
                      ))}
                      <SelectItem value="custom">Custom…</SelectItem>
                    </SelectContent>
                  </Select>
                  {store.saveSteps <= 0 && (
                    <span className="text-xs text-muted-foreground">
                      Disabled
                    </span>
                  )}
                </div>
                {!isPreset && store.saveSteps > 0 && (
                  <span className="flex items-center gap-2 text-xs text-muted-foreground">
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
                    />{" "}
                    checkpoints
                  </span>
                )}
                {countError && (
                  <span role="alert" className="text-xs text-destructive">
                    {countError}
                  </span>
                )}
                {store.saveSteps <= 0 ? (
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span>
                      Periodic backups are off because Save Steps is 0.
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      size="xs"
                      onClick={configureSaveSteps}
                    >
                      Configure Save Steps
                    </Button>
                  </div>
                ) : (
                  !countError && (
                    <p className="text-xs text-muted-foreground">
                      Effective upload interval: {effective} steps
                    </p>
                  )
                )}
              </div>
            </section>

            <section className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-3 border-t border-border/60 pt-5">
              <h3 className="text-xs font-medium text-foreground">
                Retention and final uploads
              </h3>
              <label
                htmlFor="checkpoint-backup-retention"
                className="grid max-w-sm gap-1.5"
              >
                <LabelWithHint
                  hintLabel="More information about remote checkpoint retention"
                  hint="Keeps this many recent checkpoint folders in the Hugging Face repository. It does not change local checkpoint retention. Older files may remain in Git/LFS history."
                >
                  Remote checkpoints to keep
                </LabelWithHint>
                <Input
                  id="checkpoint-backup-retention"
                  className="w-20"
                  type="number"
                  min={1}
                  max={MAX_REMOTE_CHECKPOINTS}
                  step={1}
                  value={store.backup.keepRemote}
                  onBlur={() => setRetentionTouched(true)}
                  onChange={(event) =>
                    update({ keepRemote: Number(event.target.value) })
                  }
                  aria-invalid={!!retentionError}
                  aria-label="Remote checkpoints to keep"
                />
                {retentionError && (
                  <span role="alert" className="text-xs text-destructive">
                    {retentionError}
                  </span>
                )}
              </label>
              <div className="grid max-w-lg gap-3">
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                  <LabelWithHint
                    hintLabel="More information about uploading when training stops"
                    hint="Uploads the newest complete checkpoint when training is stopped normally. A forced runtime shutdown may not allow the upload to finish."
                  >
                    Upload when training stops
                  </LabelWithHint>
                  <Switch
                    aria-label="Upload when training stops"
                    checked={store.backup.uploadOnStop}
                    onCheckedChange={(value) => update({ uploadOnStop: value })}
                  />
                </div>
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                  <LabelWithHint
                    hintLabel="More information about uploading when training completes"
                    hint="Uploads the final complete checkpoint after training finishes successfully."
                  >
                    Upload when training completes
                  </LabelWithHint>
                  <Switch
                    aria-label="Upload when training completes"
                    checked={store.backup.uploadOnComplete}
                    onCheckedChange={(value) =>
                      update({ uploadOnComplete: value })
                    }
                  />
                </div>
              </div>
            </section>
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
