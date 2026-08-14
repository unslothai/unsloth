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
import { useHfTokenStore } from "@/features/hub";
import { useTrainingConfigStore } from "@/features/training";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";
import { type ReactElement, type ReactNode, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import {
  BACKUP_INTERVAL_OPTIONS,
  MAX_REMOTE_CHECKPOINTS,
  backupIntervalError,
  effectiveBackupSteps,
  remoteCheckpointRetentionError,
} from "./checkpoint-backup-cadence";
import { FieldHint } from "./field-hint";
import { isRepositoryId } from "./repository-id";
import { useRepositoryAccessValidation } from "./use-repository-access-validation";

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
    <span className="inline-flex min-w-0 items-center gap-1.5 text-xs font-medium text-muted-foreground">
      {children}
      <FieldHint text={hint} label={hintLabel} />
    </span>
  );
}

type BackupSettingRowProps = {
  label: string;
  tooltip: string;
  tooltipLabel: string;
  htmlFor?: string;
  children: ReactNode;
  helper?: ReactNode;
};

function BackupSettingRow({
  label,
  tooltip,
  tooltipLabel,
  htmlFor,
  children,
  helper,
}: BackupSettingRowProps): ReactElement {
  return (
    <>
      <label htmlFor={htmlFor} className="min-w-0">
        <LabelWithHint hint={tooltip} hintLabel={tooltipLabel}>
          {label}
        </LabelWithHint>
      </label>
      <div className="flex w-full min-w-0 justify-self-end justify-end">
        {children}
      </div>
      {helper && (
        <>
          <div className="hidden md:block" aria-hidden="true" />
          <div className="min-w-0 text-xs text-muted-foreground">{helper}</div>
        </>
      )}
    </>
  );
}

export function CheckpointBackupSection(): ReactElement {
  const [backupExpanded, setBackupExpanded] = useState(false);
  const [repoTouched, setRepoTouched] = useState(false);
  const [retentionTouched, setRetentionTouched] = useState(false);
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
      : repoTouched && !isRepositoryId(store.backup.repoId ?? "")
        ? "Use the format owner/repository."
        : null;
  const retentionError = retentionTouched
    ? remoteCheckpointRetentionError(store.backup.keepRemote)
    : null;
  const effective = effectiveBackupSteps(store.saveSteps, count);
  const backupEnabled = store.backup.enabled;
  const repositoryValidation = useRepositoryAccessValidation(
    store.backup.repoId ?? "",
    hfToken ?? "",
  );
  const accessMessage = {
    idle: null,
    checking: "Checking repository…",
    ready: "Ready to upload",
    invalid_syntax: "Use the format owner/repository.",
    authentication_required: "Sign in to Hugging Face before enabling backups.",
    invalid_token: "Your saved Hugging Face token is invalid.",
    not_found: "Repository not found.",
    no_write_permission:
      "Your Hugging Face token does not have write access to this repository.",
    unavailable:
      "Could not verify repository access. Check your connection and try again.",
  }[repositoryValidation.state];
  const accessIsError = !["idle", "checking", "ready"].includes(
    repositoryValidation.state,
  );

  const setBackupEnabled = (enabled: boolean) => {
    update({ enabled });
    setBackupExpanded(enabled);
  };

  const configureSaveSteps = () => {
    const input = document.getElementById(
      "training-save-steps",
    ) as HTMLInputElement | null;
    input?.scrollIntoView({ behavior: "smooth", block: "center" });
    input?.focus({ preventScroll: true });
  };

  return (
    <Collapsible
      open={backupEnabled && backupExpanded}
      onOpenChange={(expanded) => {
        if (backupEnabled) {
          setBackupExpanded(expanded);
        }
      }}
      className="min-w-0 border-t border-border/60 pt-4"
    >
      <div className="flex w-full min-w-0 items-center justify-between gap-3">
        <div className="flex min-w-0 flex-1 items-center gap-1.5">
          <CollapsibleTrigger
            disabled={!backupEnabled}
            aria-disabled={!backupEnabled}
            aria-expanded={backupEnabled && backupExpanded}
            className={`flex min-w-0 items-center gap-2 text-left text-xs text-muted-foreground ${backupEnabled ? "cursor-pointer hover:text-foreground" : "cursor-default"}`}
          >
            {backupEnabled && (
              <HugeiconsIcon
                icon={ChevronDownStandardIcon}
                className={`size-3.5 shrink-0 transition-transform ${backupExpanded ? "rotate-180" : ""}`}
              />
            )}
            <span className="truncate">Automatic Hugging Face backups</span>
            {backupEnabled && (
              <span className="shrink-0">
                ·{" "}
                {repositoryValidation.state === "ready"
                  ? intervalLabel(count)
                  : "Setup required"}
              </span>
            )}
          </CollapsibleTrigger>
          <span className="inline-flex min-w-0 items-center gap-1.5">
            <span
              className="shrink-0"
              onClick={(event) => event.stopPropagation()}
              onPointerDown={(event) => event.stopPropagation()}
              onKeyDown={(event) => event.stopPropagation()}
            >
              <FieldHint
                label="About automatic Hugging Face backups"
                text="Automatically uploads selected training checkpoints to a Hugging Face model repository. Uploads run in the background and do not stop training if they fail."
              />
            </span>
          </span>
        </div>
        <div
          className="flex shrink-0 items-center"
          onClick={(event) => event.stopPropagation()}
          onPointerDown={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <Switch
            className="cursor-pointer"
            checked={backupEnabled}
            onCheckedChange={setBackupEnabled}
            aria-label="Automatic Hugging Face backups"
          />
        </div>
      </div>

      <CollapsibleContent className="mt-5 min-w-0">
        {store.backup.enabled && (
          <div className="grid min-w-0 grid-cols-1 items-center gap-x-6 gap-y-2 md:grid-cols-[minmax(0,1fr)_12rem]">
            <h3 className="col-span-full text-xs font-medium text-foreground">
              Destination
            </h3>
            <BackupSettingRow
              label="Repository ID"
              htmlFor="checkpoint-backup-repo"
              tooltipLabel="More information about Repository ID"
              tooltip="The Hugging Face model repository that will receive checkpoint backups. Use the format owner/repository. The repository must already exist and your saved Hugging Face token must have write access."
              helper={
                <span
                  id={
                    repoError || accessIsError
                      ? "backup-repo-error"
                      : "backup-repo-help"
                  }
                  className={
                    repoError || accessIsError ? "text-destructive" : undefined
                  }
                >
                  {repoError ??
                    accessMessage ??
                    "Example: username/my-training-backups"}
                </span>
              }
            >
              <div className="relative w-full min-w-0">
                <Input
                  id="checkpoint-backup-repo"
                  className="box-border w-full min-w-0 max-w-full pr-9"
                  value={store.backup.repoId ?? ""}
                  onChange={(event) =>
                    update({ repoId: event.target.value || null })
                  }
                  onBlur={() => {
                    if (store.backup.repoId) setRepoTouched(true);
                    repositoryValidation.validateNow();
                  }}
                  placeholder="username/unsloth-checkpoint-backups"
                  aria-invalid={!!repoError || accessIsError}
                  aria-describedby={
                    repoError || accessIsError
                      ? "backup-repo-error"
                      : "backup-repo-help"
                  }
                />
                <span
                  className="pointer-events-none absolute inset-y-0 right-2 flex items-center"
                  aria-hidden="true"
                >
                  {repositoryValidation.state === "checking" ? (
                    <LoaderCircle className="size-4 animate-spin text-muted-foreground" />
                  ) : repositoryValidation.state === "ready" ? (
                    <CircleCheck className="size-4 text-emerald-600" />
                  ) : accessIsError ? (
                    <CircleAlert className="size-4 text-destructive" />
                  ) : null}
                </span>
              </div>
            </BackupSettingRow>
            <output className="sr-only" aria-live="polite">
              {accessMessage}
            </output>

            <div className="col-span-full my-3 border-t border-border/60" />
            <h3 className="col-span-full text-xs font-medium text-foreground">
              Backup schedule
            </h3>
            <BackupSettingRow
              label="Upload backup every"
              tooltipLabel="More information about backup frequency"
              tooltip="Controls how often completed local checkpoints are uploaded. The local checkpoint frequency is configured by Save Steps above. Final uploads can still run when periodic backups are off."
              helper={
                store.saveSteps <= 0 ? (
                  <div className="flex flex-wrap items-center gap-2">
                    <span>
                      Periodic uploads are unavailable because Save Steps is 0.
                    </span>
                    <button
                      type="button"
                      className="text-xs font-medium text-primary underline-offset-4 hover:underline"
                      onClick={configureSaveSteps}
                    >
                      Configure Save Steps
                    </button>
                  </div>
                ) : countError ? (
                  <span role="alert" className="text-destructive">
                    {countError}
                  </span>
                ) : (
                  <span>Effective upload interval: {effective} steps</span>
                )
              }
            >
              <div className="grid w-full min-w-0 gap-1.5">
                <Select
                  disabled={store.saveSteps <= 0}
                  value={isPreset ? String(count) : "custom"}
                  onValueChange={(value) =>
                    value !== "custom" &&
                    update({ intervalCheckpoints: Number(value) })
                  }
                >
                  <SelectTrigger
                    className="w-full min-w-0"
                    aria-label="Upload backup every"
                    aria-disabled={store.saveSteps <= 0}
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
                {!isPreset && store.saveSteps > 0 && (
                  <span className="flex items-center justify-end gap-2 text-xs text-muted-foreground">
                    Upload every
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
                    />
                    checkpoints
                  </span>
                )}
              </div>
            </BackupSettingRow>

            <div className="col-span-full my-3 border-t border-border/60" />
            <h3 className="col-span-full text-xs font-medium text-foreground">
              Retention and final uploads
            </h3>
            <BackupSettingRow
              label="Remote checkpoints to keep"
              htmlFor="checkpoint-backup-retention"
              tooltipLabel="More information about remote checkpoint retention"
              tooltip="Keeps this many recent checkpoint folders in the Hugging Face repository. It does not change local checkpoint retention. Older files may remain in Git/LFS history."
              helper={
                retentionError ? (
                  <span role="alert" className="text-destructive">
                    {retentionError}
                  </span>
                ) : null
              }
            >
              <Input
                id="checkpoint-backup-retention"
                className="w-24"
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
            </BackupSettingRow>
            <BackupSettingRow
              label="Upload when training stops"
              tooltipLabel="More information about uploading when training stops"
              tooltip="Uploads the newest complete checkpoint when training is stopped normally. A forced runtime shutdown may not allow the upload to finish."
            >
              <Switch
                aria-label="Upload when training stops"
                checked={store.backup.uploadOnStop}
                onCheckedChange={(value) => update({ uploadOnStop: value })}
              />
            </BackupSettingRow>
            <BackupSettingRow
              label="Upload when training completes"
              tooltipLabel="More information about uploading when training completes"
              tooltip="Uploads the final complete checkpoint after training finishes successfully."
            >
              <Switch
                aria-label="Upload when training completes"
                checked={store.backup.uploadOnComplete}
                onCheckedChange={(value) => update({ uploadOnComplete: value })}
              />
            </BackupSettingRow>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
