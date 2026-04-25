// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState, type ReactElement } from "react";
import { ArrowRight01Icon, CheckmarkCircle02Icon, Copy01Icon, Key01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/features/i18n";
import { toastError, toastSuccess } from "@/shared/toast";
import type { RecipeExecutionRecord } from "../../execution-types";
import { copyTextToClipboard } from "../../executions/execution-helpers";

type PublishExecutionDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  execution: RecipeExecutionRecord | null;
  onPublish: (payload: {
    repo_id: string;
    description: string;
    hf_token?: string | null;
    private: boolean;
    artifact_path?: string | null;
  }) => Promise<{ url: string }>;
};

function getExecutionRecordCount(execution: RecipeExecutionRecord | null): number | null {
  if (!execution) {
    return null;
  }
  if (typeof execution.analysis?.num_records === "number") {
    return execution.analysis.num_records;
  }
  if (execution.datasetTotal > 0) {
    return execution.datasetTotal;
  }
  if (execution.rows > 0) {
    return execution.rows;
  }
  return null;
}

function buildDefaultDescription(execution: RecipeExecutionRecord | null): string {
  if (!execution) {
    return "";
  }
  const runName = execution.run_name?.trim() || "This dataset";
  const records = getExecutionRecordCount(execution);
  const recordPart =
    typeof records === "number" && records > 0
      ? ` It contains ${records.toLocaleString()} generated records.`
      : "";
  return `${runName} was generated with Unsloth Recipe Studio.${recordPart}`;
}

export function PublishExecutionDialog({
  open,
  onOpenChange,
  execution,
  onPublish,
}: PublishExecutionDialogProps): ReactElement {
  const { t } = useI18n();
  const [repoId, setRepoId] = useState("");
  const [description, setDescription] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [privateRepo, setPrivateRepo] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [publishError, setPublishError] = useState<string | null>(null);
  const [publishedUrl, setPublishedUrl] = useState<string | null>(null);

  const defaultDescription = useMemo(
    () => buildDefaultDescription(execution),
    [execution],
  );
  const runLabel = execution?.run_name?.trim() || t("recipe.publish.completedRun");
  const recordCount = getExecutionRecordCount(execution);
  const recordLabel =
    typeof recordCount === "number" ? recordCount.toLocaleString() : "--";

  useEffect(() => {
    if (!open) {
      setPublishing(false);
      setPublishError(null);
      setPublishedUrl(null);
      setRepoId("");
      setDescription("");
      setHfToken("");
      setPrivateRepo(false);
      return;
    }
    setPublishError(null);
    setPublishedUrl(null);
    setDescription(buildDefaultDescription(execution));
  }, [execution, open]);

  const canSubmit =
    !publishing &&
    Boolean(execution?.jobId) &&
    Boolean(execution?.artifact_path) &&
    repoId.trim().length > 0 &&
    description.trim().length > 0;

  const handleCopyUrl = async (): Promise<void> => {
    if (!publishedUrl) {
      return;
    }
    const ok = await copyTextToClipboard(publishedUrl);
    if (ok) {
      toastSuccess(t("recipe.publish.toast.linkCopied"));
      return;
    }
    toastError(
      t("recipe.publish.toast.copyFailedTitle"),
      t("recipe.publish.toast.copyFailedDescription"),
    );
  };

  const handlePublish = async (): Promise<void> => {
    if (!execution?.jobId) {
      setPublishError(t("recipe.publish.error.missingJobId"));
      return;
    }
    setPublishing(true);
    setPublishError(null);
    try {
      const result = await onPublish({
        repo_id: repoId.trim(),
        description: description.trim(),
        hf_token: hfToken.trim() || null,
        private: privateRepo,
        artifact_path: execution.artifact_path,
      });
      setPublishedUrl(result.url);
      toastSuccess(t("recipe.publish.toast.published"));
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : t("recipe.publish.error.couldNotPublish");
      setPublishError(message);
      toastError(t("recipe.publish.toast.publishFailedTitle"), message);
    } finally {
      setPublishing(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (publishing) {
          return;
        }
        onOpenChange(nextOpen);
      }}
    >
      <DialogContent
        className="sm:max-w-xl"
        overlayClassName="bg-black/55"
        onInteractOutside={(event) => {
          if (publishing) {
            event.preventDefault();
          }
        }}
      >
        {publishedUrl ? (
          <>
            <div className="flex flex-col items-center gap-3 py-4">
              <div className="flex size-12 items-center justify-center rounded-full bg-emerald-500/10">
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  className="size-6 text-emerald-600 dark:text-emerald-400"
                />
              </div>
              <div className="space-y-1 text-center">
                <DialogTitle>{t("recipe.publish.publishedTitle")}</DialogTitle>
                <DialogDescription>
                  {t("recipe.publish.publishedDescription")}
                </DialogDescription>
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-card/55 p-3 text-xs">
              <p className="mb-1 text-muted-foreground">{t("recipe.publish.datasetUrl")}</p>
              <p className="break-all font-medium text-foreground">{publishedUrl}</p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={handleCopyUrl}>
                <HugeiconsIcon icon={Copy01Icon} className="mr-2 size-4" />
                {t("recipe.publish.copyLink")}
              </Button>
              <Button asChild={true}>
                <a href={publishedUrl} target="_blank" rel="noreferrer">
                  {t("recipe.publish.openRepo")}
                  <HugeiconsIcon icon={ArrowRight01Icon} className="ml-2 size-4" />
                </a>
              </Button>
              <Button variant="ghost" onClick={() => onOpenChange(false)}>
                {t("recipe.progress.done")}
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>{t("recipe.publish.title")}</DialogTitle>
              <DialogDescription>
                {t("recipe.publish.description")}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <div className="rounded-2xl border border-border/60 bg-card/55 p-3 text-xs">
                <p className="font-medium text-foreground">{t("recipe.publish.fromRun")}</p>
                <div className="mt-2 grid gap-1.5 text-muted-foreground sm:grid-cols-2">
                  <p>
                    {t("recipe.publish.run")}: <span className="text-foreground">{runLabel}</span>
                  </p>
                  <p>
                    {t("recipe.publish.records")}: <span className="text-foreground">{recordLabel}</span>
                  </p>
                </div>
                <p className="mt-2 text-muted-foreground">
                  {t("recipe.publish.fromRunHint")}
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground" htmlFor="publish-repo-id">
                  {t("recipe.publish.repository")}
                </label>
                <Input
                  id="publish-repo-id"
                  placeholder={t("recipe.publish.repositoryPlaceholder")}
                  value={repoId}
                  onChange={(event) => setRepoId(event.target.value)}
                  disabled={publishing}
                />
                <p className="text-xs text-muted-foreground">
                  {t("recipe.publish.repositoryHint")}{" "}
                  <span className="font-mono">username-or-org/dataset-name</span>.
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground" htmlFor="publish-description">
                  {t("recipe.publish.aboutDataset")}
                </label>
                <Textarea
                  id="publish-description"
                  className="corner-squircle"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  disabled={publishing}
                  rows={4}
                  placeholder={
                    defaultDescription || t("recipe.publish.aboutDatasetPlaceholder")
                  }
                />
                <p className="text-xs text-muted-foreground">
                  {t("recipe.publish.aboutDatasetHint")}
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <label className="text-sm font-medium text-foreground" htmlFor="publish-hf-token">
                    {t("recipe.publish.hfWriteToken")}
                  </label>
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-muted-foreground underline underline-offset-3 hover:text-foreground"
                  >
                    {t("recipe.publish.manageTokens")}
                  </a>
                </div>
                <div className="relative">
                  <HugeiconsIcon
                    icon={Key01Icon}
                    className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
                  />
                  <Input
                    id="publish-hf-token"
                    type="password"
                    autoComplete="new-password"
                    className="pl-9"
                    placeholder={t("recipe.publish.hfTokenPlaceholder")}
                    value={hfToken}
                    onChange={(event) => setHfToken(event.target.value)}
                    disabled={publishing}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {t("recipe.publish.hfTokenHint")}
                </p>
              </div>

              <div className="corner-squircle flex items-start gap-3 rounded-2xl border border-border/60 bg-card/35 p-3">
                <Switch
                  id="publish-private"
                  size="sm"
                  checked={privateRepo}
                  onCheckedChange={setPrivateRepo}
                  disabled={publishing}
                />
                <div className="space-y-1">
                  <label
                    htmlFor="publish-private"
                    className="text-sm font-medium text-foreground"
                  >
                    {t("recipe.publish.privateDataset")}
                  </label>
                  <p className="text-xs text-muted-foreground">
                    {t("recipe.publish.privateDatasetHint")}
                  </p>
                </div>
              </div>

              {publishError ? (
                <div className="rounded-2xl border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                  {publishError}
                </div>
              ) : null}
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={publishing}
              >
                {t("recipe.run.action.cancel")}
              </Button>
              <Button onClick={() => void handlePublish()} disabled={!canSubmit}>
                {publishing ? t("recipe.publish.publishing") : t("recipe.publish.title")}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
