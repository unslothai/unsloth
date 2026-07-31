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
  const runLabel = execution?.run_name?.trim() || "Completed run";
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
      toastSuccess("Dataset link copied");
      return;
    }
    toastError("Copy failed", "Could not copy the dataset link.");
  };

  const handlePublish = async (): Promise<void> => {
    if (!execution?.jobId) {
      setPublishError("This run is missing a job id, so it cannot be published.");
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
      toastSuccess("Dataset published");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not publish this dataset.";
      setPublishError(message);
      toastError("Publish failed", message);
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
                <DialogTitle>Published</DialogTitle>
                <DialogDescription>
                  Your dataset is live on Hugging Face.
                </DialogDescription>
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-card/55 p-3 text-xs">
              <p className="mb-1 text-muted-foreground">Dataset URL</p>
              <p className="break-all font-medium text-foreground">{publishedUrl}</p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={handleCopyUrl}>
                <HugeiconsIcon icon={Copy01Icon} className="mr-2 size-4" />
                Copy link
              </Button>
              <Button asChild={true}>
                <a href={publishedUrl} target="_blank" rel="noreferrer">
                  Open repo
                  <HugeiconsIcon icon={ArrowRight01Icon} className="ml-2 size-4" />
                </a>
              </Button>
              <Button variant="ghost" onClick={() => onOpenChange(false)}>
                Done
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Publish to Hugging Face</DialogTitle>
              <DialogDescription>
                Create or update a dataset repo from this completed run.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4">
              <div className="rounded-2xl border border-border/60 bg-card/55 p-3 text-xs">
                <p className="font-medium text-foreground">From this run</p>
                <div className="mt-2 grid gap-1.5 text-muted-foreground sm:grid-cols-2">
                  <p>
                    Run: <span className="text-foreground">{runLabel}</span>
                  </p>
                  <p>
                    Records: <span className="text-foreground">{recordLabel}</span>
                  </p>
                </div>
                <p className="mt-2 text-muted-foreground">
                  We’ll upload the generated dataset, dataset card, images, and any processor
                  outputs from this execution.
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground" htmlFor="publish-repo-id">
                  Repository
                </label>
                <Input
                  id="publish-repo-id"
                  placeholder="your-name/customer-support-synth"
                  value={repoId}
                  onChange={(event) => setRepoId(event.target.value)}
                  disabled={publishing}
                />
                <p className="text-xs text-muted-foreground">
                  Use the format <span className="font-mono">username-or-org/dataset-name</span>.
                </p>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-foreground" htmlFor="publish-description">
                  About this dataset
                </label>
                <Textarea
                  id="publish-description"
                  className="corner-squircle"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  disabled={publishing}
                  rows={4}
                  placeholder={defaultDescription || "What is this dataset for?"}
                />
                <p className="text-xs text-muted-foreground">
                  This short summary is used in the dataset card on Hugging Face.
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <label className="text-sm font-medium text-foreground" htmlFor="publish-hf-token">
                    HF write token
                  </label>
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-muted-foreground underline underline-offset-3 hover:text-foreground"
                  >
                    Manage tokens
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
                    placeholder="hf_..."
                    value={hfToken}
                    onChange={(event) => setHfToken(event.target.value)}
                    disabled={publishing}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  Leave empty if you're already logged in via CLI.
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
                    Private dataset
                  </label>
                  <p className="text-xs text-muted-foreground">
                    Only people with access can view or download the repo.
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
                Cancel
              </Button>
              <Button onClick={() => void handlePublish()} disabled={!canSubmit}>
                {publishing ? "Publishing..." : "Publish to Hugging Face"}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
