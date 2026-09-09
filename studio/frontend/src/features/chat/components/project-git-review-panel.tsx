// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

import {
  PROJECT_GIT_DIFF_MODES,
  type ProjectGitDiffFile,
  type ProjectGitDiffLine,
  type ProjectGitDiffManifest,
  type ProjectGitDiffMode,
  ProjectGitReviewRequestGuard,
  type ProjectGitStatus,
  getProjectGitDiff,
  getProjectGitStatus,
} from "../api/project-git-review-api";
import type { ProjectRecord } from "../types";

const MODE_LABELS: Record<ProjectGitDiffMode, string> = {
  head: "All changes",
  staged: "Staged",
  unstaged: "Unstaged",
};

function requestError(error: unknown): string {
  return error instanceof Error ? error.message : "Git review request failed.";
}

function lineMarker(line: ProjectGitDiffLine): string {
  if (line.kind === "add") {
    return "+";
  }
  if (line.kind === "delete") {
    return "-";
  }
  return " ";
}

function lineClass(line: ProjectGitDiffLine): string {
  if (line.kind === "add") {
    return "bg-emerald-500/10 text-emerald-800 dark:text-emerald-300";
  }
  if (line.kind === "delete") {
    return "bg-red-500/10 text-red-800 dark:text-red-300";
  }
  return "text-foreground";
}

export function GitDiffFile({ file }: { file: ProjectGitDiffFile }) {
  const metadata = [
    file.binary ? "binary" : null,
    file.encoding === "invalid-utf8" ? "invalid UTF-8" : null,
    file.symlink ? "symbolic link" : null,
    file.submodule ? "submodule" : null,
    file.modeChanged ? `${file.oldMode} to ${file.newMode}` : null,
    file.scopeBoundary ? "project boundary change" : null,
    file.truncated ? "bounded preview unavailable" : null,
  ].filter(Boolean);

  return (
    <details className="rounded-lg border border-border bg-background/60">
      <summary className="cursor-pointer px-3 py-2 text-xs">
        <span className="mr-2 font-mono text-muted-foreground">
          {file.code}
        </span>
        <code dir="ltr" style={{ unicodeBidi: "isolate" }}>
          {file.oldPath ? `${file.oldPath} -> ${file.path}` : file.path}
        </code>
        <span className="ml-2 text-emerald-700 dark:text-emerald-300">
          +{file.additions}
        </span>
        <span className="ml-1 text-red-700 dark:text-red-300">
          -{file.deletions}
        </span>
      </summary>
      <div className="border-t border-border px-3 py-2">
        {file.wholeFileOnly ? (
          <p className="mb-2 text-xs text-muted-foreground">
            {metadata.length > 0 ? `${metadata.join(", ")}. ` : null}
            Content hunks are not exposed for this file.
          </p>
        ) : null}
        {file.hunks.map((hunk) => (
          <details key={hunk.id} open={true} className="mb-2 last:mb-0">
            <summary className="cursor-pointer font-mono text-[11px] text-muted-foreground">
              {hunk.header}
            </summary>
            <div className="mt-1 overflow-x-auto rounded border border-border font-mono text-[11px] leading-5">
              {hunk.lines.map((line, index) => (
                <div
                  key={`${hunk.id}:${index}`}
                  className={`grid min-w-max grid-cols-[3rem_3rem_1rem_auto] ${lineClass(line)}`}
                >
                  <span className="select-none px-1 text-right text-muted-foreground">
                    {line.oldLine ?? ""}
                  </span>
                  <span className="select-none px-1 text-right text-muted-foreground">
                    {line.newLine ?? ""}
                  </span>
                  <span className="select-none">{lineMarker(line)}</span>
                  <pre
                    dir="ltr"
                    className="pr-3"
                    style={{ unicodeBidi: "isolate" }}
                  >
                    {line.text}
                  </pre>
                </div>
              ))}
            </div>
          </details>
        ))}
      </div>
    </details>
  );
}

export function ProjectGitReviewPanel({
  project,
}: {
  project: ProjectRecord & {
    workspaceRevision?: number;
    workspaceAvailable?: boolean;
  };
}) {
  const [mode, setMode] = useState<ProjectGitDiffMode>("head");
  const [status, setStatus] = useState<ProjectGitStatus | null>(null);
  const [diff, setDiff] = useState<ProjectGitDiffManifest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const guard = useRef(new ProjectGitReviewRequestGuard());
  const workspaceRevision = project.workspaceRevision ?? 0;

  const refresh = useCallback(async () => {
    const token = guard.current.begin(project.id, workspaceRevision, mode);
    setStatus(null);
    setDiff(null);
    setLoading(true);
    setError(null);
    try {
      const [nextStatus, nextDiff] = await Promise.all([
        getProjectGitStatus(project.id, workspaceRevision),
        getProjectGitDiff(project.id, workspaceRevision, mode),
      ]);
      if (!guard.current.accepts(token, nextStatus, nextDiff)) {
        return;
      }
      setStatus(nextStatus);
      setDiff(nextDiff);
    } catch (nextError) {
      if (guard.current.acceptsToken(token)) {
        setError(requestError(nextError));
      }
    } finally {
      if (guard.current.acceptsToken(token)) {
        setLoading(false);
      }
    }
  }, [mode, project.id, workspaceRevision]);

  useEffect(() => {
    const requestGuard = guard.current;
    requestGuard.activate();
    const initialRefresh = window.setTimeout(() => {
      refresh().catch(() => undefined);
    }, 0);
    return () => {
      window.clearTimeout(initialRefresh);
      requestGuard.retire();
    };
  }, [refresh]);

  if (project.workspaceAvailable === false) {
    return (
      <div className="mt-3 rounded-xl border border-border bg-background/60 px-3 py-3">
        <p className="text-sm font-medium">Git review</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Reconnect the project folder to inspect its Git state.
        </p>
      </div>
    );
  }

  const blocked = [
    ...(status?.blockedReasons ?? []),
    ...(diff?.blockedReasons ?? []),
  ];

  return (
    <div className="mt-3 rounded-xl border border-border bg-background/60 px-3 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium">Git review</p>
          <p className="text-xs text-muted-foreground">
            Read-only status and hunk review for the primary project workspace.
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={loading}
          onClick={() => refresh().catch(() => undefined)}
        >
          {loading ? "Reading..." : "Refresh"}
        </Button>
      </div>

      <fieldset className="mt-3 flex flex-wrap gap-2">
        <legend className="sr-only">Git diff scope</legend>
        {PROJECT_GIT_DIFF_MODES.map((value) => (
          <Button
            key={value}
            type="button"
            size="sm"
            variant={mode === value ? "secondary" : "ghost"}
            disabled={loading}
            aria-pressed={mode === value}
            onClick={() => setMode(value)}
          >
            {MODE_LABELS[value]}
          </Button>
        ))}
      </fieldset>

      {error ? (
        <p role="alert" className="mt-3 text-xs text-destructive">
          {error}
        </p>
      ) : null}
      {blocked.length > 0 ? (
        <p
          role="alert"
          className="mt-3 text-xs text-amber-700 dark:text-amber-300"
        >
          Review blocked: {Array.from(new Set(blocked)).join(", ")}. Refresh
          after the repository is stable.
        </p>
      ) : null}
      {status ? (
        <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span>{status.branch ?? "Detached HEAD"}</span>
          <span>{status.counts.staged} staged</span>
          <span>{status.counts.unstaged} unstaged</span>
          <span>{status.counts.untracked} untracked</span>
          <span>{status.counts.conflicted} conflicted</span>
        </div>
      ) : null}
      {diff ? (
        <div className="mt-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            {diff.fileCount} files, {diff.hunkCount} expandable hunks. Snapshot{" "}
            {diff.sourceFingerprint.slice(0, 12)}.
          </p>
          {diff.truncated ? (
            <p className="text-xs text-amber-700 dark:text-amber-300">
              The bounded review limit was reached. No partial snapshot is
              selectable.
            </p>
          ) : null}
          {diff.files.map((file) => (
            <GitDiffFile key={file.id} file={file} />
          ))}
          {!loading && diff.files.length === 0 && blocked.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No changes in this scope.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
