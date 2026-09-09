// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export const PROJECT_GIT_DIFF_MODES = ["head", "staged", "unstaged"] as const;
export type ProjectGitDiffMode = (typeof PROJECT_GIT_DIFF_MODES)[number];

export interface ProjectGitStatusFile {
  id: string;
  code: string;
  indexStatus: string;
  worktreeStatus: string;
  path: string;
  pathEncoding: "utf-8" | "escaped";
  oldPath: string | null;
  oldPathEncoding: "utf-8" | "escaped" | null;
  conflicted: boolean;
}

export interface ProjectGitStatus {
  version: number;
  projectId: string;
  target: { kind: "primary" };
  workspaceRevision: number;
  head: string | null;
  branch: string | null;
  sourceFingerprint: string;
  fingerprintComplete: boolean;
  coherent: boolean;
  blockedReasons: string[];
  counts: {
    staged: number;
    unstaged: number;
    untracked: number;
    conflicted: number;
  };
  files: ProjectGitStatusFile[];
}

export interface ProjectGitDiffLine {
  kind: "context" | "add" | "delete";
  text: string;
  oldLine: number | null;
  newLine: number | null;
  noNewline?: boolean;
}

export interface ProjectGitDiffHunk {
  id: string;
  header: string;
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: ProjectGitDiffLine[];
}

export interface ProjectGitDiffFile {
  id: string;
  code: string;
  path: string;
  pathEncoding: "utf-8" | "escaped";
  oldPath: string | null;
  oldPathEncoding: "utf-8" | "escaped" | null;
  oldMode: string;
  newMode: string;
  oldBlob: string;
  newBlob: string;
  binary: boolean;
  encoding: "utf-8" | "invalid-utf8" | "unavailable";
  symlink: boolean;
  submodule: boolean;
  modeChanged: boolean;
  scopeBoundary?: boolean;
  wholeFileOnly: boolean;
  truncated: boolean;
  byteSize?: number | null;
  unavailableReason?: string | null;
  additions: number;
  deletions: number;
  hunks: ProjectGitDiffHunk[];
}

export interface ProjectGitDiffManifest {
  version: number;
  projectId: string;
  target: { kind: "primary" };
  workspaceRevision: number;
  mode: ProjectGitDiffMode;
  head: string | null;
  sourceFingerprint: string;
  fingerprintComplete: boolean;
  selectable: boolean;
  blockedReasons: string[];
  conflictedPaths: string[];
  files: ProjectGitDiffFile[];
  fileCount: number;
  hunkCount: number;
  lineCount: number;
  truncated: boolean;
  limits: {
    maxBytes: number;
    maxFiles: number;
    maxHunks: number;
    maxLines: number;
    maxLineChars: number;
    maxUntrackedFileBytes: number;
  };
}

function gitPath(projectId: string, endpoint: "status" | "diff"): string {
  return `/api/agent/projects/${encodeURIComponent(projectId)}/git/${endpoint}`;
}

async function request<T>(input: string): Promise<T> {
  const response = await authFetch(input, { cache: "no-store" });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      formatApiErrorBody(body) ??
        `Project Git review request failed (${response.status})`,
    );
  }
  return body as T;
}

export function getProjectGitStatus(
  projectId: string,
  workspaceRevision: number,
): Promise<ProjectGitStatus> {
  const query = new URLSearchParams({
    workspaceRevision: String(workspaceRevision),
  });
  return request(`${gitPath(projectId, "status")}?${query}`);
}

export function getProjectGitDiff(
  projectId: string,
  workspaceRevision: number,
  mode: ProjectGitDiffMode,
): Promise<ProjectGitDiffManifest> {
  const query = new URLSearchParams({
    workspaceRevision: String(workspaceRevision),
    mode,
  });
  return request(`${gitPath(projectId, "diff")}?${query}`);
}

export interface ProjectGitReviewRequestToken {
  revision: number;
  projectId: string;
  workspaceRevision: number;
  mode: ProjectGitDiffMode;
}

/** Reject stale results after refresh, project changes, mode changes, or unmount. */
export class ProjectGitReviewRequestGuard {
  private revision = 0;
  private mounted = false;

  activate(): void {
    this.mounted = true;
    this.revision += 1;
  }

  begin(
    projectId: string,
    workspaceRevision: number,
    mode: ProjectGitDiffMode,
  ): ProjectGitReviewRequestToken {
    this.revision += 1;
    return { revision: this.revision, projectId, workspaceRevision, mode };
  }

  accepts(
    token: ProjectGitReviewRequestToken,
    status: ProjectGitStatus,
    diff: ProjectGitDiffManifest,
  ): boolean {
    return (
      this.acceptsToken(token) &&
      status.projectId === token.projectId &&
      diff.projectId === token.projectId &&
      status.workspaceRevision === token.workspaceRevision &&
      diff.workspaceRevision === token.workspaceRevision &&
      diff.mode === token.mode &&
      status.target.kind === "primary" &&
      diff.target.kind === "primary"
    );
  }

  acceptsToken(token: ProjectGitReviewRequestToken): boolean {
    return this.mounted && token.revision === this.revision;
  }

  currentRevision(): number {
    return this.revision;
  }

  retire(): void {
    this.mounted = false;
    this.revision += 1;
  }
}
