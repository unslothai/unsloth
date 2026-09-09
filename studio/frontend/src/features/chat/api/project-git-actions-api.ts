// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";
import type { ProjectGitDiffFile } from "./project-git-review-api";

export interface GitCheckpoint {
  id: string;
  commitSha: string;
  ownedPaths: string[];
  sourceFingerprint: string;
}
export interface GitWorktree {
  id: string;
  branch: string;
  baseRef: string;
  status: string;
  merge?: {
    status: string;
    conflicts?: string[];
    resultHead?: string;
    primaryWorkspaceChanged?: boolean;
  };
}
export interface PreparedCommit {
  id: string;
  status: string;
  branch: string;
  baseHead: string;
  message: string;
  ownedPaths: string[];
  confirmationToken?: string;
  commitSha?: string;
  refName: string;
  previewFiles?: ProjectGitDiffFile[];
}
export interface GitManagementState {
  projectId: string;
  workspaceRevision: number;
  mutationsAvailable: boolean;
  head: string | null;
  fingerprint: string | null;
  unavailableReason?: string;
  checkpoints: GitCheckpoint[];
  worktrees: GitWorktree[];
  preparedCommits: PreparedCommit[];
}
export interface GitHubPreview {
  id: string;
  confirmationToken: string;
  requestDigest: string;
  expiresAt: number;
  connector: { id: string; displayName: string };
  request: {
    owner: string;
    repo: string;
    base: string;
    head: string;
    title: string;
    body: string;
    draft?: boolean;
  };
  reviewBinding: { head: string; branch: string; workspaceFingerprint: string };
}
export interface GitHubRequest {
  serverId: string;
  owner: string;
  repository: string;
  base: string;
  head: string;
  title: string;
  bodyNote: string;
  draft: boolean;
}

export async function gitAction<T>(
  projectId: string,
  revision: number,
  endpoint: string,
  method = "GET",
  payload?: unknown,
): Promise<T> {
  const query = new URLSearchParams({ workspaceRevision: String(revision) });
  const response = await authFetch(
    `/api/agent/projects/${encodeURIComponent(projectId)}/${endpoint}?${query}`,
    {
      method,
      cache: "no-store",
      headers:
        payload === undefined
          ? undefined
          : { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    },
  );
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      formatApiErrorBody(body) ?? `Git request failed (${response.status})`,
    );
  return body as T;
}

export async function gitConnectors(): Promise<
  { id: string; display_name: string; is_enabled: boolean }[]
> {
  const response = await authFetch("/api/mcp/servers/", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load connected tools.");
  return response.json();
}

/** Confirmation authority is usable only for the exact project and form that produced it. */
export function gitPreviewKey(
  projectId: string,
  revision: number,
  value: unknown,
): string {
  return JSON.stringify([projectId, revision, value]);
}
