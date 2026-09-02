// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const OPEN_PROJECT_FOLDER_ENDPOINT = "/api/chat/projects/open-folder";

type ConsumeProjectFolderToken = (
  token: string,
  operation: "open-project" | "set-project-workspace",
) => Promise<{ nativePathLease: string }>;

function required(value: string, message: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error(message);
  return trimmed;
}

export async function buildOpenProjectFolderRequest(
  nativePathToken: string,
  name: string,
  consume: ConsumeProjectFolderToken,
): Promise<{ input: string; init: RequestInit }> {
  const token = required(
    nativePathToken,
    "A native project-folder token is required.",
  );
  const projectName = required(name, "Project name is required.");
  const { nativePathLease } = await consume(token, "set-project-workspace");
  return {
    input: OPEN_PROJECT_FOLDER_ENDPOINT,
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nativePathLease, name: projectName }),
    },
  };
}

export async function buildChangeProjectFolderRequest(
  projectId: string,
  nativePathToken: string,
  expectedWorkspaceRevision: number,
  consume: ConsumeProjectFolderToken,
): Promise<{ input: string; init: RequestInit }> {
  const token = required(
    nativePathToken,
    "A native project-folder token is required.",
  );
  const { nativePathLease } = await consume(token, "set-project-workspace");
  return {
    input: `/api/chat/projects/${encodeURIComponent(projectId)}/workspace-folder`,
    init: {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nativePathLease,
        expectedWorkspaceRevision,
      }),
    },
  };
}

export function buildDisconnectProjectFolderRequest(
  projectId: string,
  expectedWorkspaceRevision: number,
): { input: string; init: RequestInit } {
  return {
    input: `/api/chat/projects/${encodeURIComponent(projectId)}/workspace-folder`,
    init: {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expectedWorkspaceRevision }),
    },
  };
}
