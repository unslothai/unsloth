// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const OPEN_PROJECT_FOLDER_ENDPOINT = "/api/chat/projects/open-folder";

export function buildOpenProjectFolderRequest(
  nativePathLease: string,
  name: string,
): { input: string; init: RequestInit } {
  const lease = nativePathLease.trim();
  const projectName = name.trim();
  if (!lease) {
    throw new Error("A native project-folder lease is required.");
  }
  if (!projectName) {
    throw new Error("Project name is required.");
  }

  return {
    input: OPEN_PROJECT_FOLDER_ENDPOINT,
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nativePathLease: lease, name: projectName }),
    },
  };
}

export async function buildOpenProjectFolderRequestFromToken(
  nativePathToken: string,
  name: string,
  consume: (
    token: string,
    operation: "open-project",
  ) => Promise<{ nativePathLease: string }>,
): Promise<{ input: string; init: RequestInit }> {
  const token = nativePathToken.trim();
  if (!token) {
    throw new Error("A native project-folder token is required.");
  }
  const { nativePathLease } = await consume(token, "open-project");
  return buildOpenProjectFolderRequest(nativePathLease, name);
}
