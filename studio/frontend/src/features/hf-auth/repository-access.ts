// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub";

export type RepositoryAccessStatus =
  | "ready"
  | "authentication_required"
  | "invalid_token"
  | "not_found"
  | "no_write_permission"
  | "unavailable";

export async function validateRepositoryAccess(
  repoId: string,
  token: string | null,
  signal: AbortSignal,
): Promise<RepositoryAccessStatus> {
  const response = await authFetch("/api/hub/repository/access", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(token),
    },
    body: JSON.stringify({ repo_id: repoId }),
    signal,
  });
  if (!response.ok) return "unavailable";
  const body = (await response.json()) as { status?: RepositoryAccessStatus };
  return body.status ?? "unavailable";
}
