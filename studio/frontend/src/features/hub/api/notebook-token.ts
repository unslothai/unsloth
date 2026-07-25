// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

interface NotebookTokenResponse {
  token: string | null;
  source: "colab" | "kaggle" | null;
}

export async function loadNotebookHfToken(): Promise<string | null> {
  const response = await authFetch("/api/settings/notebook-hf-token", {
    method: "POST",
  });
  if (!response.ok) {
    return null;
  }
  const payload = (await response.json()) as NotebookTokenResponse;
  return payload.token?.trim() || null;
}
