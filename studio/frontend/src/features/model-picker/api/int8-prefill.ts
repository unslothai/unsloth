// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type PrefillOption = "int8" | "ane";

/** Failure only hides the setting: a stored choice is kept and the load-time check guards it. */
export async function fetchPrefillOptionAvailable(
  option: PrefillOption,
  modelPath: string,
  hfToken: string | null,
  signal?: AbortSignal,
): Promise<boolean> {
  const response = await authFetch(`/api/inference/${option}-prefill-availability`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ model_path: modelPath, hf_token: hfToken }),
  });
  if (!response.ok) {
    return false;
  }
  const body: unknown = await response.json().catch(() => null);
  return (
    typeof body === "object" &&
    body !== null &&
    (body as { available?: unknown }).available === true
  );
}
