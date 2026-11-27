// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of api.ts so startup chunks skip them; only the lazy-loaded Images page uses them.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import {
  parsePromptUpsamplerStatus,
  PROMPT_UPSAMPLER_ROUTES,
  upsamplerRouteFor,
  type PromptUpsamplerPreviewRequest,
  type PromptUpsamplerStatus,
} from "./lib/prompt-enhance";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getPromptUpsamplerStatus(
  signal?: AbortSignal,
  family?: string | null,
): Promise<PromptUpsamplerStatus> {
  return parsePromptUpsamplerStatus(
    await parseJson<unknown>(
      await authFetch(upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.status, family), { signal }),
    ),
  );
}

/** Start the background download of the prompt upsampler. Progress is read back from the status route. */
export async function startPromptUpsamplerDownload(
  hfToken?: string,
  family?: string | null,
): Promise<void> {
  await parseJson<unknown>(
    await authFetch(upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.download, family), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hfToken ? { hf_token: hfToken } : {}),
    }),
  );
}

/** Stop the download. Partial files stay cached, so a later download resumes. */
export async function cancelPromptUpsamplerDownload(family?: string | null): Promise<void> {
  await parseJson<unknown>(
    await authFetch(upsamplerRouteFor(PROMPT_UPSAMPLER_ROUTES.cancel, family), { method: "POST" }),
  );
}

/** Rewrite a prompt without generating, so the caption can be read or edited first. */
export async function previewEnhancedPrompt(
  body: PromptUpsamplerPreviewRequest,
): Promise<{ enhanced_prompt: string }> {
  return parseJson(
    await authFetch(PROMPT_UPSAMPLER_ROUTES.preview, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** Best-effort background start so the first run skips it; returns the backend's outcome. */
export async function warmPromptUpsampler(signal?: AbortSignal): Promise<string> {
  // A release sent before this (Enhance off, then quickly on) is answered first, so it cannot stop what this starts.
  await pendingRelease;
  signal?.throwIfAborted();
  const body = await parseJson<{ warm?: unknown }>(
    await authFetch(PROMPT_UPSAMPLER_ROUTES.warm, { method: "POST", signal }),
  );
  return typeof body.warm === "string" ? body.warm : "";
}

let pendingRelease: Promise<void> = Promise.resolve();

/** Enhance prompt turned off: cancel a warm start still launching and stop an idle server. Best effort. */
export function releasePromptUpsampler(): Promise<void> {
  const release = (async () => {
    await parseJson<unknown>(await authFetch(PROMPT_UPSAMPLER_ROUTES.release, { method: "POST" }));
  })();
  pendingRelease = release.catch(() => {});
  return release;
}
