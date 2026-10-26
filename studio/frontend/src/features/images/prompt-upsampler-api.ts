// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The prompt upsampler's routes, kept out of api.ts: startup screens import api.ts (loaded models, hub
// companion bytes), and only the lazily loaded Images page uses these, so they stay out of the startup chunks.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import {
  parsePromptUpsamplerStatus,
  PROMPT_UPSAMPLER_ROUTES,
  type PromptUpsamplerPreviewRequest,
  type PromptUpsamplerStatus,
} from "./lib/prompt-enhance";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getPromptUpsamplerStatus(signal?: AbortSignal): Promise<PromptUpsamplerStatus> {
  return parsePromptUpsamplerStatus(
    await parseJson<unknown>(await authFetch(PROMPT_UPSAMPLER_ROUTES.status, { signal })),
  );
}

/** Start the background download of the prompt upsampler. Progress is read back from the status route. */
export async function startPromptUpsamplerDownload(hfToken?: string): Promise<void> {
  await parseJson<unknown>(
    await authFetch(PROMPT_UPSAMPLER_ROUTES.download, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hfToken ? { hf_token: hfToken } : {}),
    }),
  );
}

/** Stop the download. Partial files stay cached, so a later download resumes. */
export async function cancelPromptUpsamplerDownload(): Promise<void> {
  await parseJson<unknown>(
    await authFetch(PROMPT_UPSAMPLER_ROUTES.cancel, { method: "POST" }),
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

/** Start the upsampler's server in the background so the first enhanced run skips its start. Best effort: the
 *  backend's outcome ("scheduled", "loaded", or why it declined). */
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
