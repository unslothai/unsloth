// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The "Mask from text" segmenter routes, kept out of api.ts like the prompt upsampler's: only the lazily loaded Images
// page uses them.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import {
  MASK_SEGMENT_ROUTES,
  parseSegmenterStatus,
  SEGMENTER_STATUS_EVENT,
  parseSegmentResult,
  type SegmenterStatus,
  type SegmentRequest,
  type SegmentResult,
} from "./lib/mask-from-text";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getSegmenterStatus(signal?: AbortSignal): Promise<SegmenterStatus> {
  const status = parseSegmenterStatus(
    await parseJson<unknown>(await authFetch(MASK_SEGMENT_ROUTES.status, { signal })),
  );
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(SEGMENTER_STATUS_EVENT, { detail: status }));
  }
  return status;
}

/** Start the background download of the active segmenter variant. Progress is read back from the status route. */
export async function startSegmenterDownload(hfToken?: string): Promise<void> {
  await parseJson<unknown>(
    await authFetch(MASK_SEGMENT_ROUTES.download, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hfToken ? { hf_token: hfToken } : {}),
    }),
  );
}

/** Stop the download. Partial files stay cached, so a later download resumes. */
export async function cancelSegmenterDownload(): Promise<void> {
  await parseJson<unknown>(await authFetch(MASK_SEGMENT_ROUTES.cancel, { method: "POST" }));
}

/** The mask for ``prompt`` on ``image``, at the image's own size. */
export async function segmentImage(body: SegmentRequest, signal?: AbortSignal): Promise<SegmentResult> {
  return parseSegmentResult(
    await parseJson<unknown>(
      await authFetch(MASK_SEGMENT_ROUTES.segment, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      }),
    ),
  );
}

/** The painter closed: free the segmenter now instead of holding it for its keep-alive. Best effort. */
export function releaseSegmenter(): Promise<void> {
  return (async () => {
    await parseJson<unknown>(await authFetch(MASK_SEGMENT_ROUTES.release, { method: "POST" }));
  })();
}
