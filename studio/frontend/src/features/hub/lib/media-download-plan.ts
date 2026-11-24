// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports -- Direct API imports keep the media pages code-split.
import {
  type DiffusionDownloadPlan,
  type DiffusionLoadRequest,
  getDiffusionDownloadPlan,
} from "@/features/images/api";
// eslint-disable-next-line no-restricted-imports -- Direct API imports keep the media pages code-split.
import { getVideoDownloadPlan } from "@/features/video/api";
import { withAllowOversizedLoad } from "@/lib/unified-memory-load-refusal";

export type MediaPlanPage = "images" | "video";

/** The fields both plan routes share; the Hub cards send nothing else. */
export type MediaPlanBody = Pick<
  DiffusionLoadRequest,
  "model_path" | "gguf_filename" | "model_kind" | "hf_token"
>;

// The card's two asks start ~200 ms apart; a plan still pending past this is not joined, so a
// stalled request cannot hold every later open of the same file.
const SHARE_WINDOW_MS = 5_000;

const inFlight = new Map<
  string,
  { request: Promise<DiffusionDownloadPlan>; started: number }
>();

/** One request for identical concurrent plans; dropped once settled, so nothing outlives the request. */
export function inFlightMediaDownloadPlan(
  page: MediaPlanPage,
  body: MediaPlanBody,
  inventoryVersion: number,
): Promise<DiffusionDownloadPlan> {
  const key = JSON.stringify([
    page,
    getAuthSessionEpoch(),
    inventoryVersion,
    JSON.stringify(withAllowOversizedLoad<DiffusionLoadRequest>(body)),
  ]);
  const pending = inFlight.get(key);
  if (pending && Date.now() - pending.started < SHARE_WINDOW_MS) {
    return pending.request;
  }
  const request =
    page === "video"
      ? getVideoDownloadPlan(body)
      : getDiffusionDownloadPlan(body);
  inFlight.set(key, { request, started: Date.now() });
  const settle = () => {
    if (inFlight.get(key)?.request === request) {
      inFlight.delete(key);
    }
  };
  request.then(settle, settle);
  return request;
}
