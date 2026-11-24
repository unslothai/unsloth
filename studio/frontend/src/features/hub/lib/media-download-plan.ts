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

const inFlight = new Map<string, Promise<DiffusionDownloadPlan>>();

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
  if (pending) return pending;
  const request =
    page === "video"
      ? getVideoDownloadPlan(body)
      : getDiffusionDownloadPlan(body);
  inFlight.set(key, request);
  const settle = () => {
    if (inFlight.get(key) === request) inFlight.delete(key);
  };
  request.then(settle, settle);
  return request;
}
