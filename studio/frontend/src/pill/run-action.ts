// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pollSignal } from "@/features/hub/lib/abort-signals";
import {
  fetchInferenceStatus,
  getCachedSettings,
  requestModelLoad,
} from "./api";

const STATUS_POLL_MS = 3_000;
const MODEL_LOAD_TIMEOUT_MS = 5 * 60_000;

function modelMatchesLoaded(
  model: string,
  status: { active_model: string | null; model_identifier: string | null },
): boolean {
  const target = model.toLowerCase();
  return (
    status.model_identifier?.toLowerCase() === target ||
    status.active_model?.toLowerCase() === target
  );
}

export async function ensureModelLoaded(
  model: string,
  ggufVariant: string | null,
  signal: AbortSignal,
  onLoading: (model: string) => void,
): Promise<void> {
  let status = await fetchInferenceStatus();
  if (modelMatchesLoaded(model, status)) return;

  const settings = getCachedSettings();
  if (settings && !settings.autoLoad) {
    throw new PillRunError("loadFailed", model);
  }

  onLoading(model);
  // The load itself is now the long part, so the budget has to start here and
  // bound the request: a load that never finishes would otherwise park on this
  // await forever, leaving the panel loading until the user gives up.
  const deadline = Date.now() + MODEL_LOAD_TIMEOUT_MS;
  const budget = pollSignal(signal, MODEL_LOAD_TIMEOUT_MS);
  try {
    // Resolves only once the load itself finished, so the poll loop below just
    // confirms which model ended up active.
    await requestModelLoad(model, ggufVariant, budget.signal);
  } finally {
    budget.dispose();
  }

  // Two consecutive polls with nothing loading means the load never
  // registered or already failed; one idle poll is grace for registration lag.
  let idlePolls = 0;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    await new Promise((resolve) => setTimeout(resolve, STATUS_POLL_MS));
    status = await fetchInferenceStatus().catch(() => status);
    if (modelMatchesLoaded(model, status)) return;
    idlePolls = status.loading.length === 0 ? idlePolls + 1 : 0;
    if (idlePolls >= 2) break;
  }
  throw new PillRunError("loadFailed", model);
}

export class PillRunError extends Error {
  errorKey: string;
  model: string | null;

  constructor(errorKey: string, model: string | null = null) {
    super(errorKey);
    this.errorKey = errorKey;
    this.model = model;
  }
}

export function classifyFetchError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/isn't running|Failed to fetch|NetworkError|offline/i.test(message)) {
    return "backendDown";
  }
  if (/401|Unauthorized|sign in/i.test(message)) {
    return "signedOut";
  }
  if (/No model loaded/i.test(message)) {
    return "modelMissing";
  }
  return "captureFailed";
}
