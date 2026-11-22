// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { formatFastApiDetail } from "@/lib/format-fastapi-error";

// Whether RAG can run on this host: the KB list answers 200 with a marker, other endpoints 503.
// Gate only on isUnavailable(); everything is optimistic until the backend actually answers.

interface RagAvailabilityMarker {
  ragAvailable?: unknown;
  ragUnavailableReason?: unknown;
}

const DEFAULT_UNAVAILABLE_REASON =
  "RAG is unavailable on this machine: the sqlite-vec extension could not be loaded.";

// Match only the extension name: a proxy 503 or a generic "RAG is unavailable" must not persist a
// capability verdict from a transient outage.
const RAG_UNAVAILABLE_MARKERS = ["sqlite-vec"];

function isRagUnavailableDetail(detail: string | null | undefined): detail is string {
  if (!detail) return false;
  const text = detail.toLowerCase();
  return RAG_UNAVAILABLE_MARKERS.some((marker) => text.includes(marker));
}

interface RagAvailabilityState {
  // Optimistic seed; never gate on it directly, see isUnavailable().
  available: boolean;
  reason: string | null;
  answered: boolean;
  isUnavailable: () => boolean;
  availabilityUnknown: () => boolean;
  unavailableReason: () => string | null;
}

export const useRagAvailabilityStore = create<RagAvailabilityState>()(
  (_, get) => ({
    available: true,
    reason: null,
    answered: false,
    isUnavailable: () => {
      const state = get();
      return state.answered && !state.available;
    },
    availabilityUnknown: () => !get().answered,
    unavailableReason: () => {
      const state = get();
      return state.answered && !state.available ? state.reason : null;
    },
  }),
);

export function hasRagAvailabilityMarker(body: unknown): boolean {
  if (!body || typeof body !== "object") return false;
  return typeof (body as RagAvailabilityMarker).ragAvailable === "boolean";
}

/** No-op without a marker (older backend): inventing an answer would gray the dialog on a guess. */
export function noteRagAvailability(body: unknown): void {
  if (!hasRagAvailabilityMarker(body)) return;
  const { ragAvailable, ragUnavailableReason } = body as RagAvailabilityMarker;
  if (ragAvailable === true) {
    useRagAvailabilityStore.setState({
      available: true,
      reason: null,
      answered: true,
    });
    return;
  }
  useRagAvailabilityStore.setState({
    available: false,
    reason:
      typeof ragUnavailableReason === "string" && ragUnavailableReason
        ? ragUnavailableReason
        : DEFAULT_UNAVAILABLE_REASON,
    answered: true,
  });
}

/**
 * A 2xx from a gated endpoint clears a stale unavailable; the list is excluded since it
 * answers 200 either way and its marker is the authority.
 */
export function noteRagResponse(status: number, body: unknown): void {
  if (status === 503) {
    const detail = formatFastApiDetail(
      (body as { detail?: unknown } | null)?.detail,
    );
    // Only the backend's own wording is a verdict; proxies and overloaded servers also return 503.
    if (!isRagUnavailableDetail(detail)) return;
    useRagAvailabilityStore.setState({
      available: false,
      reason: detail,
      answered: true,
    });
    return;
  }
  if (status < 200 || status >= 300) return;
  if (hasRagAvailabilityMarker(body)) return;
  useRagAvailabilityStore.setState({
    available: true,
    reason: null,
    answered: true,
  });
}
