// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { formatFastApiDetail } from "@/lib/format-fastapi-error";

// Whether RAG can run on this host at all, as reported by the backend.
//
// A machine where sqlite-vec imports but its native library will not load has a working
// server and a dead RAG engine. The router says so rather than raising: GET
// /api/rag/knowledge-bases answers 200 with an availability marker beside the (empty)
// list, and every other RAG endpoint answers 503 with the same reason. Without this
// store both readings are dropped on the floor and the Knowledge bases dialog looks
// like an empty store, offering a Create button that can only 503.
//
// Read it the way the platform store in config/env.ts is read: `isUnavailable()`, never
// `available` on its own. Everything here is optimistic until the backend has actually
// answered, because graying a feature out on a guess is indistinguishable from a
// measured "unsupported", and a backend that predates the marker never answers at all.

/** The shape the KB list carries; older backends send neither field. */
interface RagAvailabilityMarker {
  ragAvailable?: unknown;
  ragUnavailableReason?: unknown;
}

/** Used when the backend states unavailable without stating why. */
const DEFAULT_UNAVAILABLE_REASON =
  "RAG is unavailable on this machine: the sqlite-vec extension could not be loaded.";

// What routes/rag.py sends with its 503 ("RAG is unavailable: the sqlite-vec extension
// could not be loaded."). Matched on the extension name rather than the whole sentence,
// so a reworded detail still reads as the backend's verdict while a proxy's "Service
// Temporarily Unavailable" does not.
//
// Deliberately NOT "rag is unavailable" as well. That phrase is the loose half of the
// pair: anything RAG-aware in front of the backend could emit it without meaning the
// extension, and matching either fragment would then persist a capability verdict from a
// transient outage. "sqlite-vec" is a package name, so nothing upstream says it by
// accident, and the whole capability being gated here is exactly that extension.
const RAG_UNAVAILABLE_MARKERS = ["sqlite-vec"];

/** True only for a 503 body the RAG router itself produced. */
function isRagUnavailableDetail(detail: string | null | undefined): detail is string {
  if (!detail) return false;
  const text = detail.toLowerCase();
  return RAG_UNAVAILABLE_MARKERS.some((marker) => text.includes(marker));
}

interface RagAvailabilityState {
  // Optimistic seed. Never gate on this directly; it is "not known to be broken",
  // not an answer. See isUnavailable().
  available: boolean;
  // Why RAG cannot run, straight from the backend, so the UI explains itself instead
  // of showing a generic empty state. Null while available or unknown.
  reason: string | null;
  // True once a RAG response has actually said something about availability.
  answered: boolean;
  /** The only safe gate: a measured "RAG cannot run here", never the optimistic seed. */
  isUnavailable: () => boolean;
  /** True until a response has settled it. Callers that would disable something hold. */
  availabilityUnknown: () => boolean;
  /** The reason to show, or null when there is nothing measured to explain. */
  unavailableReason: () => string | null;
}

export const useRagAvailabilityStore = create<RagAvailabilityState>()(
  (_, get) => ({
    available: true,
    reason: null,
    answered: false,
    // `answered` is the unknown/known line, so the optimistic seed can never gate
    // anything: an unreachable backend, a slow first poll and a backend with no marker
    // at all all render exactly as they do today.
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

/** True when a response body carries the backend's availability marker. */
export function hasRagAvailabilityMarker(body: unknown): boolean {
  if (!body || typeof body !== "object") return false;
  return typeof (body as RagAvailabilityMarker).ragAvailable === "boolean";
}

/**
 * Record the availability marker the KB list carries.
 *
 * No-op when the body has no marker: that is an older backend, or a build without the
 * contract, and inventing an answer for it is what would gray the dialog out on a guess.
 */
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
 * Record what an arbitrary RAG response says about availability.
 *
 * The KB list is the only endpoint that states it in the body; the rest state it as a
 * 503, which every one of them gates on. Reading both means a user who lands on a
 * mutating route first gets a coherent UI instead of waiting for the list poll.
 *
 * A 2xx from a gated endpoint is proof RAG is runnable, so it clears a stale
 * unavailable. The list is excluded from that rule: it answers 200 either way, and its
 * marker is the authority.
 */
export function noteRagResponse(status: number, body: unknown): void {
  if (status === 503) {
    const detail = formatFastApiDetail(
      (body as { detail?: unknown } | null)?.detail,
    );
    // Only the backend's own wording is a capability verdict. A 503 is also what a
    // reverse proxy, Cloudflare, or a briefly overloaded server returns, and those
    // bodies say nothing about sqlite-vec. Recording one as unavailable would gate the
    // dialog for the session behind a transient outage, showing an extension
    // explanation for something that was never the extension.
    if (!isRagUnavailableDetail(detail)) return;
    useRagAvailabilityStore.setState({
      available: false,
      reason: detail,
      answered: true,
    });
    return;
  }
  // Any other failure is transient (auth, validation, a dead network) and says nothing
  // about whether the extension loads.
  if (status < 200 || status >= 300) return;
  if (hasRagAvailabilityMarker(body)) return;
  useRagAvailabilityStore.setState({
    available: true,
    reason: null,
    answered: true,
  });
}
