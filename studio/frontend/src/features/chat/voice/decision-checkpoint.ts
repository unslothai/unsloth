// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Decision checkpoint voice mode decides with (turn end, filler, interrupt). A switch is
// stored installation-wide and starts loading at once; the next decision uses it.

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export type VoiceLoadState = "loaded" | "loading" | "not_loaded" | "error";

export interface VoiceDecisionModel {
  name: string;
  label: string;
  description: string;
  kind: "catalog" | "fine_tune" | "folder";
  path: string | null;
  available: boolean;
  unavailable_reason: string | null;
}

export interface VoiceDecisionStatus {
  current: string | null;
  source: "setting" | "env" | null;
  launch_default: string | null;
  models: VoiceDecisionModel[];
  state: VoiceLoadState;
  error: string | null;
}

const URL = "/api/inference/voice/decision";

async function read(res: Response): Promise<VoiceDecisionStatus> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // keep the status line
    }
    throw new Error(detail);
  }
  return (await res.json()) as VoiceDecisionStatus;
}

export async function fetchVoiceDecision(): Promise<VoiceDecisionStatus> {
  return read(await authFetch(URL));
}

/** null clears the pick, so the launch default applies again. */
export async function setVoiceDecision(name: string | null): Promise<VoiceDecisionStatus> {
  return read(
    await authFetch(URL, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  );
}

const POLL_MS = 1500;

/** The current pick, refreshed on `refreshKey` and polled while it loads. null until read or when refused. */
export function useVoiceDecision(refreshKey: unknown): {
  status: VoiceDecisionStatus | null;
  /** The load state to show: a switch still starting reads as loading. */
  state: VoiceLoadState | null;
  /** Take a switch's answer and poll until the new pick has loaded or failed. */
  switched: (s: VoiceDecisionStatus) => void;
  failed: boolean;
} {
  const [status, setStatus] = useState<VoiceDecisionStatus | null>(null);
  const [failed, setFailed] = useState(false);
  // A switch answers before its load has started, so "not loaded" right after one is still pending.
  const [pending, setPending] = useState(false);
  const settled = status?.state === "loaded" || status?.state === "error";
  const loading = status?.state === "loading" || (pending && !settled);

  useEffect(() => {
    if (pending && settled) setPending(false);
  }, [pending, settled]);
  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(() => setPending(false), 120_000);
    return () => clearTimeout(timer);
  }, [pending]);

  useEffect(() => {
    let cancelled = false;
    fetchVoiceDecision()
      .then((s) => {
        if (!cancelled) {
          setStatus(s);
          setFailed(false);
        }
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => {
      fetchVoiceDecision()
        .then(setStatus)
        .catch(() => {});
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [loading]);

  return {
    status,
    state: loading ? "loading" : (status?.state ?? null),
    switched: (s) => {
      setStatus(s);
      setPending(s.current !== null);
    },
    failed,
  };
}
