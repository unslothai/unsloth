// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
    }
    throw new Error(detail);
  }
  return (await res.json()) as VoiceDecisionStatus;
}

export async function fetchVoiceDecision(): Promise<VoiceDecisionStatus> {
  return read(await authFetch(URL));
}

export async function setVoiceDecision(
  name: string | null,
): Promise<VoiceDecisionStatus> {
  return read(
    await authFetch(URL, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  );
}

const POLL_MS = 1500;

export function useVoiceDecision(refreshKey: unknown): {
  status: VoiceDecisionStatus | null;
  state: VoiceLoadState | null;
  switched: (s: VoiceDecisionStatus) => void;
  failed: boolean;
  reload: () => void;
} {
  const [status, setStatus] = useState<VoiceDecisionStatus | null>(null);
  const [failed, setFailed] = useState(false);
  const [pending, setPending] = useState(false);
  const [reads, setReads] = useState(0);
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
  }, [refreshKey, reads]);

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
    reload: () => setReads((n) => n + 1),
  };
}
