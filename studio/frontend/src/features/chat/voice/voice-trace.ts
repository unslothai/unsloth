// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Loop events for offline debugging. Sent only when the backend has a VAD debug
// dir (UNSLOTH_VAD_DEBUG_DIR), where they land in voice-trace.jsonl beside the clips.

import { authFetch } from "@/features/auth";

let enabled: boolean | null = null;
let probe: Promise<boolean> | null = null;

async function check(): Promise<boolean> {
  if (enabled !== null) return enabled;
  probe ??= authFetch("/api/inference/voice/models/status")
    .then(async (r) => (r.ok ? Boolean(((await r.json()) as { trace?: boolean }).trace) : false))
    .catch(() => false)
    .then((on) => (enabled = on));
  return probe;
}

export function voiceTrace(event: string, detail?: Record<string, unknown>): void {
  const t = performance.now();
  void check().then((on) => {
    if (!on) return;
    void authFetch("/api/inference/voice/trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, detail, t }),
    }).catch(() => {});
  });
}
