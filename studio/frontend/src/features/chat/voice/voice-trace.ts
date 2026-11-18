// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { authFetch } from "@/features/auth";

let enabled: boolean | null = null;
let probe: Promise<boolean> | null = null;

async function check(): Promise<boolean> {
  if (enabled !== null) return enabled;
  // A failed probe stays unknown so the next event asks again; not /voice/models/status, which loads every model.
  probe ??= authFetch("/api/inference/voice/trace")
    .then(async (r) => {
      if (!r.ok) return null;
      return Boolean(((await r.json()) as { trace?: boolean }).trace);
    })
    .catch(() => null)
    .then((on) => {
      probe = null;
      if (on !== null) enabled = on;
      return on ?? false;
    });
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
