// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Cosmos3-Nano's guardrail package: spotting the load refusal that offers an install, and following the install. */

export interface CosmosGuardrailStatus {
  state: "idle" | "running" | "done" | "failed";
  installed: boolean;
  error: string | null;
  added?: string[];
  command?: string;
}

/** The load refusal for a missing cosmos-guardrail package (not the licence-access one, which no install fixes). */
export function isGuardrailInstallRefusal(text: string | null | undefined): boolean {
  const value = text ?? "";
  return value.includes("needs NVIDIA's gated Cosmos guardrail") && value.includes("Install guardrail");
}

/** Whether a polled install has finished, and how it ended. */
export function guardrailInstallOutcome(
  status: CosmosGuardrailStatus,
): { finished: false } | { finished: true; ok: boolean; message: string } {
  if (status.installed) {
    const added = status.added?.length ? ` Added ${status.added.join(", ")}.` : "";
    return {
      finished: true,
      ok: true,
      message:
        `cosmos-guardrail is installed.${added} Accept the licence at ` +
        "https://huggingface.co/nvidia/Cosmos-1.0-Guardrail, add a Hugging Face token in Settings, and load the model again.",
    };
  }
  if (status.state === "failed") {
    return { finished: true, ok: false, message: status.error || "The cosmos-guardrail install failed." };
  }
  if (status.state === "running") return { finished: false };
  return { finished: true, ok: false, message: "The cosmos-guardrail install did not start." };
}

/** Start the install, then poll until it ends (or ``maxPolls`` runs out). */
export async function runGuardrailInstall(
  start: () => Promise<CosmosGuardrailStatus>,
  poll: () => Promise<CosmosGuardrailStatus>,
  wait: (ms: number) => Promise<void>,
  { intervalMs = 3000, maxPolls = 900 }: { intervalMs?: number; maxPolls?: number } = {},
): Promise<{ ok: boolean; message: string }> {
  let status = await start();
  let misses = 0;
  for (let i = 0; i <= maxPolls; i += 1) {
    const outcome = guardrailInstallOutcome(status);
    if (outcome.finished) return { ok: outcome.ok, message: outcome.message };
    await wait(intervalMs);
    try {
      status = await poll();
      misses = 0;
    } catch (err) {
      // One dropped poll is not a failed install; give up only after several in a row.
      misses += 1;
      if (misses >= 5) throw err;
    }
  }
  return { ok: false, message: "The cosmos-guardrail install is still running; check again later." };
}
