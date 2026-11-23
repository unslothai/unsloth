// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the Restyle tab shows about setup, from /video/restyle/status. Pure, so it is testable. */

export interface RestyleSetupProgress {
  active: boolean;
  phase: string | null;
  weights?: "bf16" | "q8_0" | null;
  bytes_done?: number | null;
  bytes_total?: number | null;
  error?: string | null;
  cancelling?: boolean;
}

export interface RestyleStatusLike {
  available: boolean;
  reason: string | null;
  missing?: string[];
  can_setup?: boolean;
  refusal?: string | null;
  weights?: "bf16" | "q8_0" | null;
  vram_gib?: number | null;
  setup?: RestyleSetupProgress | null;
}

export type RestyleSetupView =
  | { kind: "ready" }
  | { kind: "refused"; message: string }
  | { kind: "blocked"; message: string }
  | { kind: "offer"; message: string; error: string | null }
  | {
      kind: "running";
      label: string;
      percent: number | null;
      cancelling: boolean;
    };

const PHASE_LABEL: Record<string, string> = {
  engine: "Installing the stable-diffusion.cpp server",
  download: "Downloading Cosmos3-Nano-Sim-Transfer",
  merge: "Merging the transformer into one file",
  vae: "Preparing the Wan2.2 VAE",
  quantize: "Making the 8-bit copy for this GPU",
};

function gb(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

export function restyleSetupView(
  status: RestyleStatusLike | null,
): RestyleSetupView | null {
  if (status === null) return null;
  const setup = status.setup;
  if (setup?.active) {
    const done = setup.bytes_done ?? null;
    const total = setup.bytes_total ?? null;
    const percent =
      done != null && total
        ? Math.min(100, Math.round((done / total) * 100))
        : null;
    const base = PHASE_LABEL[setup.phase ?? ""] ?? "Setting up Restyle";
    const label =
      done != null && total ? `${base} (${gb(done)} of ${gb(total)})` : base;
    return {
      kind: "running",
      label,
      percent,
      cancelling: setup.cancelling === true,
    };
  }
  if (status.available) return { kind: "ready" };
  if (status.refusal) return { kind: "refused", message: status.refusal };
  if (!status.can_setup) {
    return {
      kind: "blocked",
      message: status.reason ?? "Restyle is not set up here.",
    };
  }
  const missing = status.missing ?? [];
  const parts: string[] = [];
  if (missing.includes("engine"))
    parts.push("installs the stable-diffusion.cpp server");
  if (missing.includes("weights")) {
    parts.push(
      status.weights === "q8_0"
        ? "downloads the model (about 35 GB) and makes an 8-bit copy that fits this GPU (needs about 82 GB of free disk while it runs, 17 GB after)"
        : "downloads the model (about 35 GB; needs about 67 GB of free disk while it runs, 31 GB after)",
    );
  }
  const message = `Set up Restyle ${parts.join(" and ")}. Tested on Linux with an NVIDIA GPU and on an Apple Silicon Mac.`;
  const error =
    setup?.phase === "error" ? (setup.error ?? "Restyle setup failed.") : null;
  return { kind: "offer", message, error };
}
