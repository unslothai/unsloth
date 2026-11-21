// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAccountOwner } from "@/features/auth/account-session";
import { translate } from "@/i18n";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export type FailureLogFamily = "llama-server" | "diffusion-server" | "server";

export function loadFailureLogFamily(
  isGguf: boolean | undefined,
  isDiffusion: boolean | undefined,
  runnerLogPath: string | null | undefined,
): FailureLogFamily {
  if (!runnerLogPath || !isGguf) return "server";
  return isDiffusion === true ? "diffusion-server" : "llama-server";
}

/** `llama_cpp.py` appends `Full log: <path>`. */
export function failureLogPath(message: string): string | null {
  const at = message.lastIndexOf("Full log: ");
  if (at === -1) return null;
  const path = message
    .slice(at + "Full log: ".length)
    .split("\n")[0]
    .trim();
  return path || null;
}

export function viewLogsAction(
  family: FailureLogFamily,
  sourcePath?: string | null,
):
  | {
      label: string;
      onClick: () => void;
    }
  | undefined {
  if (!isAccountOwner()) return undefined;
  return {
    label: translate("settings.debugging.viewLogs"),
    onClick: () =>
      useSettingsDialogStore.getState().openLogs(family, sourcePath ?? null),
  };
}

// Messages the backend logs; client-input refusals carry their own text instead.
const LOGGED_GENERATION_FAILURES = [
  "Image generation failed.",
  "Video generation failed.",
  "Failed to save the generated image.",
  "Failed to save the generated video.",
];

export function generationFailureLogsAction(message: string) {
  return LOGGED_GENERATION_FAILURES.some((prefix) => message.startsWith(prefix))
    ? viewLogsAction("server")
    : undefined;
}
