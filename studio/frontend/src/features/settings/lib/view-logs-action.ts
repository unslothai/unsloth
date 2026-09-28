// SPDX-License-Identifier: AGPL-3.0-only Copyright 2026-present the Unsloth AI Inc.

/** The "View logs" affordance a failure offers, in sonner's action shape. */

import { isAccountOwner } from "@/features/auth/account-session";
import { translate } from "@/i18n";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Which log a given failure is explained by. */
export type FailureLogFamily = "llama-server" | "diffusion-server" | "server";

/** The family a model-load failure is explained by. */
export function loadFailureLogFamily(
  isGguf: boolean | undefined,
  isDiffusion: boolean | undefined,
  runnerLogPath: string | null | undefined,
): FailureLogFamily {
  if (!runnerLogPath || !isGguf) return "server";
  return isDiffusion === true ? "diffusion-server" : "llama-server";
}

/** Pull the log path out of a load diagnostic (`llama_cpp.py` appends `Full log: <path>`). */
export function failureLogPath(message: string): string | null {
  const at = message.lastIndexOf("Full log: ");
  if (at === -1) return null;
  const path = message
    .slice(at + "Full log: ".length)
    .split("\n")[0]
    .trim();
  return path || null;
}

/** The action, or undefined for an account with nowhere to be sent. */
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
