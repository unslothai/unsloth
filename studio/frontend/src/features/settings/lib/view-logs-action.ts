// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The "View logs" affordance a failure offers, in sonner's action shape.
 *
 * Shared so the three failure surfaces (GGUF load, video, image) name the same family.
 * `translate`, not the `useT` hook: two of the three raise their toast outside a
 * component body.
 */

import { isAccountOwner } from "@/features/auth/account-session";
import { translate } from "@/i18n";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Which log a given failure is explained by. Each GGUF runner writes its own file per
 * attempt (`utils/debug_log_sources.py` names both families); the diffusers and sd.cpp
 * paths log through the backend's stream, so theirs are in `server`. */
export type FailureLogFamily = "llama-server" | "diffusion-server" | "server";

/** The family a model-load failure is explained by.
 *
 * Only a GGUF load has a runner of its own; a Transformers or MLX failure is in the
 * current server log, so naming a runner there opens an unrelated older attempt.
 *
 * `runnerLogPath` answers the same question in time: a GGUF load can fail before any
 * runner starts, in the client's preflight or in the backend ahead of the launch, and the
 * backend names a path exactly when one ran. No path, no file of this attempt's to open.
 */
export function loadFailureLogFamily(
  isGguf: boolean | undefined,
  isDiffusion: boolean | undefined,
  runnerLogPath: string | null | undefined,
): FailureLogFamily {
  if (!runnerLogPath || !isGguf) return "server";
  return isDiffusion === true ? "diffusion-server" : "llama-server";
}

/** Pull the log path out of a load diagnostic (`llama_cpp.py` appends `Full log: <path>`).
 *
 * Family recency alone picks the wrong file after a rolled-back switch, where the newest
 * log in the family belongs to the rollback that succeeded. The path pins the attempt
 * that failed.
 */
export function failureLogPath(message: string): string | null {
  const at = message.lastIndexOf("Full log: ");
  if (at === -1) return null;
  const path = message
    .slice(at + "Full log: ".length)
    .split("\n")[0]
    .trim();
  return path || null;
}

/** The action, or undefined for an account with nowhere to be sent.
 *
 * Settings > Logs is owner-only and resolveSettingsTab reroutes a managed account to
 * General, so the button landed on an unrelated tab; the sources route is owner-guarded
 * too. Callers pass this straight through as `action`, where undefined renders nothing. */
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
