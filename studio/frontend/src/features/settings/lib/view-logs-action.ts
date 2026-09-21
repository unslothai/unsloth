// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The "View logs" affordance a failure offers, in sonner's action shape.
 *
 * Shared so the three failure surfaces (a GGUF load, a video generation, an image
 * generation) offer the same thing and name the same family, rather than each growing
 * its own wording. The reported experience was a failure with no reason and no route to
 * one: Settings > Logs is the route, and until now nothing pointed at it.
 *
 * Translated through `translate` rather than the `useT` hook, because two of the three
 * call sites raise their toast from a callback outside a component body.
 */

import { isAccountOwner } from "@/features/auth/account-session";
import { translate } from "@/i18n";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Which log a given failure is explained by.
 *
 * `llama-server` for a GGUF text load and `diffusion-server` for a GGUF diffusion one,
 * because each runner writes its own file per attempt and the reason is in there rather
 * than in the server log (utils/debug_log_sources.py names both families). The diffusers
 * and sd.cpp paths log through the backend's own stream, so their failures are in `server`.
 */
export type FailureLogFamily = "llama-server" | "diffusion-server" | "server";

/** The family a model-load failure is explained by, from what was being loaded.
 *
 * Only a GGUF load goes through a runner that writes its own file. A Transformers or MLX
 * load has none, and its failure is recorded by the backend logger in the CURRENT server
 * log, so answering `llama-server` there opens whatever older runner attempt happens to be
 * on the host -- an unrelated log, which is worse than none.
 *
 * `runnerAttempted` is the same question in time rather than in kind: a GGUF load that
 * fails in PREFLIGHT, before the load request is issued, started no runner either, so it
 * has no file of its own however it was going to be served.
 */
export function loadFailureLogFamily(
  isGguf: boolean | undefined,
  isDiffusion: boolean | undefined,
  runnerAttempted: boolean,
): FailureLogFamily {
  if (!runnerAttempted || !isGguf) return "server";
  return isDiffusion === true ? "diffusion-server" : "llama-server";
}

/** Pull the log path the backend already names out of a load diagnostic.
 *
 * llama_cpp.py appends `Full log: <path>` to the message it raises. Selecting by family
 * recency alone picks the WRONG file whenever the failed switch was rolled back: the
 * rollback load writes its own runner log afterwards, so the newest file in the family is
 * the one that succeeded. The path pins the attempt that actually failed.
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

/** The action, or undefined for an account that has nowhere to be sent.
 *
 * Settings > Logs is owner-only (OWNER_ONLY_SETTINGS_TABS), and resolveSettingsTab sends a
 * managed account asking for it to General, so on a managed installation the button opened
 * Settings on an unrelated tab and looked broken. The /debug/logs/sources route is
 * owner-guarded too, so there is no log to offer, and no button is better than a dead one.
 * Callers pass the result straight through as `action`, where undefined renders nothing. */
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
