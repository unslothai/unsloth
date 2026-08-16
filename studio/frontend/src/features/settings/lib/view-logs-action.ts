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

import { translate } from "@/i18n";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

/** Which log a given failure is explained by.
 *
 * `llama-server` for a GGUF load, because the runner writes its own file per attempt
 * and the reason is in there rather than in the server log. The diffusion runners log
 * through the backend's own stream, so their failures are in `server`.
 */
export type FailureLogFamily = "llama-server" | "server";

export function viewLogsAction(family: FailureLogFamily): {
  label: string;
  onClick: () => void;
} {
  return {
    label: translate("settings.debugging.viewLogs"),
    onClick: () => useSettingsDialogStore.getState().openLogs(family),
  };
}
