// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const SWITCHING_MODEL_NOTICE =
  "Switching models. Generate turns back on once the new model has loaded.";
export const RELOAD_AND_GENERATE_LABEL = "Reload and generate";

export interface GenerateGateInput {
  busy: string | null;
  presetsHydrated: boolean;
  loaded: boolean;
  /** A model Generate can load first (the remembered one), or an external connection. */
  canRecallModel: boolean;
  external: boolean;
  /** A model pick is planning or downloading: the resident model is about to be replaced. */
  switching: boolean;
  /** Why the LoRA selection needs a reload first (lora-reload.ts), or null. */
  loraReload: string | null;
  queueWaiting: boolean;
  canQueue: boolean;
}

export interface GenerateGate {
  disabled: boolean;
  action: "generate" | "queue" | "reload";
  label: string;
  notice: string | null;
}

export function generateGate(input: GenerateGateInput): GenerateGate {
  const switching = input.switching && !input.external;
  const reload = !input.external && input.loaded ? input.loraReload : null;
  const action: GenerateGate["action"] = reload
    ? "reload"
    : input.queueWaiting && input.canQueue
      ? "queue"
      : "generate";
  const disabled =
    input.busy !== null ||
    !input.presetsHydrated ||
    switching ||
    (!input.loaded && !input.canRecallModel && !input.external);
  return {
    disabled,
    action,
    label: action === "reload" ? RELOAD_AND_GENERATE_LABEL : action === "queue" ? "Queue" : "Generate",
    notice: input.busy !== null ? null : switching ? SWITCHING_MODEL_NOTICE : reload,
  };
}

/** A pick released by a page switch while still newest: nothing settles it, so the page takes it back (else Generate
 *  stays held forever). A newer claim or a cancel answers false: that owner settles it. */
export function pickReleasedWhileNewest(
  guard: { holds(token: number): boolean; isLatest(token: number): boolean },
  token: number | undefined,
): boolean {
  return token !== undefined && token !== 0 && !guard.holds(token) && guard.isLatest(token);
}
