// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the Images page's primary button does and whether it is on. No app deps, so it is testable on its own.

export const SWITCHING_MODEL_NOTICE =
  "Switching models. Generate turns back on once the new model has loaded.";
export const RELOAD_AND_GENERATE_LABEL = "Reload and generate";

export interface GenerateGateInput {
  /** The page's busy state: "loading" | "unloading" | "generating" | null. */
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
  /** One line above the button saying why it is off or what it will do; null for none. */
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
