// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useEngines } from "@/features/model-picker/hooks/use-engines";
import { useSettingsDialogStore } from "@/features/settings";

const ENGINE_NAMES = { vllm: "vLLM", sglang: "SGLang" } as const;

export function isManagedEngineType(
  providerType: string,
): providerType is keyof typeof ENGINE_NAMES {
  return providerType in ENGINE_NAMES;
}

/** Points a new vLLM / SGLang connection at Unsloth's own install of that engine, when this host can run it. */
export function ManagedEngineHint({
  engine,
}: {
  engine: keyof typeof ENGINE_NAMES;
}) {
  const isOwner = useIsAccountOwner();
  const { engines } = useEngines();
  const status = engines.find((item) => item.engine === engine);
  if (!status || status.unsupported_reason || !(isOwner || status.installed)) {
    return null;
  }
  const name = ENGINE_NAMES[engine];
  return (
    <p className="pb-3 text-xs leading-snug text-muted-foreground">
      {status.installed
        ? `${name} is already installed in Unsloth. Pick it as the Inference engine in a model's run settings to run models without a separate server.`
        : `Unsloth can directly install ${name} and run models with it, so you do not need a separate server.`}{" "}
      <button
        type="button"
        onClick={() =>
          useSettingsDialogStore
            .getState()
            .openDialog("resources", { scrollTarget: "resources-engines" })
        }
        className="font-medium text-foreground underline-offset-2 hover:underline"
      >
        {status.installed ? "Manage engines" : `Install ${name}`}
      </button>
    </p>
  );
}
