// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Evals tab's model pick, in the same header slot next to the tabs where the
// config-sweep page puts its "Pick a GGUF" pill. Same shared BenchModelPicker, wired
// to the Evals selection: picking loads the model right away (chat's runtime) and
// offers LoRA checkpoints and Hub repos on top of on-device GGUFs.

import { BenchModelPicker } from "@/features/benchmarks/components/bench-model-picker";
import type { EvalsModel } from "../use-evals-model";

export function EvalsModelPicker({ evals }: { evals: EvalsModel }) {
  return (
    <BenchModelPicker
      models={evals.models}
      loraModels={evals.loraModels}
      value={evals.selectedModel ?? undefined}
      ggufVariant={
        evals.selectedModel != null &&
        evals.selectedModel === evals.residentModel
          ? evals.activeGgufVariant
          : evals.selectedGgufVariant
      }
      loaded={
        !evals.selectedModel || evals.selectedModel === evals.residentModel
      }
      variant="ghost"
      className="!h-[calc(34px*var(--ui-space-scale,1))] max-w-full"
      placeholder="Pick a GGUF to benchmark"
      onPick={evals.pick}
    />
  );
}
