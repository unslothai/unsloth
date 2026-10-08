// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Tabs } from "@/components/ui/tabs";
import { usePlatformStore } from "@/config/env";
import {
  rlObjectiveSupported,
  useTrainingConfigStore,
} from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import type { TrainingObjective } from "@/types/training";
import type { ReactElement } from "react";

const HINT_KEYS: Record<TrainingObjective, TranslationKey> = {
  sft: "rl.objective.sftHint",
  dpo: "rl.objective.dpoHint",
  orpo: "rl.objective.orpoHint",
  grpo: "rl.objective.grpoHint",
};

export function ObjectiveSelect(): ReactElement {
  const t = useT();
  const selected = useTrainingConfigStore((s) => s.trainingObjective);
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  const setObjective = useTrainingConfigStore((s) => s.setTrainingObjective);
  const isMac = usePlatformStore((s) => s.deviceType) === "mac";
  const modelLocked = !useTrainingConfigStore(rlObjectiveSupported);
  // MLX has no RL trainer, and CPT is an objective of its own.
  const rlLocked = isMac || trainingMethod === "cpt" || modelLocked;
  const lockedReason = isMac
    ? t("rl.objective.macLocked")
    : modelLocked
      ? t("rl.objective.modelLocked")
      : t("rl.objective.cptLocked");
  // A locked selector shows what the run will use, not a stored RL choice it ignores.
  const objective = rlLocked ? "sft" : selected;

  return (
    <div className="flex flex-col gap-1.5">
      <Tabs
        value={objective}
        onValueChange={(value) => setObjective(value as TrainingObjective)}
        className="contents"
      >
        <SegmentedTabsList
          value={objective}
          ariaLabel={t("rl.objective.ariaLabel")}
          size="compact"
          options={[
            { value: "sft", label: t("rl.objective.sft") },
            { value: "dpo", label: t("rl.objective.dpo"), disabled: rlLocked },
            {
              value: "orpo",
              label: t("rl.objective.orpo"),
              disabled: rlLocked,
            },
            {
              value: "grpo",
              label: t("rl.objective.grpo"),
              disabled: rlLocked,
            },
          ]}
        />
      </Tabs>
      <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
        {rlLocked ? lockedReason : t(HINT_KEYS[objective])}
      </p>
    </div>
  );
}
