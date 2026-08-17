// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { TrainingMethod } from "@/types/training";

export const TRAINING_METHOD_ORDER: readonly TrainingMethod[] = [
  "qlora",
  "lora",
  "full",
  "cpt",
];

interface TrainingMethodMeta {
  labelKey: TranslationKey;
  hintKey: TranslationKey;
  noteKey: TranslationKey;
  dotClass: string;
}

export const TRAINING_METHOD_META: Record<TrainingMethod, TrainingMethodMeta> =
  {
    qlora: {
      labelKey: "studio.methods.qlora.label",
      hintKey: "studio.methods.qlora.hint",
      noteKey: "studio.methods.qlora.note",
      dotClass: "bg-emerald-500/70",
    },
    lora: {
      labelKey: "studio.methods.lora.label",
      hintKey: "studio.methods.lora.hint",
      noteKey: "studio.methods.lora.note",
      dotClass: "bg-sky-500/70",
    },
    full: {
      labelKey: "studio.methods.full.label",
      hintKey: "studio.methods.full.hint",
      noteKey: "studio.methods.full.note",
      dotClass: "bg-amber-500/70",
    },
    cpt: {
      labelKey: "studio.methods.cpt.label",
      hintKey: "studio.methods.cpt.hint",
      noteKey: "studio.methods.cpt.note",
      dotClass: "bg-violet-500/70",
    },
  };
