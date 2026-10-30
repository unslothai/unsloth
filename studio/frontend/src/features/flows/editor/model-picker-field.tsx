// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listModels } from "@/features/chat";
import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
} from "@/features/model-picker/components/model-selector/model-catalog";
import {
  IMAGE_GEN_TASKS,
  VIDEO_GEN_TASKS,
} from "@/features/model-picker/components/model-selector/pickers";
import { type ModelOption, ModelSelector } from "@/features/model-picker";
import { usePlatformStore } from "@/config/env";
import { priorityTrainingModels } from "@/config/training";
import { type ComponentProps, useEffect, useState } from "react";
import { choiceFromPick, isTrainableRepo } from "./model-choice";
import type { ModelChoice } from "../types";

type SelectorProps = ComponentProps<typeof ModelSelector>;

/** Each Flows model kind opens Studio's picker on the page that owns that kind. */
const KIND_PROPS: Record<string, Partial<SelectorProps>> = {
  chat: { hubCapability: "text" },
  // Fine-tune's base model: a GGUF can't be trained, so the picker leaves them out.
  train: { hubCapability: "text", rowFilter: (row) => isTrainableRepo(row.id) },
  vision: { hubCapability: "vision" },
  image: {
    task: IMAGE_GEN_TASKS,
    catalog: IMAGE_CATALOG,
    hubCapability: "image-gen",
  },
  video: {
    task: VIDEO_GEN_TASKS,
    catalog: VIDEO_CATALOG,
    hubCapability: "video-gen",
  },
  tts: { task: ["text-to-speech", "text-to-audio"], hubCapability: "audio" },
  audio: { task: ["text-to-audio"], hubCapability: "audio" },
};

let chatModels: Promise<ModelOption[]> | null = null;
function loadChatModels(): Promise<ModelOption[]> {
  chatModels ??= listModels()
    .then((body) =>
      body.models
        .filter((model) => !model.is_lora)
        .map((model) => ({
          id: model.id,
          name: model.name || model.id,
          isGguf: Boolean(model.is_gguf),
          isVision: Boolean(model.is_vision),
        })),
    )
    .catch(() => {
      chatModels = null;
      return [];
    });
  return chatModels;
}

/** Studio's model picker (search, quants, downloaded state) on a Flows model param. Picking never loads. */
export function FlowModelPicker({
  kind,
  choice,
  onChange,
}: {
  kind: string;
  choice: ModelChoice;
  onChange: (choice: ModelChoice) => void;
}) {
  const textKind = kind === "chat" || kind === "vision" || kind === "train";
  const [models, setModels] = useState<ModelOption[]>([]);
  const deviceType = usePlatformStore((s) => s.deviceType);
  useEffect(() => {
    if (!textKind) return;
    let alive = true;
    void loadChatModels().then((all) => {
      if (!alive) return;
      const vision = all.filter((m) => (m as { isVision?: boolean }).isVision);
      // Chat's list is mostly GGUF, so Recommended leads with the Train page's own picks.
      if (kind === "train")
        setModels(trainModels(deviceType, all));
      else setModels(kind === "vision" && vision.length ? vision : all);
    });
    return () => {
      alive = false;
    };
  }, [kind, textKind, deviceType]);
  return (
    <ModelSelector
      models={models}
      value={choice.repo}
      loaded={false}
      selectedGgufVariant={choice.variant ?? null}
      onValueChange={(id, meta) => onChange(choiceFromPick(id, meta))}
      variant="outline"
      size="sm"
      placeholder={`Choose a ${kind === "tts" ? "speech" : kind === "train" ? "base" : kind} model`}
      className="nodrag h-8 w-full max-w-full justify-between rounded-md px-2.5 text-ui-12"
      contentClassName="nodrag nowheel"
      {...KIND_PROPS[kind]}
    />
  );
}

function trainModels(
  deviceType: string | null | undefined,
  local: ModelOption[],
): ModelOption[] {
  const seen = new Set<string>();
  const out: ModelOption[] = [];
  for (const m of [
    ...priorityTrainingModels(deviceType).map((id) => ({ id, name: id })),
    ...local.filter((m) => !(m as { isGguf?: boolean }).isGguf),
  ]) {
    const key = m.id.toLowerCase();
    if (!isTrainableRepo(m.id) || seen.has(key)) continue;
    seen.add(key);
    out.push(m);
  }
  return out;
}
