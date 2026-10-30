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
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { TrainModelSelector } from "@/features/train-model-picker";
import { type ComponentProps, useEffect, useState } from "react";
import { choiceFromPick } from "./model-choice";
import type { ModelChoice } from "../types";

type SelectorProps = ComponentProps<typeof ModelSelector>;

/** Each Flows model kind opens Studio's picker on the page that owns that kind. */
const KIND_PROPS: Record<string, Partial<SelectorProps>> = {
  chat: { hubCapability: "text" },
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
  const textKind = kind === "chat" || kind === "vision";
  const [models, setModels] = useState<ModelOption[]>([]);
  // Chat's check mark, only when this exact repo (and quant) is the server's resident model.
  const resident = useChatRuntimeStore(
    (s) =>
      Boolean(choice.repo) &&
      s.residentCheckpoint?.toLowerCase() === choice.repo.toLowerCase() &&
      (!choice.variant ||
        s.activeGgufVariant?.toLowerCase() === choice.variant.toLowerCase()),
  );
  useEffect(() => {
    if (!textKind) return;
    let alive = true;
    void loadChatModels().then((all) => {
      if (!alive) return;
      const vision = all.filter((m) => (m as { isVision?: boolean }).isVision);
      setModels(kind === "vision" && vision.length ? vision : all);
    });
    return () => {
      alive = false;
    };
  }, [kind, textKind]);
  return (
    <ModelSelector
      models={models}
      value={choice.repo}
      loaded={resident}
      loadedMark="check"
      // Names the picked quant in the trigger ("GGUF · Q4_K_M"), as Chat does for the loaded one.
      activeGgufVariant={choice.variant ?? null}
      selectedGgufVariant={choice.variant ?? null}
      onValueChange={(id, meta) => onChange(choiceFromPick(id, meta))}
      variant="ghost"
      size="sm"
      placeholder={`Choose a ${kind === "tts" ? "speech" : kind} model`}
      className="nodrag -ml-2 h-8 w-fit max-w-[calc(100%+0.5rem)] !pl-2 !pr-1.5"
      contentClassName="nodrag nowheel"
      {...KIND_PROPS[kind]}
    />
  );
}

/** Fine-tune's base model: the Train page's own picker (trainable weights, VRAM fit), kept as a plain repo string. */
export function FlowTrainModelPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (model: string) => void;
}) {
  return <TrainModelSelector value={value} onValueChange={onChange} />;
}
