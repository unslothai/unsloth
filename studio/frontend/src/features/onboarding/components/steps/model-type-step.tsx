// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { MODEL_TYPES } from "@/config/training";
import {
  isTrainingModelTypeSupportedOnDevice,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ModelType } from "@/types/training";
import {
  BubbleChatIcon,
  Database02Icon,
  ImageIcon,
  InformationCircleIcon,
  TextIcon,
  VoiceIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const TYPE_ICONS: Record<ModelType, typeof ImageIcon> = {
  vision: ImageIcon,
  audio: VoiceIcon,
  embeddings: Database02Icon,
  text: TextIcon,
};

const TYPE_TOOLTIPS: Record<ModelType, string> = {
  vision: "Fine-tune models that understand images and text together",
  audio: "Fine-tune text-to-speech and audio models",
  embeddings: "Fine-tune models for semantic search and similarity",
  text: "Fine-tune large language models for text generation",
};

const COMING_SOON: ModelType[] = [];

function getDisabledLabel(
  modelType: ModelType,
  deviceType: string,
  unsupportedOnDeviceLabel: string,
): string | null {
  if (COMING_SOON.includes(modelType)) {
    return "Coming Soon";
  }
  if (!isTrainingModelTypeSupportedOnDevice(modelType, deviceType)) {
    return unsupportedOnDeviceLabel;
  }
  return null;
}

export function ModelTypeStep(): ReactElement {
  const t = useT();
  const deviceType = usePlatformStore((state) => state.deviceType);
  const { modelType, setModelType } = useTrainingConfigStore(
    useShallow((s) => ({
      modelType: s.modelType,
      setModelType: s.setModelType,
    })),
  );
  const [chatOnlySelected, setChatOnlySelected] = useState(false);
  const selectedModelType = chatOnlySelected ? null : modelType;
  const selectChatOnly = () => {
    setChatOnlySelected(true);
    setModelType("text");
    sessionStorage.setItem("unsloth_chat_only", "1");
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold">Welcome to Unsloth</h2>
        <p className="text-sm text-muted-foreground">
          Choose a path - fine-tune LLMs, vision, embedding, audio models or
          just chat.{" "}
          <a
            href="https://unsloth.ai/docs/new/studio/start"
            target="_blank"
            rel="noreferrer"
            className="text-primary underline"
          >
            Get started with our guide
          </a>
        </p>
      </div>
      <RadioGroup
        value={selectedModelType ?? ""}
        onValueChange={(v) => {
          if (!COMING_SOON.includes(v as ModelType)) {
            setChatOnlySelected(false);
            sessionStorage.removeItem("unsloth_chat_only");
            setModelType(v as ModelType);
          }
        }}
        className="grid grid-cols-2 gap-4"
      >
        {MODEL_TYPES.map((type) => {
          const Icon = TYPE_ICONS[type.value];
          const isSelected = selectedModelType === type.value;
          const disabledLabel = getDisabledLabel(
            type.value,
            deviceType,
            t("studio.params.notSupportedAppleSilicon"),
          );
          const isDisabled = disabledLabel !== null;
          const isActive = isSelected && !isDisabled;
          const inputId = `model-type-${type.value}`;

          return (
            <label
              key={type.value}
              htmlFor={inputId}
              className={cn(
                isDisabled ? "cursor-not-allowed" : "cursor-pointer",
              )}
            >
              <Card
                size="sm"
                className={cn(
                  "relative shadow-primary/30 transition-all duration-150 ease-out",
                  isDisabled && "opacity-50 bg-muted/50",
                  !isDisabled &&
                    "hover:ring-ring hover:-translate-y-0.5 hover:shadow-sm",
                  isActive &&
                    "ring-1 ring-ring-strong -translate-y-0.5 shadow-sm",
                )}
              >
                {isDisabled && (
                  <Badge
                    variant="secondary"
                    className="absolute top-2 right-2 text-ui-10"
                  >
                    {disabledLabel}
                  </Badge>
                )}
                <CardContent className="flex items-center gap-4 py-4">
                  <RadioGroupItem
                    id={inputId}
                    value={type.value}
                    className="sr-only"
                    disabled={isDisabled}
                  />
                  <div
                    className={cn(
                      "size-10 rounded-xl corner-squircle flex items-center justify-center shrink-0",
                      "transition-all duration-100 ease-out",
                      isDisabled && "bg-muted/50 text-muted-foreground/50",
                      isActive && "bg-primary/10 text-primary scale-105",
                      !(isDisabled || isSelected) &&
                        "bg-muted text-muted-foreground",
                    )}
                  >
                    <HugeiconsIcon
                      icon={Icon}
                      className={cn(
                        "size-5 transition-transform duration-100 ease-out",
                        isActive && "scale-110",
                      )}
                      strokeWidth={isActive ? 2.5 : 2}
                    />
                  </div>
                  <div className="flex flex-col gap-0.5 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span
                        className={cn(
                          "font-medium",
                          isDisabled && "text-muted-foreground",
                        )}
                      >
                        {type.label}
                      </span>
                      <Tooltip>
                        <TooltipTrigger asChild={true}>
                          <button
                            type="button"
                            className="text-muted-foreground/50 hover:text-muted-foreground"
                          >
                            <HugeiconsIcon
                              icon={InformationCircleIcon}
                              className="size-3.5"
                            />
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          {TYPE_TOOLTIPS[type.value]}
                        </TooltipContent>
                      </Tooltip>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {type.description}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </label>
          );
        })}
        <button
          type="button"
          className="cursor-pointer text-left"
          onClick={selectChatOnly}
        >
          <Card
            size="sm"
            className={cn(
              "relative shadow-primary/30 transition-all duration-150 ease-out",
              "hover:ring-ring hover:-translate-y-0.5 hover:shadow-sm",
              chatOnlySelected &&
                "ring-1 ring-ring-strong -translate-y-0.5 shadow-sm",
            )}
          >
            <CardContent className="flex items-center gap-4 py-4">
              {/* Invisible spacer matching RadioGroupItem (size-4) in other cards */}
              <div className="size-4 shrink-0" aria-hidden="true" />
              <div
                className={cn(
                  "size-10 rounded-xl corner-squircle flex items-center justify-center shrink-0",
                  "transition-all duration-100 ease-out",
                  chatOnlySelected
                    ? "bg-primary/10 text-primary scale-105"
                    : "bg-muted text-muted-foreground",
                )}
              >
                <HugeiconsIcon
                  icon={BubbleChatIcon}
                  className={cn(
                    "size-5 transition-transform duration-100 ease-out",
                    chatOnlySelected && "scale-110",
                  )}
                  strokeWidth={chatOnlySelected ? 2.5 : 2}
                />
              </div>
              <div className="flex flex-col gap-0.5 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="font-medium">Chat</span>
                  <Tooltip>
                    <TooltipTrigger asChild={true}>
                      <span className="text-muted-foreground/50 hover:text-muted-foreground">
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      Chat with any model. Has tool calling, web search and
                      more.
                    </TooltipContent>
                  </Tooltip>
                </div>
                <span className="text-xs text-muted-foreground">
                  Chat with LLMs & vision models + audio generation.
                </span>
              </div>
            </CardContent>
          </Card>
        </button>
      </RadioGroup>
    </div>
  );
}
