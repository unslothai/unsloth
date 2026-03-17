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
import { MODEL_TYPES } from "@/config/training";
import { cn } from "@/lib/utils";
import { useTrainingConfigStore } from "@/features/training";
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

export function ModelTypeStep(): ReactElement {
  const { modelType, setModelType } = useTrainingConfigStore(
    useShallow((s) => ({
      modelType: s.modelType,
      setModelType: s.setModelType,
    })),
  );
  const [chatOnlySelected, setChatOnlySelected] = useState(false);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold">Welcome to Unsloth Studio</h2>
        <p className="text-sm text-muted-foreground">
          Choose a path - fine-tune LLMs, vision, embedding, audio models or just chat.{" "}
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
        value={chatOnlySelected ? "" : (modelType ?? "")}
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
          const isSelected = !chatOnlySelected && modelType === type.value;
          const isDisabled = COMING_SOON.includes(type.value);
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
                    "hover:ring-primary/40 hover:-translate-y-0.5 hover:shadow-sm",
                  isSelected &&
                    !isDisabled &&
                    "ring-2 ring-primary -translate-y-0.5 shadow-sm",
                )}
              >
                {isDisabled && (
                  <Badge
                    variant="secondary"
                    className="absolute top-2 right-2 text-[10px]"
                  >
                    Coming Soon
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
                      !isDisabled &&
                        isSelected &&
                        "bg-primary/10 text-primary scale-105",
                      !(isDisabled || isSelected) &&
                        "bg-muted text-muted-foreground",
                    )}
                  >
                    <HugeiconsIcon
                      icon={Icon}
                      className={cn(
                        "size-5 transition-transform duration-100 ease-out",
                        isSelected && !isDisabled && "scale-110",
                      )}
                      strokeWidth={isSelected && !isDisabled ? 2.5 : 2}
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
        <div
          className="cursor-pointer"
          onClick={() => {
            setChatOnlySelected(true);
            setModelType("text" as ModelType);
            sessionStorage.setItem("unsloth_chat_only", "1");
          }}
        >
          <Card
            size="sm"
            className={cn(
              "relative shadow-primary/30 transition-all duration-150 ease-out",
              "hover:ring-primary/40 hover:-translate-y-0.5 hover:shadow-sm",
              chatOnlySelected && "ring-2 ring-primary -translate-y-0.5 shadow-sm",
            )}
          >
            <CardContent className="flex items-center gap-4 py-4">
              {/* Invisible spacer matching RadioGroupItem (size-4 flex) in other cards */}
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
                      <button
                        type="button"
                        className="text-muted-foreground/50 hover:text-muted-foreground"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <HugeiconsIcon
                          icon={InformationCircleIcon}
                          className="size-3.5"
                        />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>
                      Chat with any model. Has tool calling, web search and more.
                    </TooltipContent>
                  </Tooltip>
                </div>
                <span className="text-xs text-muted-foreground">
                  Chat with LLMs & vision models + audio generation.
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </RadioGroup>
    </div>
  );
}
