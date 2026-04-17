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
  vision: "微调可同时理解图像与文本的模型",
  audio: "微调文本转语音与音频模型",
  embeddings: "微调用于语义检索与相似度的模型",
  text: "微调用于文本生成的大语言模型",
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
        <h2 className="text-lg font-semibold">欢迎使用 Unsloth Studio</h2>
        <p className="text-sm text-muted-foreground">
          选择你的路径：微调 LLM、视觉、嵌入、音频模型，或直接聊天。{" "}
          <a
            href="https://unsloth.ai/docs/new/studio/start"
            target="_blank"
            rel="noreferrer"
            className="text-primary underline"
          >
            查看快速上手指南
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
                    即将推出
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
                  <span className="font-medium">聊天</span>
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
                      与任意模型聊天，支持工具调用、联网搜索等能力。
                    </TooltipContent>
                  </Tooltip>
                </div>
                <span className="text-xs text-muted-foreground">
                  与 LLM、视觉模型聊天，并支持音频生成。
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </RadioGroup>
    </div>
  );
}
