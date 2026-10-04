// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  GRPO_DEFAULT_SYSTEM_PROMPT,
  GRPO_REASONING_SYSTEM_PROMPT,
} from "@/config/training";
import {
  previewCell,
  ruleTags,
  useRlWorkspaceStore,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Brain02Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";

type Preset = "notebook" | "none" | "custom";

// Matches WARMUP_MIN_SEQ_LENGTH in studio/backend/core/training/rl_format.py.
const WARMUP_MIN_CONTEXT = 2048;

function presetOf(prompt: string, notebook: string): Preset {
  if (prompt === notebook) {
    return "notebook";
  }
  return prompt.trim() === "" ? "none" : "custom";
}

export function PromptFormatSection(): ReactElement {
  const t = useT();
  const s = useTrainingConfigStore(
    useShallow((state) => ({
      prompt: state.grpoSystemPrompt,
      setPrompt: state.setGrpoSystemPrompt,
      thinking: state.grpoEnableThinking,
      setThinking: state.setGrpoEnableThinking,
      reasoning: state.grpoReasoningFormat,
      setReasoning: state.setGrpoReasoningFormat,
      warmupSteps: state.grpoFormatWarmupSteps,
      contextLength: state.contextLength,
      setWarmupSteps: state.setGrpoFormatWarmupSteps,
      rewards: state.grpoRewards,
      mapping: state.rlRoleMapping,
    })),
  );
  const { library, previewRow } = useRlWorkspaceStore(
    useShallow((state) => ({
      library: state.library,
      previewRow: state.previewRow,
    })),
  );
  const [editing, setEditing] = useState(false);
  const notebookPrompt = s.reasoning
    ? GRPO_REASONING_SYSTEM_PROMPT
    : GRPO_DEFAULT_SYSTEM_PROMPT;
  const preset = presetOf(s.prompt, notebookPrompt);
  const showEditor = preset === "custom" || editing;

  const promptColumn = Object.entries(s.mapping).find(
    ([, role]) => role === "prompt",
  )?.[0];
  const userText = promptColumn ? previewCell(previewRow?.[promptColumn]) : "";

  const selected = new Set(s.rewards.map((r) => r.name));
  const tags = [
    ...new Set(
      library
        .filter((r) => selected.has(r.name) && !r.shadowed)
        .flatMap((r) => ruleTags(r.rule)),
    ),
  ];
  const missingTags = tags.filter((tag) => !s.prompt.includes(tag));

  const choose = (value: string) => {
    if (value === "notebook") {
      s.setPrompt(notebookPrompt);
      setEditing(false);
    } else if (value === "none") {
      s.setPrompt("");
      setEditing(false);
    } else if (value === "custom") {
      // Start from the notebook text rather than an empty box that looks filled in.
      if (s.prompt.trim() === "") {
        s.setPrompt(notebookPrompt);
      }
      setEditing(true);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-xl border border-border/70 p-3 text-ui-11p5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-medium text-foreground">
              {t("rl.prompt.reasoningFormat")}
            </p>
            <p className="text-muted-foreground/85">
              {t("rl.prompt.reasoningFormatHint")}
            </p>
          </div>
          <Switch
            checked={s.reasoning}
            onCheckedChange={s.setReasoning}
            aria-label={t("rl.prompt.reasoningFormat")}
          />
        </div>
        {s.reasoning && (
          <div className="flex items-start justify-between gap-3 border-t border-border/60 pt-3">
            <div className="min-w-0">
              <p className="font-medium text-foreground">
                {t("rl.prompt.warmupSteps")}
              </p>
              <p className="text-muted-foreground/85">
                {t("rl.prompt.warmupStepsHint")}
              </p>
              {s.warmupSteps > 0 && s.contextLength < WARMUP_MIN_CONTEXT && (
                <p className="mt-1 flex items-start gap-1.5 text-amber-700 dark:text-amber-300">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-px size-3.5 shrink-0"
                  />
                  {t("rl.prompt.warmupContext", { n: WARMUP_MIN_CONTEXT })}
                </p>
              )}
            </div>
            <Input
              type="number"
              inputMode="numeric"
              min={0}
              max={1000}
              step={10}
              aria-label={t("rl.prompt.warmupSteps")}
              className="h-8 w-[calc(110px*var(--ui-space-scale,1))] shrink-0 self-center text-right text-xs"
              value={s.warmupSteps}
              onChange={(e) => {
                const n = Number.parseInt(e.target.value, 10);
                s.setWarmupSteps(
                  Number.isFinite(n) ? Math.min(1000, Math.max(0, n)) : 0,
                );
              }}
            />
          </div>
        )}
      </div>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={showEditor ? "custom" : preset}
        onValueChange={(value) => {
          // Radix clears on re-click; ignore empty so one stays selected.
          if (value) {
            choose(value);
          }
        }}
        aria-label={t("rl.prompt.presetLabel")}
        className="w-full"
      >
        <ToggleGroupItem
          value="notebook"
          className="flex-1 data-[state=on]:border-primary/40 data-[state=on]:bg-primary/10 data-[state=on]:text-primary"
        >
          {t("rl.prompt.presets.notebook")}
        </ToggleGroupItem>
        <ToggleGroupItem
          value="none"
          className="flex-1 data-[state=on]:border-primary/40 data-[state=on]:bg-primary/10 data-[state=on]:text-primary"
        >
          {t("rl.prompt.presets.none")}
        </ToggleGroupItem>
        <ToggleGroupItem
          value="custom"
          className="flex-1 data-[state=on]:border-primary/40 data-[state=on]:bg-primary/10 data-[state=on]:text-primary"
        >
          {t("rl.prompt.presets.custom")}
        </ToggleGroupItem>
      </ToggleGroup>

      {showEditor && (
        <Textarea
          rows={6}
          aria-label={t("rl.prompt.systemPrompt")}
          className="font-mono text-xs"
          value={s.prompt}
          onChange={(e) => s.setPrompt(e.target.value)}
        />
      )}

      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(0,16rem)]">
        <div className="flex min-w-0 flex-col gap-2 rounded-xl border border-border/70 bg-muted/30 p-3">
          <p className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground/70">
            {t("rl.prompt.previewTitle")}
          </p>
          <div className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-2 gap-y-1.5 font-mono text-ui-11p5">
            <span className="text-violet-700 dark:text-violet-300">
              {t("rl.prompt.system")}
            </span>
            <span
              className={cn(
                "whitespace-pre-wrap break-words",
                s.prompt.trim() === "" && "text-muted-foreground italic",
              )}
            >
              {s.prompt.trim() === "" ? t("rl.prompt.noSystem") : s.prompt}
            </span>
            <span className="text-sky-700 dark:text-sky-300">
              {t("rl.prompt.user")}
            </span>
            <span className="line-clamp-3 break-words text-foreground/85">
              {userText || t("rl.prompt.noRow")}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-3 text-ui-11p5">
          {tags.length > 0 &&
            (missingTags.length === 0 ? (
              <p className="flex items-start gap-1.5 text-emerald-700 dark:text-emerald-300">
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  className="mt-px size-3.5 shrink-0"
                />
                {t("rl.prompt.tagsOk", { tags: tags.join(" ") })}
              </p>
            ) : (
              <p className="flex items-start gap-1.5 text-amber-700 dark:text-amber-300">
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="mt-px size-3.5 shrink-0"
                />
                {t("rl.prompt.tagsMissing", { tags: missingTags.join(" ") })}
              </p>
            ))}
          <div className="flex items-start justify-between gap-3">
            <div className="flex min-w-0 items-start gap-1.5">
              <HugeiconsIcon
                icon={Brain02Icon}
                className="mt-px size-3.5 shrink-0 text-muted-foreground"
              />
              <div>
                <p className="font-medium text-foreground">
                  {t("rl.params.thinking")}
                </p>
                <p className="text-muted-foreground/85">
                  {t("rl.prompt.thinkingHint")}
                </p>
              </div>
            </div>
            <Switch
              checked={s.thinking}
              disabled={s.reasoning}
              onCheckedChange={s.setThinking}
              aria-label={t("rl.params.thinking")}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
