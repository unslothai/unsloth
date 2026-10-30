// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionField } from "@/components/section-box";
import { Input } from "@/components/ui/input";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Switch } from "@/components/ui/switch";
import { Tabs } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  GRPO_DEFAULT_SYSTEM_PROMPT,
  GRPO_FORMAT_WARMUP_MIN_CONTEXT,
  GRPO_REASONING_SYSTEM_PROMPT,
} from "@/config/training";
import {
  configuredSystemPrompt,
  effectiveSystemPrompt,
  previewCell,
  ruleTags,
  useRlWorkspaceStore,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Alert02Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { FieldHint } from "../field-hint";
import { ParamsRow } from "../params-section-controls";
import { RL_PANEL_CLASS } from "./rl-panel";

type Preset = "notebook" | "none" | "custom";

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
  // What training gets: with the reasoning format on, an empty prompt is the reasoning one.
  const configured = configuredSystemPrompt(
    s.prompt,
    s.reasoning,
    GRPO_REASONING_SYSTEM_PROMPT,
  );
  const preset = presetOf(configured, notebookPrompt);
  const showEditor = preset === "custom" || editing;
  const active: Preset = showEditor ? "custom" : preset;
  // The preset is the reasoning format's prompt while that is on.
  const presetLabel = (option: Preset) =>
    t(
      option === "notebook" && s.reasoning
        ? "rl.prompt.presets.reasoningTags"
        : `rl.prompt.presets.${option}`,
    );

  const promptColumn = Object.entries(s.mapping).find(
    ([, role]) => role === "prompt",
  )?.[0];
  const userText = promptColumn ? previewCell(previewRow?.[promptColumn]) : "";
  // A row's own system message wins in training, so the preview and tag check use it too.
  const system = effectiveSystemPrompt(previewRow, s.mapping, configured);

  const selected = new Set(s.rewards.map((r) => r.name));
  const tags = [
    ...new Set(
      library
        .filter((r) => selected.has(r.name) && !r.shadowed)
        .flatMap((r) => ruleTags(r.rule)),
    ),
  ];
  const missingTags = tags.filter((tag) => !system.includes(tag));

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
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3">
        <ParamsRow
          label={t("rl.prompt.reasoningFormat")}
          tooltip={t("rl.prompt.reasoningFormatHint")}
        >
          <Switch
            checked={s.reasoning}
            onCheckedChange={s.setReasoning}
            aria-label={t("rl.prompt.reasoningFormat")}
          />
        </ParamsRow>
        {s.reasoning && (
          <ParamsRow
            label={t("rl.prompt.warmupSteps")}
            tooltip={t("rl.prompt.warmupStepsHint")}
          >
            <Input
              type="number"
              inputMode="numeric"
              min={0}
              max={1000}
              step={10}
              aria-label={t("rl.prompt.warmupSteps")}
              className="w-28 font-mono"
              value={s.warmupSteps}
              onChange={(e) => {
                const n = Number.parseInt(e.target.value, 10);
                s.setWarmupSteps(
                  Number.isFinite(n) ? Math.min(1000, Math.max(0, n)) : 0,
                );
              }}
            />
          </ParamsRow>
        )}
        {s.reasoning &&
          s.warmupSteps > 0 &&
          s.contextLength < GRPO_FORMAT_WARMUP_MIN_CONTEXT && (
            <p className="flex items-start gap-1.5 text-ui-11p5 leading-ui-15 text-amber-700 dark:text-amber-300">
              <HugeiconsIcon icon={Alert02Icon} className="size-3.5 shrink-0" />
              {t("rl.prompt.warmupContext", {
                n: GRPO_FORMAT_WARMUP_MIN_CONTEXT,
              })}
            </p>
          )}
      </div>

      <SectionField
        label={t("rl.prompt.systemPrompt")}
        hint={
          <FieldHint
            text={t("rl.prompt.presetHint")}
            label={t("rl.prompt.systemPrompt")}
          />
        }
      >
        <div className="flex flex-col gap-3">
          {/* The same control and hint line as the Objective picker in the Train setup. */}
          <div className="flex flex-col gap-1.5">
            <Tabs value={active} onValueChange={choose} className="contents">
              <SegmentedTabsList
                value={active}
                ariaLabel={t("rl.prompt.presetLabel")}
                size="compact"
                options={[
                  { value: "notebook", label: presetLabel("notebook") },
                  {
                    value: "none",
                    label: presetLabel("none"),
                    // Training adds the reasoning prompt to an empty one.
                    disabled: s.reasoning,
                  },
                  { value: "custom", label: presetLabel("custom") },
                ]}
              />
            </Tabs>
            <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
              {t(
                active === "notebook" && s.reasoning
                  ? "rl.prompt.presetHints.reasoningTags"
                  : `rl.prompt.presetHints.${active}`,
              )}
            </p>
          </div>
          {showEditor && (
            <Textarea
              rows={6}
              aria-label={t("rl.prompt.systemPrompt")}
              className="font-mono text-xs"
              value={s.prompt}
              onChange={(e) => s.setPrompt(e.target.value)}
            />
          )}
        </div>
      </SectionField>

      <SectionField label={t("rl.prompt.previewTitle")}>
        <div className={RL_PANEL_CLASS}>
          <div className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-3 gap-y-2 font-mono text-ui-11p5 leading-ui-15">
            <span className="text-violet-700 dark:text-violet-300">
              {t("rl.prompt.system")}
            </span>
            <span
              className={cn(
                "whitespace-pre-wrap break-words text-foreground",
                system.trim() === "" && "text-muted-foreground italic",
              )}
            >
              {system.trim() === "" ? t("rl.prompt.noSystem") : system}
            </span>
            <span className="text-sky-700 dark:text-sky-300">
              {t("rl.prompt.user")}
            </span>
            <span className="line-clamp-3 break-words text-foreground/85">
              {userText || t("rl.prompt.noRow")}
            </span>
          </div>
          {tags.length > 0 &&
            (missingTags.length === 0 ? (
              <p className="flex items-start gap-1.5 border-t border-border/60 pt-3 text-ui-11p5 leading-ui-15 text-emerald-700 dark:text-emerald-300">
                <HugeiconsIcon
                  icon={CheckmarkCircle02Icon}
                  className="size-3.5 shrink-0"
                />
                {t("rl.prompt.tagsOk", { tags: tags.join(" ") })}
              </p>
            ) : (
              <p className="flex items-start gap-1.5 border-t border-border/60 pt-3 text-ui-11p5 leading-ui-15 text-amber-700 dark:text-amber-300">
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="size-3.5 shrink-0"
                />
                {t("rl.prompt.tagsMissing", { tags: missingTags.join(" ") })}
              </p>
            ))}
        </div>
      </SectionField>

      <ParamsRow
        label={t("rl.params.thinking")}
        tooltip={t("rl.prompt.thinkingHint")}
      >
        <Switch
          checked={s.thinking}
          disabled={s.reasoning}
          onCheckedChange={s.setThinking}
          aria-label={t("rl.params.thinking")}
        />
      </ParamsRow>
    </div>
  );
}
