// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionField } from "@/components/section-box";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Switch } from "@/components/ui/switch";
import { Tabs } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { GRPO_DEFAULT_SYSTEM_PROMPT } from "@/config/training";
import {
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

function presetOf(prompt: string): Preset {
  if (prompt === GRPO_DEFAULT_SYSTEM_PROMPT) {
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
  const preset = presetOf(s.prompt);
  const showEditor = preset === "custom" || editing;
  const active: Preset = showEditor ? "custom" : preset;
  const presetLabel = (option: Preset) => t(`rl.prompt.presets.${option}`);

  const promptColumn = Object.entries(s.mapping).find(
    ([, role]) => role === "prompt",
  )?.[0];
  const userText = promptColumn ? previewCell(previewRow?.[promptColumn]) : "";
  // A row's own system message wins in training, so the preview and tag check use it too.
  const system = effectiveSystemPrompt(previewRow, s.mapping, s.prompt);

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
      s.setPrompt(GRPO_DEFAULT_SYSTEM_PROMPT);
      setEditing(false);
    } else if (value === "none") {
      s.setPrompt("");
      setEditing(false);
    } else if (value === "custom") {
      // Start from the notebook text rather than an empty box that looks filled in.
      if (s.prompt.trim() === "") {
        s.setPrompt(GRPO_DEFAULT_SYSTEM_PROMPT);
      }
      setEditing(true);
    }
  };

  return (
    <div className="flex flex-col gap-5">
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
                  { value: "none", label: presetLabel("none") },
                  { value: "custom", label: presetLabel("custom") },
                ]}
              />
            </Tabs>
            <p className="text-ui-11p5 leading-ui-15 text-muted-foreground/85">
              {t(`rl.prompt.presetHints.${active}`)}
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
          onCheckedChange={s.setThinking}
          aria-label={t("rl.params.thinking")}
        />
      </ParamsRow>
    </div>
  );
}
