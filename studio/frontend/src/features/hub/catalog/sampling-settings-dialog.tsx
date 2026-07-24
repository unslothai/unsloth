// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Gear button for the GGUF run bar: model config, system prompt, reasoning,
// sampling, tools and retrieval, using the same controls as the chat page's
// Run settings. Edits write to the chat runtime store's persisted state.

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { InfoHint } from "@/components/ui/info-hint";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ParamSlider,
  useChatModelRuntime,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  type PerModelConfig,
  SidebarModelConfig,
  applyPerModelConfigToRuntime,
  currentRuntimePerModelConfig,
  useActiveModelConfig,
} from "@/features/model-picker";
import { RetrievalSettingsSection } from "@/features/rag";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Settings02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useState } from "react";
import { HubOptionMenu } from "./hub-option-menu";

function SettingsSection({
  label,
  labelClassName,
  children,
}: {
  label: string;
  labelClassName?: string;
  children: ReactNode;
}) {
  return (
    <div className="border-t border-border/50 pb-5 pt-5 first:border-t-0 first:pt-0">
      <div
        className={cn(
          "pb-4 text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground",
          labelClassName,
        )}
      >
        {label}
      </div>
      <div className="flex flex-col gap-5">{children}</div>
    </div>
  );
}

function ToggleRow({
  label,
  info,
  checked,
  disabled,
  onCheckedChange,
}: {
  label: string;
  info?: string;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="text-ui-13 font-medium text-nav-fg">{label}</span>
        {info && <InfoHint>{info}</InfoHint>}
      </div>
      <Switch
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => onCheckedChange(value === true)}
        aria-label={label}
      />
    </div>
  );
}

export function SamplingSettingsButton({ className }: { className?: string }) {
  const [open, setOpen] = useState(false);
  const params = useChatRuntimeStore((s) => s.params);
  const setParams = useChatRuntimeStore((s) => s.setParams);

  // Loaded model's per-model config (context length etc.), mirroring the chat
  // page's Run settings Model section.
  const { selectModel } = useChatModelRuntime();
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const activeGgufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const ggufNativeContextLength = useChatRuntimeStore(
    (s) => s.ggufNativeContextLength,
  );
  const {
    checkpoint,
    isGguf: activeModelIsGguf,
    config: activeModelConfig,
  } = useActiveModelConfig();
  const handleReloadActiveModel = useCallback(
    (config: PerModelConfig) => {
      const runtime = useChatRuntimeStore.getState();
      const activeCheckpoint = runtime.params.checkpoint;
      if (!activeCheckpoint) return;
      const nativeToken = runtime.activeNativePathToken;
      const nativeExpiry = runtime.activeNativePathExpiresAtMs;
      // Mirrors the chat page: an expired native-path token can't reload.
      if (nativeToken && nativeExpiry != null && Date.now() >= nativeExpiry) {
        toast.error("This local model file's access has expired.", {
          description: "Re-select the model file to reload it.",
        });
        return;
      }
      // selectModel reads config from the runtime store, not the selection, so
      // apply it first (snapshotting the current one for rollback).
      const previousConfig = currentRuntimePerModelConfig({
        includeMaxSeqLength: true,
      });
      applyPerModelConfigToRuntime(config);
      void selectModel({
        id: activeCheckpoint,
        source: "local",
        ggufVariant: runtime.activeGgufVariant ?? undefined,
        nativePathToken: nativeToken ?? undefined,
        nativePathExpiresAtMs: nativeExpiry,
        isGguf: activeModelIsGguf,
        isDownloaded: true,
        keepSpeculative: true,
        previousConfig,
        forceReload: true,
      });
    },
    [selectModel, activeModelIsGguf],
  );

  // Reasoning + tools: same store bindings as the chat page.
  const reasoningEnabled = useChatRuntimeStore((s) => s.reasoningEnabled);
  const reasoningAlwaysOn = useChatRuntimeStore((s) => s.reasoningAlwaysOn);
  const setReasoningEnabled = useChatRuntimeStore((s) => s.setReasoningEnabled);
  const reasoningEffort = useChatRuntimeStore((s) => s.reasoningEffort);
  const reasoningEffortLevels = useChatRuntimeStore(
    (s) => s.reasoningEffortLevels,
  );
  const setReasoningEffort = useChatRuntimeStore((s) => s.setReasoningEffort);
  const supportsPreserveThinking = useChatRuntimeStore(
    (s) => s.supportsPreserveThinking,
  );
  const preserveThinking = useChatRuntimeStore((s) => s.preserveThinking);
  const setPreserveThinking = useChatRuntimeStore((s) => s.setPreserveThinking);
  const maxToolCalls = useChatRuntimeStore((s) => s.maxToolCallsPerMessage);
  const setMaxToolCalls = useChatRuntimeStore(
    (s) => s.setMaxToolCallsPerMessage,
  );
  const toolCallTimeout = useChatRuntimeStore((s) => s.toolCallTimeout);
  const setToolCallTimeout = useChatRuntimeStore((s) => s.setToolCallTimeout);
  const autoHealToolCalls = useChatRuntimeStore((s) => s.autoHealToolCalls);
  const setAutoHealToolCalls = useChatRuntimeStore(
    (s) => s.setAutoHealToolCalls,
  );
  const nudgeToolCalls = useChatRuntimeStore((s) => s.nudgeToolCalls);
  const setNudgeToolCalls = useChatRuntimeStore((s) => s.setNudgeToolCalls);
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const setConfirmToolCalls = useChatRuntimeStore((s) => s.setConfirmToolCalls);

  const toolsEnabled = useChatRuntimeStore((s) => s.toolsEnabled);
  const setToolsEnabled = useChatRuntimeStore((s) => s.setToolsEnabled);

  const set = (key: keyof typeof params) => (value: number) =>
    setParams({ ...params, [key]: value });

  // Slider 0-41; 41 maps to 9999 ("Max"), mirroring the chat page.
  const toolCallsSliderValue =
    maxToolCalls >= 9999 ? 41 : Math.min(maxToolCalls, 40);
  // Slider 1-31; 31 maps to 9999 ("Max").
  const timeoutSliderValue =
    toolCallTimeout >= 9999 ? 31 : Math.min(Math.max(toolCallTimeout, 1), 30);

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label="Inference settings"
            onClick={(e) => {
              e.stopPropagation();
              setOpen(true);
            }}
            className={cn(
              "inline-flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              className,
            )}
          >
            <HugeiconsIcon
              icon={Settings02Icon}
              strokeWidth={1.75}
              className="size-4"
            />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="tooltip-compact">
          Inference settings
        </TooltipContent>
      </Tooltip>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          className="flex max-h-[85vh] flex-col gap-5 sm:max-w-[580px]"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>Inference settings</DialogTitle>
            <DialogDescription>
              Applies to chats with local models.
            </DialogDescription>
          </DialogHeader>
          <div className="-mr-2 flex min-h-0 flex-col overflow-y-auto overflow-x-hidden pr-2 [scrollbar-width:thin]">
            {checkpoint && activeModelConfig && !modelLoading && (
              <SettingsSection label="Model">
                <SidebarModelConfig
                  modelId={checkpoint}
                  ggufVariant={activeGgufVariant ?? null}
                  isGguf={activeModelIsGguf}
                  nativeContextLength={ggufNativeContextLength}
                  loadedContextLength={ggufContextLength}
                  loadedConfig={activeModelConfig}
                  onReload={handleReloadActiveModel}
                />
              </SettingsSection>
            )}

            <SettingsSection label="System Prompt">
              <Textarea
                value={params.systemPrompt}
                onChange={(e) =>
                  setParams({ ...params, systemPrompt: e.target.value })
                }
                placeholder="Instructions sent before every conversation."
                aria-label="System prompt"
                className="min-h-[84px] resize-y text-ui-13"
              />
            </SettingsSection>

            <SettingsSection label="Reasoning" labelClassName="pb-5">
              <ToggleRow
                label="Enable reasoning"
                checked={reasoningAlwaysOn || reasoningEnabled}
                disabled={reasoningAlwaysOn}
                onCheckedChange={setReasoningEnabled}
              />
              {reasoningEffortLevels.length > 0 && (
                <div className="flex items-center justify-between gap-3">
                  <span className="text-ui-13 font-medium text-nav-fg">
                    Reasoning effort
                  </span>
                  <HubOptionMenu
                    value={reasoningEffort}
                    options={reasoningEffortLevels.map((level) => ({
                      value: level,
                      label: level.charAt(0).toUpperCase() + level.slice(1),
                    }))}
                    onValueChange={setReasoningEffort}
                    ariaLabel="Reasoning effort"
                    align="end"
                    className="h-8 text-ui-11p5"
                  />
                </div>
              )}
              {supportsPreserveThinking && (
                <ToggleRow
                  label="Preserve thinking"
                  info="Keep earlier turns' reasoning in context so the model can build on it. Uses more context."
                  checked={preserveThinking}
                  onCheckedChange={setPreserveThinking}
                />
              )}
            </SettingsSection>

            <SettingsSection label="Sampling">
              <ParamSlider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.01}
                onChange={set("temperature")}
                info="Controls randomness. Lower values make output focused and deterministic; higher values increase variety and creativity."
              />
              <ParamSlider
                label="Top P"
                value={params.topP}
                min={0}
                max={1}
                step={0.05}
                onChange={set("topP")}
                displayValue={params.topP === 1 ? "Off" : undefined}
                info="Nucleus sampling. Restricts choices to the smallest set of tokens whose cumulative probability reaches this threshold. 1.0 = off."
              />
              <ParamSlider
                label="Top K"
                value={params.topK}
                min={0}
                max={100}
                step={1}
                onChange={set("topK")}
                displayValue={params.topK === 0 ? "Off" : undefined}
                info="Limits sampling to the K most likely tokens at each step. 0 = off."
              />
              <ParamSlider
                label="Min P"
                value={params.minP}
                min={0}
                max={1}
                step={0.01}
                onChange={set("minP")}
                info="Drops tokens whose probability is below this fraction of the top token's probability. Filters unlikely candidates."
              />
              <ParamSlider
                label="Repetition Penalty"
                value={params.repetitionPenalty}
                min={1}
                max={2}
                step={0.05}
                onChange={set("repetitionPenalty")}
                displayValue={
                  params.repetitionPenalty === 1 ? "Off" : undefined
                }
                info="Down-weights tokens that have already appeared, reducing repetition. 1.0 = off; higher values penalize more strongly."
              />
              <ParamSlider
                label="Presence Penalty"
                value={params.presencePenalty}
                min={0}
                max={2}
                step={0.1}
                onChange={set("presencePenalty")}
                displayValue={params.presencePenalty === 0 ? "Off" : undefined}
                info="Penalizes any token that has already appeared at least once, encouraging the model to introduce new topics. 0 = off."
              />
              <ParamSlider
                label="Max Tokens"
                value={params.maxTokens}
                min={64}
                max={131072}
                step={64}
                onChange={set("maxTokens")}
                valueSize={6}
                info="Maximum number of tokens to generate per response. Generation stops at this limit or when the model emits an end-of-sequence token."
              />
            </SettingsSection>

            <SettingsSection label="Tools">
              <ToggleRow
                label="Enable tools"
                info="Master switch for tool use in chat. When off, the model answers without calling any tools."
                checked={toolsEnabled}
                onCheckedChange={setToolsEnabled}
              />
              <ToggleRow
                label="Auto-healing tool calls"
                info="Unsloth auto-fixes broken tool calls so inference output is never broken."
                checked={autoHealToolCalls}
                onCheckedChange={setAutoHealToolCalls}
              />
              <ToggleRow
                label="Nudge tool calls"
                info="When a tool call cannot be repaired, re-ask the model once so the intended tool still runs."
                checked={nudgeToolCalls}
                onCheckedChange={setNudgeToolCalls}
              />
              <ToggleRow
                label="Confirm tool calls"
                info="Pause every local tool call for your approval before it runs. Overridden by Full access."
                checked={permissionMode === "ask"}
                disabled={permissionMode === "full"}
                onCheckedChange={setConfirmToolCalls}
              />
              <ParamSlider
                label="Max Tool Calls Per Message"
                value={toolCallsSliderValue}
                min={0}
                max={41}
                step={1}
                onChange={(v) => setMaxToolCalls(v >= 41 ? 9999 : v)}
                displayValue={
                  toolCallsSliderValue >= 41
                    ? "Max"
                    : toolCallsSliderValue === 0
                      ? "Off"
                      : undefined
                }
                info="Cap on tool/function calls the model may invoke within a single response. 0 disables tool use; Max removes the cap."
              />
              <ParamSlider
                label="Max Tool Call Duration"
                value={timeoutSliderValue}
                min={1}
                max={31}
                step={1}
                onChange={(v) => setToolCallTimeout(v >= 31 ? 9999 : v)}
                displayValue={
                  timeoutSliderValue >= 31
                    ? "Max"
                    : timeoutSliderValue === 1
                      ? "1 minute"
                      : `${timeoutSliderValue} minutes`
                }
                valueSize={10}
                info="Per-call wall-clock limit. Long-running tool executions are terminated when this elapses; the model continues with what completed."
              />
            </SettingsSection>

            <SettingsSection label="Retrieval">
              <RetrievalSettingsSection />
            </SettingsSection>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
