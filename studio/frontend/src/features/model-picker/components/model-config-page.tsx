// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { InfoHint } from "@/components/ui/info-hint";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useMemo, useState } from "react";
import { useDefaultChatTemplate } from "../hooks/use-model-defaults";
import { perModelConfigsEqual } from "../model-config/apply-per-model-config";
import {
  DEFAULT_PER_MODEL_CONFIG,
  MTP_SPECULATIVE_TYPES,
  type PerModelConfig,
  deletePerModelConfig,
  resolveInitialConfig,
  savePerModelConfig,
} from "../model-config/per-model-config";
import { ChatTemplateEditorDialog } from "./chat-template-editor-dialog";
import type { ModelPickTarget } from "./model-selector/types";

const ROW_CLASS = "flex min-h-7 items-center justify-between gap-4";
const LABEL_CLASS =
  "min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg";
const CONTROL_SURFACE =
  "rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1]";
const SELECT_TRIGGER_CLASS = `grid h-7 min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0`;
const NUMBER_INPUT_CLASS = `h-7 w-[92px] ${CONTROL_SURFACE} pl-3 pr-2 py-0 text-right text-[13px] font-medium text-nav-fg outline-none focus-visible:ring-0`;

interface ModelConfigPageProps {
  target: ModelPickTarget;
  onBack: () => void;
  onRun: (config: PerModelConfig, remember: boolean) => void;
  loadedConfig?: PerModelConfig | null;
}

export function ModelConfigPage({
  target,
  onBack,
  onRun,
  loadedConfig = null,
}: ModelConfigPageProps) {
  const rememberId = useId();
  const isActiveModel = loadedConfig != null;
  const initial = useMemo(() => {
    const resolved = resolveInitialConfig(target.id, target.ggufVariant);
    return loadedConfig
      ? { config: loadedConfig, remembered: resolved.remembered }
      : resolved;
  }, [target.id, target.ggufVariant, loadedConfig]);
  const [config, setConfig] = useState<PerModelConfig>(initial.config);
  const [remember, setRemember] = useState(initial.remembered);
  const [templateOpen, setTemplateOpen] = useState(false);
  const templateDefaults = useDefaultChatTemplate(
    target.id,
    target.ggufVariant,
    templateOpen,
  );

  const update = (patch: Partial<PerModelConfig>) =>
    setConfig((current) => ({ ...current, ...patch }));

  const isMtp =
    config.speculativeType != null &&
    MTP_SPECULATIVE_TYPES.has(config.speculativeType);
  const nativeContextLength = target.meta.contextLength ?? null;
  const baseline = loadedConfig ?? DEFAULT_PER_MODEL_CONFIG;
  const atBaseline = perModelConfigsEqual(config, baseline);

  const handleRun = () => {
    if (remember) {
      const saved = savePerModelConfig(target.id, target.ggufVariant, config);
      if (!saved) {
        toast.error("Couldn't save settings for this model.");
        return;
      }
    } else {
      deletePerModelConfig(target.id, target.ggufVariant);
    }
    onRun(config, remember);
  };

  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2.5 pb-4">
        <button
          type="button"
          onClick={onBack}
          className="nav-icon-btn shrink-0 text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
          aria-label="Back to model list"
        >
          <HugeiconsIcon
            icon={ArrowLeft01Icon}
            className="size-4"
            strokeWidth={1.75}
          />
        </button>
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-semibold uppercase leading-none tracking-wider text-muted-foreground">
            Run settings
          </div>
          <div className="mt-1.5 truncate text-[14px] font-semibold leading-tight text-nav-fg">
            {target.displayName}
          </div>
        </div>
      </div>

      <div className="-mr-1 max-h-[58vh] space-y-3.5 overflow-y-auto pr-1">
        {target.isGguf && (
          <>
            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>Context Length</span>
                <InfoHint>
                  Tokens of context to allocate. Leave blank for the model
                  default. Higher uses more VRAM.
                  {nativeContextLength != null
                    ? ` This model's native context is ${nativeContextLength.toLocaleString()} tokens.`
                    : ""}
                </InfoHint>
              </div>
              <input
                type="number"
                min={128}
                max={nativeContextLength ?? undefined}
                step={1}
                value={config.customContextLength ?? ""}
                placeholder="default"
                onChange={(event) => {
                  const raw = event.target.value;
                  if (raw === "") {
                    update({ customContextLength: null });
                    return;
                  }
                  const parsed = Number.parseInt(raw, 10);
                  if (!(Number.isFinite(parsed) && parsed >= 128)) {
                    update({ customContextLength: null });
                    return;
                  }
                  update({
                    customContextLength:
                      nativeContextLength != null
                        ? Math.min(parsed, nativeContextLength)
                        : parsed,
                  });
                }}
                aria-label="Context Length"
                className={NUMBER_INPUT_CLASS}
              />
            </div>

            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>KV Cache Dtype</span>
                <InfoHint>
                  Lower KV cache precision to save VRAM at the cost of some
                  quality. f16/bf16 are full precision; q8_0/q5_1/q4_1 are
                  quantized.
                </InfoHint>
              </div>
              <Select
                value={config.kvCacheDtype ?? "f16"}
                onValueChange={(v) =>
                  update({ kvCacheDtype: v === "f16" ? null : v })
                }
              >
                <SelectTrigger
                  animateRadius={false}
                  icon={ChevronDownStandardIcon}
                  iconClassName="size-3.5"
                  className={`w-[92px] ${SELECT_TRIGGER_CLASS}`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
                  <SelectItem value="f16">f16</SelectItem>
                  <SelectItem value="bf16">bf16</SelectItem>
                  <SelectItem value="q8_0">q8_0</SelectItem>
                  <SelectItem value="q5_1">q5_1</SelectItem>
                  <SelectItem value="q4_1">q4_1</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>Speculative Decoding</span>
                <InfoHint>
                  Faster generation with no accuracy hit. Auto picks MTP / ngram
                  based on the model and platform. Pick a strategy to force it.
                </InfoHint>
              </div>
              <Select
                value={config.speculativeType ?? "auto"}
                onValueChange={(v) =>
                  update({
                    speculativeType: v,
                    specDraftNMax:
                      v === "mtp" || v === "mtp+ngram"
                        ? config.specDraftNMax
                        : null,
                  })
                }
              >
                <SelectTrigger
                  animateRadius={false}
                  icon={ChevronDownStandardIcon}
                  iconClassName="size-3.5"
                  className={`w-[124px] ${SELECT_TRIGGER_CLASS}`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="mtp">MTP</SelectItem>
                  <SelectItem value="ngram">Ngram</SelectItem>
                  <SelectItem value="mtp+ngram">MTP+Ngram</SelectItem>
                  <SelectItem value="off">Off</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {isMtp && (
              <div className={ROW_CLASS}>
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className={LABEL_CLASS}>Draft Tokens</span>
                  <InfoHint>
                    Max MTP draft tokens per step. Leave blank for the platform
                    default (2 on GPU, 3 on CPU/Mac).
                  </InfoHint>
                </div>
                <input
                  type="number"
                  min={1}
                  max={16}
                  step={1}
                  value={config.specDraftNMax ?? ""}
                  placeholder="auto"
                  onChange={(event) => {
                    const raw = event.target.value;
                    if (raw === "") {
                      update({ specDraftNMax: null });
                      return;
                    }
                    const parsed = Number.parseInt(raw, 10);
                    if (Number.isFinite(parsed)) {
                      update({
                        specDraftNMax: Math.max(1, Math.min(16, parsed)),
                      });
                    }
                  }}
                  aria-label="Speculative decoding draft tokens"
                  className={NUMBER_INPUT_CLASS}
                />
              </div>
            )}

            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>Tensor Parallelism</span>
                <InfoHint>
                  No effect on a single GPU. On multi-GPU setups, improves
                  tokens/sec for dense models. MoE models don't benefit.
                </InfoHint>
              </div>
              <Switch
                className="panel-switch shrink-0"
                checked={config.tensorParallel}
                onCheckedChange={(checked) =>
                  update({ tensorParallel: checked })
                }
              />
            </div>
          </>
        )}

        {target.isGguf && (
          <div className={ROW_CLASS}>
            <div className="flex min-w-0 items-center gap-1.5">
              <span className={LABEL_CLASS}>Chat Template</span>
              <InfoHint>
                Override the model's chat template with custom Jinja. Applies
                when the model loads.
              </InfoHint>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <span className="text-[12px] text-muted-foreground">
                {config.chatTemplateOverride ? "Custom" : "Default"}
              </span>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className={`h-7 px-3 text-[13px] ${CONTROL_SURFACE}`}
                onClick={() => setTemplateOpen(true)}
              >
                Edit
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-border/60 pt-4">
        <div className="flex min-w-0 items-center gap-2">
          <Checkbox
            id={rememberId}
            checked={remember}
            onCheckedChange={(checked) => setRemember(checked === true)}
          />
          <label
            htmlFor={rememberId}
            className="cursor-pointer select-none truncate text-[13px] text-nav-fg"
          >
            Remember for this model
          </label>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={atBaseline}
            onClick={() => setConfig({ ...baseline })}
          >
            Reset
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-8"
            disabled={isActiveModel && atBaseline}
            onClick={handleRun}
          >
            {isActiveModel ? "Reload model" : "Load model"}
          </Button>
        </div>
      </div>

      <ChatTemplateEditorDialog
        open={templateOpen}
        onOpenChange={setTemplateOpen}
        value={config.chatTemplateOverride}
        defaultTemplate={templateDefaults.template}
        defaultLoading={templateDefaults.loading}
        onSave={(override) => update({ chatTemplateOverride: override })}
      />
    </div>
  );
}
