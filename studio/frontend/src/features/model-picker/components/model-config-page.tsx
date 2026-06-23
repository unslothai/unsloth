// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
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
import { useMemo, useState } from "react";
import { useDefaultChatTemplate } from "../hooks/use-model-defaults";
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

const ROW_CLASS = "flex items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg";
const SELECT_TRIGGER_CLASS =
  "grid h-7 min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px]! font-medium text-nav-fg focus-visible:ring-0 focus-visible:border-transparent [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0";
const NUMBER_INPUT_CLASS =
  "h-7 w-[88px] rounded-full border-transparent bg-black/[0.04] dark:bg-white/[0.05] hover:bg-black/[0.06] dark:hover:bg-white/[0.1] pl-3 pr-2 py-0 text-[13px] font-medium text-nav-fg outline-none focus-visible:ring-0";

interface ModelConfigPageProps {
  target: ModelPickTarget;
  onBack: () => void;
  onRun: (config: PerModelConfig, remember: boolean) => void;
}

export function ModelConfigPage({ target, onBack, onRun }: ModelConfigPageProps) {
  const initial = useMemo(
    () => resolveInitialConfig(target.id, target.ggufVariant),
    [target.id, target.ggufVariant],
  );
  const [config, setConfig] = useState<PerModelConfig>(initial.config);
  const [remember, setRemember] = useState(initial.remembered);
  const [templateOpen, setTemplateOpen] = useState(false);
  const templateDefaults = useDefaultChatTemplate(target.id, templateOpen);

  const update = (patch: Partial<PerModelConfig>) =>
    setConfig((current) => ({ ...current, ...patch }));

  const isMtp =
    config.speculativeType != null &&
    MTP_SPECULATIVE_TYPES.has(config.speculativeType);

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
    <div className="flex max-h-[62vh] flex-col">
      <div className="flex items-center gap-2 pb-3">
        <button
          type="button"
          onClick={onBack}
          className="nav-icon-btn text-nav-icon-idle hover:bg-panel-surface-hover hover:text-black dark:hover:text-white"
          aria-label="Back to model list"
        >
          <HugeiconsIcon
            icon={ArrowLeft01Icon}
            className="size-4"
            strokeWidth={1.75}
          />
        </button>
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold text-nav-fg">
            {target.displayName}
          </div>
          <div className="text-[11px] text-muted-foreground">Run settings</div>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto pr-1">
        {target.isGguf && (
          <>
            <div className={ROW_CLASS}>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>Context Length</span>
                <InfoHint>
                  Tokens of context to allocate. Leave blank for the model
                  default. Higher uses more VRAM.
                </InfoHint>
              </div>
              <input
                type="number"
                min={128}
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
                  update({
                    customContextLength:
                      Number.isFinite(parsed) && parsed >= 128 ? parsed : null,
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
                  className={`w-[64px] ${SELECT_TRIGGER_CLASS}`}
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
                  Faster generation with no accuracy hit. Auto picks MTP /
                  ngram based on the model and platform. Pick a strategy to
                  force it.
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
                  className={`w-[76px] ${NUMBER_INPUT_CLASS}`}
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

        <div className={ROW_CLASS}>
          <div className="flex min-w-0 items-center gap-1.5">
            <span className={LABEL_CLASS}>Chat Template</span>
            <InfoHint>
              Override the model's chat template with custom Jinja. Applies when
              the model loads.
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
              className="h-7 rounded-full text-[13px]"
              onClick={() => setTemplateOpen(true)}
            >
              Edit
            </Button>
          </div>
        </div>
      </div>

      <div className="mt-1 flex items-center justify-between gap-2 border-t border-border/60 pt-3">
        <label className="flex cursor-pointer items-center gap-2 text-[13px] text-nav-fg">
          <input
            type="checkbox"
            checked={remember}
            onChange={(event) => setRemember(event.target.checked)}
            className="size-3.5 accent-current"
          />
          Remember for this model
        </label>
        <div className="flex gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 rounded-full"
            onClick={() => setConfig({ ...DEFAULT_PER_MODEL_CONFIG })}
          >
            Reset
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-8 rounded-full"
            onClick={handleRun}
          >
            Run model
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
