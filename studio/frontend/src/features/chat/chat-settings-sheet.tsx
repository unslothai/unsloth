// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ArrowDown01Icon,
  CodeIcon,
  Delete02Icon,
  FloppyDiskIcon,
  PencilEdit01Icon,
  Settings02Icon,
  SlidersHorizontalIcon,
  UserSettings01Icon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion } from "motion/react";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useIsMobile } from "@/hooks/use-mobile";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "./types/runtime";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import { Switch } from "@/components/ui/switch";

export const defaultInferenceParams = DEFAULT_INFERENCE_PARAMS;
export type { InferenceParams } from "./types/runtime";

export interface Preset {
  name: string;
  params: InferenceParams;
}

const BUILTIN_PRESETS: Preset[] = [
  { name: "Default", params: { ...defaultInferenceParams } },
  {
    name: "Creative",
    params: {
      ...defaultInferenceParams,
      temperature: 1.5,
      topP: 1.0,
      topK: 0,
      minP: 0.1,
      repetitionPenalty: 1.0,
    },
  },
  {
    name: "Precise",
    params: {
      ...defaultInferenceParams,
      temperature: 0.1,
      topP: 0.95,
      topK: 80,
      minP: 0.01,
      repetitionPenalty: 1.0,
    },
  },
];

const CHAT_PRESETS_KEY = "unsloth_chat_custom_presets";
const CHAT_ACTIVE_PRESET_KEY = "unsloth_chat_active_preset";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadSavedCustomPresets(): Preset[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(CHAT_PRESETS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is Preset => {
        if (!item || typeof item !== "object") return false;
        const maybe = item as Partial<Preset>;
        return typeof maybe.name === "string" && !!maybe.params;
      })
      .map((preset) => ({
        name: preset.name.trim(),
        params: {
          ...defaultInferenceParams,
          ...preset.params,
        },
      }))
      .filter(
        (preset) =>
          preset.name.length > 0 &&
          !BUILTIN_PRESETS.some((builtin) => builtin.name === preset.name),
      );
  } catch {
    return [];
  }
}

function loadSavedActivePreset(): string {
  if (!canUseStorage()) return "Default";
  try {
    return localStorage.getItem(CHAT_ACTIVE_PRESET_KEY) ?? "Default";
  } catch {
    return "Default";
  }
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  displayValue,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  displayValue?: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{label}</span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {displayValue ?? value}
        </span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([v]) => onChange(v)}
      />
    </div>
  );
}

const COLLAPSIBLE_STATE_KEY = "unsloth_chat_collapsible_state";

function loadCollapsibleState(): Record<string, boolean> {
  if (!canUseStorage()) return {};
  try {
    const raw = localStorage.getItem(COLLAPSIBLE_STATE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, boolean>;
    }
    return {};
  } catch (error) {
    console.error("Failed to load collapsible state from localStorage:", error);
    return {};
  }
}

function saveCollapsibleOpen(label: string, open: boolean) {
  if (!canUseStorage()) return;
  try {
    const state = loadCollapsibleState();
    state[label] = open;
    localStorage.setItem(COLLAPSIBLE_STATE_KEY, JSON.stringify(state));
  } catch {
    // ignore
  }
}

function CollapsibleSection({
  icon,
  label,
  children,
  defaultOpen = false,
}: {
  icon: Parameters<typeof HugeiconsIcon>[0]["icon"];
  label: string;
  children?: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(() => {
    const saved = loadCollapsibleState();
    return label in saved ? saved[label] : defaultOpen;
  });

  return (
    <div>
      <button
        type="button"
        onClick={() => {
          const next = !open;
          setOpen(next);
          saveCollapsibleOpen(label, next);
        }}
        className="flex w-full items-center corner-squircle gap-2.5 rounded-md px-2 py-2 text-sm transition-colors hover:bg-accent"
      >
        <HugeiconsIcon icon={icon} className="size-4 text-muted-foreground" />
        <span className="flex-1 text-left font-medium">{label}</span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.15 }}
        >
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            className="size-3.5 text-muted-foreground"
          />
        </motion.div>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-2 pb-3 pt-1">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface ChatSettingsPanelProps {
  open: boolean;
  onOpenChange?: (open: boolean) => void;
  params: InferenceParams;
  onParamsChange: (params: InferenceParams) => void;
  autoTitle: boolean;
  onAutoTitleChange: (enabled: boolean) => void;
  onReloadModel?: () => void;
}

export function ChatSettingsPanel({
  open,
  onOpenChange,
  params,
  onParamsChange,
  autoTitle,
  onAutoTitleChange,
  onReloadModel,
}: ChatSettingsPanelProps) {
  const isMobile = useIsMobile();
  const isGguf = useChatRuntimeStore((s) => s.activeGgufVariant) != null;
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const setKvCacheDtype = useChatRuntimeStore((s) => s.setKvCacheDtype);
  const [customPresets, setCustomPresets] = useState<Preset[]>(() =>
    loadSavedCustomPresets(),
  );
  const [activePreset, setActivePreset] = useState(() => loadSavedActivePreset());
  const [savePresetOpen, setSavePresetOpen] = useState(false);
  const [presetNameDraft, setPresetNameDraft] = useState("");
  const presets = useMemo(
    () => [...BUILTIN_PRESETS, ...customPresets],
    [customPresets],
  );
  const isBuiltinPreset = BUILTIN_PRESETS.some((p) => p.name === activePreset);

  function set<K extends keyof InferenceParams>(key: K) {
    return (v: InferenceParams[K]) => onParamsChange({ ...params, [key]: v });
  }

  function applyPreset(name: string) {
    const p = presets.find((pr) => pr.name === name);
    if (p) {
      onParamsChange({
        ...p.params,
        systemPrompt: params.systemPrompt,
        checkpoint: params.checkpoint,
        trustRemoteCode: params.trustRemoteCode,
      });
      setActivePreset(name);
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, name);
        } catch {
          // ignore
        }
      }
    }
  }

  function openSavePresetDialog() {
    setPresetNameDraft(activePreset === "Default" ? "" : activePreset);
    setSavePresetOpen(true);
  }

  function savePresetWithName(rawName: string) {
    const trimmed = rawName.trim();
    if (!trimmed) {
      return;
    }
    if (BUILTIN_PRESETS.some((preset) => preset.name === trimmed)) {
      return;
    }
    setCustomPresets((prev) => {
      const next = [
        ...prev.filter((preset) => preset.name !== trimmed),
        { name: trimmed, params: { ...params } },
      ];
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(next));
        } catch {
          // ignore
        }
      }
      return next;
    });
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, trimmed);
      } catch {
        // ignore
      }
    }
    setActivePreset(trimmed);
    setSavePresetOpen(false);
  }

  function deletePreset(name: string) {
    if (BUILTIN_PRESETS.some((p) => p.name === name)) {
      return;
    }
    setCustomPresets((prev) => {
      const next = prev.filter((preset) => preset.name !== name);
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(next));
        } catch {
          // ignore
        }
      }
      return next;
    });
    if (activePreset === name) {
      setActivePreset("Default");
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, "Default");
        } catch {
          // ignore
        }
      }
    }
  }

  useEffect(() => {
    if (presets.some((preset) => preset.name === activePreset)) return;
    setActivePreset("Default");
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, "Default");
      } catch {
        // ignore
      }
    }
  }, [activePreset, presets]);

  const settingsContent = (
    <>
      <div className="flex items-center gap-2 px-4 py-3">
        <HugeiconsIcon
          icon={PencilEdit01Icon}
          className="size-4 text-muted-foreground/70"
        />
        <span className="flex-1 text-base font-semibold tracking-tight">
          Configuration
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-1.5">
        {/* mt-4 matches the Playground sidebar gap (SidebarHeader py-3 + SidebarGroup pt-1) */}
        <div className="mt-4 px-2 pb-3">
            <div className="flex items-center gap-2">
              <Select value={activePreset} onValueChange={applyPreset}>
                <SelectTrigger className="h-8 flex-1 corner-squircle text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {presets.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <button
                type="button"
                onClick={openSavePresetDialog}
                className="flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs text-muted-foreground transition-colors hover:bg-accent"
                title="Save preset"
              >
                <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
                Save
              </button>
              <button
                type="button"
                onClick={() => deletePreset(activePreset)}
                disabled={isBuiltinPreset}
                className="flex h-8 items-center gap-1.5 rounded-md border px-2.5 text-xs text-muted-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                title={
                  isBuiltinPreset
                    ? "Built-in presets cannot be deleted"
                    : "Delete selected preset"
                }
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                Delete
              </button>
            </div>
          </div>

          <div className="px-2 pb-4">
            <label
              htmlFor="system-prompt"
              className="mb-1.5 block text-xs font-medium"
            >
              System Prompt
            </label>
            <Textarea
              id="system-prompt"
              value={params.systemPrompt}
              onChange={(e) => set("systemPrompt")(e.target.value)}
              placeholder="You are a helpful assistant..."
              className="min-h-20 text-xs corner-squircle"
              rows={3}
            />
          </div>

          <CollapsibleSection icon={Settings02Icon} label="Model">
            <div className="flex flex-col gap-3 py-1">
              {isGguf && (
                <>
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs font-medium">Context Length</div>
                      <div className="text-[11px] text-muted-foreground">
                        Set before loading model (coming soon)
                      </div>
                    </div>
                    <Input
                      value={ggufContextLength ?? ""}
                      placeholder="Loading..."
                      disabled={true}
                      className="h-7 w-[90px] text-xs opacity-50 cursor-not-allowed"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs font-medium">KV Cache Dtype</div>
                      <div className="text-[11px] text-muted-foreground">
                        Quantize KV cache to reduce VRAM. Reload to apply.
                      </div>
                    </div>
                    <Select
                      value={kvCacheDtype ?? "f16"}
                      onValueChange={(v) => {
                        setKvCacheDtype(v === "f16" ? null : v);
                        onReloadModel?.();
                      }}
                    >
                      <SelectTrigger className="h-7 w-[90px] text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="f16">f16</SelectItem>
                        <SelectItem value="bf16">bf16</SelectItem>
                        <SelectItem value="q8_0">q8_0</SelectItem>
                        <SelectItem value="q5_1">q5_1</SelectItem>
                        <SelectItem value="q4_1">q4_1</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </>
              )}
              {!isGguf && (
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-xs font-medium">Trust remote code</div>
                    <div className="text-[11px] text-muted-foreground">
                      Allow models with custom code (e.g. Nemotron). Only enable for repos you trust.
                    </div>
                  </div>
                  <Switch
                    checked={params.trustRemoteCode ?? false}
                    onCheckedChange={set("trustRemoteCode")}
                  />
                </div>
              )}
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            icon={SlidersHorizontalIcon}
            label="Sampling"
            defaultOpen={true}
          >
            <div className="flex flex-col gap-5">
              <ParamSlider
                label="Temperature"
                value={params.temperature}
                min={0}
                max={2}
                step={0.1}
                onChange={set("temperature")}
              />
              <ParamSlider
                label="Top P"
                value={params.topP}
                min={0}
                max={1}
                step={0.05}
                onChange={set("topP")}
                displayValue={params.topP === 1 ? "Off" : undefined}
              />
              <ParamSlider
                label="Top K"
                value={params.topK}
                min={0}
                max={100}
                step={1}
                onChange={set("topK")}
                displayValue={params.topK === 0 ? "Off" : undefined}
              />
              <ParamSlider
                label="Min P"
                value={params.minP}
                min={0}
                max={1}
                step={0.01}
                onChange={set("minP")}
              />
              <ParamSlider
                label="Repetition Penalty"
                value={params.repetitionPenalty}
                min={1}
                max={2}
                step={0.05}
                onChange={set("repetitionPenalty")}
                displayValue={params.repetitionPenalty === 1 ? "Off" : undefined}
              />
              <ParamSlider
                label="Presence Penalty"
                value={params.presencePenalty}
                min={0}
                max={2}
                step={0.1}
                onChange={set("presencePenalty")}
                displayValue={params.presencePenalty === 0 ? "Off" : undefined}
              />
              {!isGguf && (
                <ParamSlider
                  label="Max Seq Length"
                  value={params.maxSeqLength}
                  min={128}
                  max={32768}
                  step={128}
                  onChange={set("maxSeqLength")}
                />
              )}
              <ParamSlider
                label="Max Tokens"
                value={params.maxTokens}
                min={64}
                max={isGguf && ggufContextLength ? ggufContextLength : 32768}
                step={64}
                onChange={set("maxTokens")}
                displayValue={
                  isGguf && ggufContextLength && params.maxTokens >= ggufContextLength
                    ? "Max"
                    : undefined
                }
              />
            </div>
          </CollapsibleSection>

          <CollapsibleSection icon={Wrench01Icon} label="Tools">
            <div className="flex flex-col gap-3 py-1">
              <AutoHealToolCallsToggle />
              <MaxToolCallsSlider />
              <ToolCallTimeoutSlider />
            </div>
          </CollapsibleSection>

          <CollapsibleSection icon={UserSettings01Icon} label="Preferences">
            <div className="flex flex-col gap-3 py-1">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium">Auto title</div>
                  <div className="text-[11px] text-muted-foreground">
                    Generate short title after reply.
                  </div>
                </div>
                <Switch
                  checked={autoTitle}
                  onCheckedChange={onAutoTitleChange}
                />
              </div>
            </div>
          </CollapsibleSection>

          <ChatTemplateSection onReloadModel={onReloadModel} />
        </div>
        <Dialog
          open={savePresetOpen}
          onOpenChange={(nextOpen) => {
            setSavePresetOpen(nextOpen);
            if (!nextOpen) {
              setPresetNameDraft("");
            }
          }}
        >
          <DialogContent className="corner-squircle sm:max-w-sm">
            <DialogHeader>
              <DialogTitle>Save Preset</DialogTitle>
              <DialogDescription>
                Enter a name for this inference preset.
              </DialogDescription>
            </DialogHeader>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                savePresetWithName(presetNameDraft);
              }}
              className="space-y-4"
            >
              <Input
                autoFocus={true}
                value={presetNameDraft}
                onChange={(event) => setPresetNameDraft(event.target.value)}
                placeholder="Preset name"
                maxLength={80}
              />
              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setSavePresetOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={presetNameDraft.trim().length === 0}>
                  Save
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </>
  );

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={onOpenChange}>
        <SheetContent side="right" className="w-[18rem] p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Configuration</SheetTitle>
            <SheetDescription>Chat inference settings</SheetDescription>
          </SheetHeader>
          <div className="flex h-full flex-col">{settingsContent}</div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <aside
      className={`shrink-0 self-start h-[calc(100%-0.875rem)] overflow-hidden bg-muted/70 rounded-2xl corner-squircle transition-[width] duration-200 ease-linear ${open ? "w-[17rem] border-l border-sidebar-border/70" : "w-0"}`}
    >
      <div className="flex h-full w-[17rem] flex-col">{settingsContent}</div>
    </aside>
  );
}

function MaxToolCallsSlider() {
  const maxToolCalls = useChatRuntimeStore((s) => s.maxToolCallsPerMessage);
  const setMaxToolCalls = useChatRuntimeStore((s) => s.setMaxToolCallsPerMessage);

  // Slider range 0-41; 41 maps to 9999 ("Max")
  const sliderValue = maxToolCalls >= 9999 ? 41 : Math.min(maxToolCalls, 40);

  return (
    <ParamSlider
      label="Max Tool Calls Per Message"
      value={sliderValue}
      min={0}
      max={41}
      step={1}
      onChange={(v) => setMaxToolCalls(v >= 41 ? 9999 : v)}
      displayValue={sliderValue >= 41 ? "Max" : sliderValue === 0 ? "Off" : undefined}
    />
  );
}

function ToolCallTimeoutSlider() {
  const timeout = useChatRuntimeStore((s) => s.toolCallTimeout);
  const setTimeout_ = useChatRuntimeStore((s) => s.setToolCallTimeout);

  // Slider 1-31; 31 maps to 9999 ("Max")
  const sliderValue = timeout >= 9999 ? 31 : Math.min(Math.max(timeout, 1), 30);

  const displayValue =
    sliderValue >= 31
      ? "Max"
      : sliderValue === 1
        ? "1 minute"
        : `${sliderValue} minutes`;

  return (
    <ParamSlider
      label="Max Tool Call Duration"
      value={sliderValue}
      min={1}
      max={31}
      step={1}
      onChange={(v) => setTimeout_(v >= 31 ? 9999 : v)}
      displayValue={displayValue}
    />
  );
}

function AutoHealToolCallsToggle() {
  const autoHealToolCalls = useChatRuntimeStore((s) => s.autoHealToolCalls);
  const setAutoHealToolCalls = useChatRuntimeStore((s) => s.setAutoHealToolCalls);

  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="text-xs font-medium">Auto Heal Tool Calls 🦥</div>
        <div className="text-[11px] text-muted-foreground">
          Fix malformed tool calls from the model automatically.
        </div>
      </div>
      <Switch
        checked={autoHealToolCalls}
        onCheckedChange={setAutoHealToolCalls}
      />
    </div>
  );
}

function ChatTemplateSection({
  onReloadModel,
}: {
  onReloadModel?: () => void;
}) {
  const defaultTemplate = useChatRuntimeStore((s) => s.defaultChatTemplate);
  const override = useChatRuntimeStore((s) => s.chatTemplateOverride);
  const setOverride = useChatRuntimeStore((s) => s.setChatTemplateOverride);

  if (!defaultTemplate) return null;

  const displayValue = override ?? defaultTemplate;
  const isModified = override !== null;

  return (
    <CollapsibleSection icon={CodeIcon} label="Chat Template">
      <div className="flex flex-col gap-2 py-1">
        <Textarea
          value={displayValue}
          onChange={(e) => setOverride(e.target.value)}
          className="min-h-32 font-mono text-[10px] leading-relaxed corner-squircle"
          rows={6}
          spellCheck={false}
        />
        <div className="flex flex-wrap gap-1.5">
          {isModified && (
            <>
              <button
                type="button"
                onClick={() => {
                  onReloadModel?.();
                }}
                className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                Apply & Reload
              </button>
              <button
                type="button"
                onClick={() => setOverride(null)}
                className="rounded-md border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent"
              >
                Revert changes
              </button>
            </>
          )}
        </div>
      </div>
    </CollapsibleSection>
  );
}
