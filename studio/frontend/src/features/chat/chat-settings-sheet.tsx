// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useIsMobile } from "@/hooks/use-mobile";
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
import type { ReactNode } from "react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";
import {
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "./types/runtime";

export const defaultInferenceParams = DEFAULT_INFERENCE_PARAMS;
export type { InferenceParams } from "./types/runtime";

export interface Preset {
  name: string;
  params: InferenceParams;
}

interface SystemPromptTemplate {
  id: string;
  name: string;
  content: string;
  updatedAt: number;
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
const CHAT_SYSTEM_PROMPTS_KEY = "unsloth_chat_system_prompts";

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

function createSystemPromptId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `prompt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function loadSavedSystemPrompts(): SystemPromptTemplate[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(CHAT_SYSTEM_PROMPTS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is SystemPromptTemplate => {
        if (!item || typeof item !== "object") return false;
        const maybe = item as Partial<SystemPromptTemplate>;
        return (
          typeof maybe.id === "string" &&
          typeof maybe.name === "string" &&
          typeof maybe.content === "string"
        );
      })
      .map((item) => ({
        id: item.id.trim(),
        name: item.name.trim() || "Untitled prompt",
        content: item.content,
        updatedAt: typeof item.updatedAt === "number" ? item.updatedAt : Date.now(),
      }))
      .filter((item) => item.id.length > 0);
  } catch {
    return [];
  }
}

function saveSystemPrompts(templates: SystemPromptTemplate[]) {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(CHAT_SYSTEM_PROMPTS_KEY, JSON.stringify(templates));
  } catch {
    // ignore
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
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, boolean] => typeof entry[1] === "boolean",
      ),
    );
  } catch (error) {
    console.warn("Failed to load collapsible state from localStorage:", error);
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
    return Object.hasOwn(saved, label) ? saved[label] : defaultOpen;
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
  const speculativeType = useChatRuntimeStore((s) => s.speculativeType);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const loadedSpeculativeType = useChatRuntimeStore(
    (s) => s.loadedSpeculativeType,
  );
  const currentModels = useChatRuntimeStore((s) => s.models);
  const currentCheckpoint = params.checkpoint;
  const currentModelIsVision =
    currentModels.find((m) => m.id === currentCheckpoint)?.isVision ?? false;
  const ggufContextLength = useChatRuntimeStore((s) => s.ggufContextLength);
  const ggufMaxContextLength = useChatRuntimeStore(
    (s) => s.ggufMaxContextLength,
  );
  const ggufNativeContextLength = useChatRuntimeStore(
    (s) => s.ggufNativeContextLength,
  );
  const kvCacheDtype = useChatRuntimeStore((s) => s.kvCacheDtype);
  const setKvCacheDtype = useChatRuntimeStore((s) => s.setKvCacheDtype);
  const loadedKvCacheDtype = useChatRuntimeStore((s) => s.loadedKvCacheDtype);
  const customContextLength = useChatRuntimeStore((s) => s.customContextLength);
  const setCustomContextLength = useChatRuntimeStore(
    (s) => s.setCustomContextLength,
  );

  const ctxDisplayValue = customContextLength ?? ggufContextLength ?? "";
  const ctxMaxValue = ggufNativeContextLength ?? ggufContextLength ?? null;
  const kvDirty = kvCacheDtype !== loadedKvCacheDtype;
  const ctxDirty = customContextLength !== null;
  const specDirty = speculativeType !== loadedSpeculativeType;
  const modelSettingsDirty = kvDirty || ctxDirty || specDirty;
  const [customPresets, setCustomPresets] = useState<Preset[]>(() =>
    loadSavedCustomPresets(),
  );
  const [activePreset, setActivePreset] = useState(() =>
    loadSavedActivePreset(),
  );
  const [presetNameInput, setPresetNameInput] = useState(() =>
    loadSavedActivePreset(),
  );
  const presetControlRowRef = useRef<HTMLDivElement>(null);
  const [presetMenuWidthPx, setPresetMenuWidthPx] = useState<number | undefined>(
    undefined,
  );
  const [systemPromptManagerOpen, setSystemPromptManagerOpen] = useState(false);
  const [systemPrompts, setSystemPrompts] = useState<SystemPromptTemplate[]>(
    () => loadSavedSystemPrompts(),
  );
  const [selectedSystemPromptId, setSelectedSystemPromptId] = useState<string | null>(null);
  const [systemPromptNameDraft, setSystemPromptNameDraft] = useState("");
  const [systemPromptContentDraft, setSystemPromptContentDraft] = useState("");
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

  function savePresetWithName(rawName: string) {
    const trimmed = rawName.trim();
    if (!trimmed) {
      toast.error("Enter a preset name");
      return;
    }
    if (BUILTIN_PRESETS.some((preset) => preset.name === trimmed)) {
      toast.error(`"${trimmed}" is reserved. Pick a different name.`);
      return;
    }
    setCustomPresets((prev) => {
      const next = prev.filter((p) => p.name !== trimmed);
      const merged = [...next, { name: trimmed, params: { ...params } }];
      if (canUseStorage()) {
        try {
          localStorage.setItem(CHAT_PRESETS_KEY, JSON.stringify(merged));
        } catch {
          // ignore
        }
      }
      return merged;
    });
    if (canUseStorage()) {
      try {
        localStorage.setItem(CHAT_ACTIVE_PRESET_KEY, trimmed);
      } catch {
        // ignore
      }
    }
    setActivePreset(trimmed);
    setPresetNameInput(trimmed);
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

  function openSystemPromptManager() {
    const available =
      systemPrompts.length > 0
        ? systemPrompts
        : [
            {
              id: createSystemPromptId(),
              name: "Prompt 1",
              content: params.systemPrompt,
              updatedAt: Date.now(),
            },
          ];
    if (systemPrompts.length === 0) {
      setSystemPrompts(available);
      saveSystemPrompts(available);
    }
    const selected =
      available.find((item) => item.id === selectedSystemPromptId) ?? available[0];
    setSelectedSystemPromptId(selected.id);
    setSystemPromptNameDraft(selected.name);
    setSystemPromptContentDraft(selected.content);
    setSystemPromptManagerOpen(true);
  }

  function saveCurrentSystemPromptTemplate() {
    if (!selectedSystemPromptId) return;
    const now = Date.now();
    const next = systemPrompts.map((item) =>
      item.id === selectedSystemPromptId
        ? {
            ...item,
            name: systemPromptNameDraft.trim() || "Untitled prompt",
            content: systemPromptContentDraft,
            updatedAt: now,
          }
        : item,
    );
    setSystemPrompts(next);
    saveSystemPrompts(next);
  }

  function createSystemPromptTemplate() {
    const created: SystemPromptTemplate = {
      id: createSystemPromptId(),
      name: "Untitled prompt",
      content: "",
      updatedAt: Date.now(),
    };
    const next = [created, ...systemPrompts];
    setSystemPrompts(next);
    saveSystemPrompts(next);
    setSelectedSystemPromptId(created.id);
    setSystemPromptNameDraft(created.name);
    setSystemPromptContentDraft(created.content);
  }

  function duplicateSystemPromptTemplate() {
    const duplicate: SystemPromptTemplate = {
      id: createSystemPromptId(),
      name: `${(systemPromptNameDraft.trim() || "Prompt")} copy`,
      content: systemPromptContentDraft,
      updatedAt: Date.now(),
    };
    const next = [duplicate, ...systemPrompts];
    setSystemPrompts(next);
    saveSystemPrompts(next);
    setSelectedSystemPromptId(duplicate.id);
    setSystemPromptNameDraft(duplicate.name);
    setSystemPromptContentDraft(duplicate.content);
  }

  function deleteCurrentSystemPromptTemplate() {
    if (!selectedSystemPromptId) return;
    const remaining = systemPrompts.filter((item) => item.id !== selectedSystemPromptId);
    const next =
      remaining.length > 0
        ? remaining
        : [
            {
              id: createSystemPromptId(),
              name: "Prompt 1",
              content: params.systemPrompt,
              updatedAt: Date.now(),
            },
          ];
    setSystemPrompts(next);
    saveSystemPrompts(next);
    const selected = next[0];
    setSelectedSystemPromptId(selected.id);
    setSystemPromptNameDraft(selected.name);
    setSystemPromptContentDraft(selected.content);
  }

  function applySystemPromptTemplate() {
    set("systemPrompt")(systemPromptContentDraft);
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

  useEffect(() => {
    setPresetNameInput(activePreset);
  }, [activePreset]);

  useLayoutEffect(() => {
    const el = presetControlRowRef.current;
    if (!el) return;
    const measure = () => {
      setPresetMenuWidthPx(el.getBoundingClientRect().width);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

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
          <div className="space-y-1.5">
            <div ref={presetControlRowRef} className="w-full min-w-0">
              <DropdownMenu>
                <InputGroup className="!h-8 min-h-8 min-w-0 items-stretch gap-0 rounded-2xl pr-0 focus-within:border-input focus-within:ring-0 focus-within:shadow-none has-[[data-slot=input-group-control]:focus-visible]:border-input has-[[data-slot=input-group-control]:focus-visible]:ring-0 has-[[data-slot=input-group-control]:focus-visible]:shadow-none">
                  <InputGroupInput
                    id="inference-preset-name"
                    value={presetNameInput}
                    onChange={(e) => setPresetNameInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        savePresetWithName(presetNameInput);
                      }
                    }}
                    placeholder="Preset name"
                    maxLength={80}
                    autoComplete="off"
                    className="!h-8 min-h-0 min-w-0 self-stretch !pl-2.5 !pr-2 pt-1 pb-1 text-sm leading-10 md:text-sm"
                    aria-label="Inference preset name"
                  />
                  <InputGroupAddon
                    align="inline-end"
                    className="min-h-0 shrink-0 gap-0 self-stretch border-0 py-0 pl-0 !pr-0 has-[>button]:mr-0"
                  >
                    <DropdownMenuTrigger asChild={true}>
                      <InputGroupButton
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        className="!h-8 min-h-8 !w-7 min-w-7 shrink-0 rounded-none rounded-r-2xl border-l border-border px-0 text-muted-foreground transition-colors hover:bg-primary/15 hover:text-primary data-[state=open]:bg-primary/20 data-[state=open]:text-primary"
                        title="Choose a preset"
                        aria-label="Open preset list"
                      >
                        <HugeiconsIcon
                          icon={ArrowDown01Icon}
                          className="size-3.5"
                          strokeWidth={2}
                        />
                      </InputGroupButton>
                    </DropdownMenuTrigger>
                  </InputGroupAddon>
                </InputGroup>
                <DropdownMenuContent
                  align="end"
                  className="min-w-40 max-w-none"
                  style={
                    presetMenuWidthPx != null
                      ? {
                          width: presetMenuWidthPx,
                          minWidth: presetMenuWidthPx,
                        }
                      : undefined
                  }
                >
                  {presets.map((p) => (
                    <DropdownMenuItem
                      key={p.name}
                      onSelect={() => applyPreset(p.name)}
                    >
                      {p.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => savePresetWithName(presetNameInput)}
                disabled={presetNameInput.trim().length === 0}
                className="inline-flex h-8 shrink-0 items-center justify-center gap-0 rounded-4xl border px-2.5 text-xs text-muted-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                title="Save current settings under this name"
              >
                <span className="inline-flex shrink-0 items-center pr-1.5">
                  <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
                </span>
                Save
              </button>
              <button
                type="button"
                onClick={() => deletePreset(activePreset)}
                disabled={isBuiltinPreset}
                className="inline-flex h-8 shrink-0 items-center justify-center gap-0 rounded-4xl border px-2.5 text-xs text-muted-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
                title={
                  isBuiltinPreset
                    ? "Built-in presets cannot be deleted"
                    : "Delete selected preset"
                }
              >
                <span className="inline-flex shrink-0 items-center pr-1.5">
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </span>
                Delete
              </button>
            </div>
          </div>
        </div>

        <div className="px-2 pb-4">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <label htmlFor="system-prompt" className="block text-xs font-medium">
              System Prompt
            </label>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={openSystemPromptManager}
            >
              Manage
            </Button>
          </div>
          <Textarea
            id="system-prompt"
            value={params.systemPrompt}
            onChange={(e) => set("systemPrompt")(e.target.value)}
            placeholder="You are a helpful assistant..."
            className="min-h-20 text-xs corner-squircle"
            rows={3}
          />
        </div>

        <CollapsibleSection
          icon={Settings02Icon}
          label="Model"
          defaultOpen={true}
        >
          <div className="flex flex-col gap-3 py-1">
            {isGguf && (
              <>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium">Context Length</span>
                    <Input
                      type="number"
                      value={
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (ggufContextLength ?? "")
                      }
                      placeholder="..."
                      min={128}
                      max={ctxMaxValue ?? undefined}
                      step={1024}
                      className="h-6 w-[100px] text-right text-xs tabular-nums"
                      onChange={(e) => {
                        const raw = e.target.value;
                        if (raw === "") {
                          setCustomContextLength(null);
                          return;
                        }
                        const v = Number.parseInt(raw, 10);
                        if (!Number.isNaN(v) && v >= 0) {
                          const maxCtx =
                            ctxMaxValue ?? Number.POSITIVE_INFINITY;
                          const clamped = Math.min(v, maxCtx);
                          setCustomContextLength(
                            clamped === (ggufContextLength ?? 0)
                              ? null
                              : clamped,
                          );
                        }
                      }}
                    />
                  </div>
                  <Slider
                    min={1024}
                    max={ctxMaxValue ?? 4096}
                    step={1024}
                    value={[
                      Math.min(
                        typeof ctxDisplayValue === "number"
                          ? ctxDisplayValue
                          : (ggufContextLength ?? 4096),
                        ctxMaxValue ?? 4096,
                      ),
                    ]}
                    onValueChange={([v]) => {
                      setCustomContextLength(
                        v === (ggufContextLength ?? 0) ? null : v,
                      );
                    }}
                  />
                  {ggufMaxContextLength != null &&
                    typeof ctxDisplayValue === "number" &&
                    ctxDisplayValue > ggufMaxContextLength && (
                      <p className="text-[11px] text-amber-500">
                        Exceeds estimated VRAM capacity ({ggufMaxContextLength.toLocaleString()} tokens). The model may use system RAM.
                      </p>
                    )}
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_65px] items-center gap-x-3">
                  <div className="min-w-0">
                    <div className="text-xs font-medium">KV Cache Dtype</div>
                    <div className="text-[11px] text-muted-foreground">
                      Quantize KV cache to reduce VRAM.
                    </div>
                  </div>
                  <div className="w-full min-w-0">
                    <Select
                      value={kvCacheDtype ?? "f16"}
                      onValueChange={(v) => {
                        setKvCacheDtype(v === "f16" ? null : v);
                      }}
                    >
                      <SelectTrigger className="grid h-7 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 px-2 py-0 text-xs [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0">
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
                </div>
                {!currentModelIsVision && (
                  <div className="grid grid-cols-[minmax(0,1fr)_65px] items-center gap-x-3">
                    <div className="min-w-0">
                      <div className="text-xs font-medium">
                        Speculative Decoding
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        Speed up generation with no VRAM cost.
                      </div>
                    </div>
                    <div className="w-full min-w-0">
                      <Select
                        value={speculativeType ?? "off"}
                        onValueChange={(v) => {
                          setSpeculativeType(v === "off" ? null : v);
                        }}
                      >
                        <SelectTrigger className="grid h-7 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-1 px-2 py-0 text-xs [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ngram-mod">On</SelectItem>
                          <SelectItem value="off">Off</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
                {modelSettingsDirty && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    <button
                      type="button"
                      onClick={() => onReloadModel?.()}
                      className="rounded-md bg-primary px-2.5 py-1 text-[11px] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
                    >
                      Apply
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setCustomContextLength(null);
                        setKvCacheDtype(loadedKvCacheDtype);
                        setSpeculativeType(loadedSpeculativeType);
                      }}
                      className="rounded-md border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-accent"
                    >
                      Reset
                    </button>
                  </div>
                )}
              </>
            )}
            {!isGguf && params.checkpoint && (
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium">Enable custom code</div>
                  <div className="text-[11px] text-muted-foreground">
                    Allow models with custom code (e.g. Nemotron). Only enable
                    if sure.
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
                isGguf &&
                ggufContextLength &&
                params.maxTokens >= ggufContextLength
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

        <CollapsibleSection
          icon={UserSettings01Icon}
          label="Preferences"
          defaultOpen={true}
        >
          <div className="flex flex-col gap-3 py-1">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs font-medium">Auto title</div>
                <div className="text-[11px] text-muted-foreground">
                  Generate short title after reply.
                </div>
              </div>
              <Switch checked={autoTitle} onCheckedChange={onAutoTitleChange} />
            </div>
            <HfTokenField />
          </div>
        </CollapsibleSection>

        <ChatTemplateSection onReloadModel={onReloadModel} />
      </div>
      <Dialog
        open={systemPromptManagerOpen}
        onOpenChange={(nextOpen) => {
          setSystemPromptManagerOpen(nextOpen);
        }}
      >
        <DialogContent className="corner-squircle sm:max-w-[50.4rem]">
          <DialogHeader>
            <DialogTitle>System Prompt Manager</DialogTitle>
            <DialogDescription>
              Save and reuse system prompts for chat sessions.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 sm:grid-cols-[13rem_minmax(0,1fr)]">
            <div className="space-y-2">
              <div className="flex gap-1.5">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 flex-1 text-[11px]"
                  onClick={createSystemPromptTemplate}
                >
                  New
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 flex-1 text-[11px]"
                  onClick={duplicateSystemPromptTemplate}
                  disabled={!selectedSystemPromptId}
                >
                  Duplicate
                </Button>
              </div>
              <div className="max-h-[16.8rem] space-y-1 overflow-y-auto rounded-md border p-1.5">
                {systemPrompts.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setSelectedSystemPromptId(item.id);
                      setSystemPromptNameDraft(item.name);
                      setSystemPromptContentDraft(item.content);
                    }}
                    className={`w-full rounded-md px-2 py-1.5 text-left text-xs transition-colors ${
                      selectedSystemPromptId === item.id
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent/60"
                    }`}
                  >
                    <span className="block truncate">{item.name}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-2">
              <Input
                value={systemPromptNameDraft}
                onChange={(event) => setSystemPromptNameDraft(event.target.value)}
                placeholder="Prompt name"
                maxLength={80}
              />
              <Textarea
                value={systemPromptContentDraft}
                onChange={(event) => setSystemPromptContentDraft(event.target.value)}
                placeholder="You are a helpful assistant..."
                className="min-h-[14.7rem] text-xs corner-squircle"
                rows={10}
              />
            </div>
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={deleteCurrentSystemPromptTemplate}
              disabled={!selectedSystemPromptId}
            >
              Delete
            </Button>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={saveCurrentSystemPromptTemplate}
                disabled={!selectedSystemPromptId}
              >
                Save
              </Button>
              <Button
                type="button"
                onClick={applySystemPromptTemplate}
                disabled={!selectedSystemPromptId}
              >
                Apply
              </Button>
            </div>
          </DialogFooter>
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
  const setMaxToolCalls = useChatRuntimeStore(
    (s) => s.setMaxToolCallsPerMessage,
  );

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
      displayValue={
        sliderValue >= 41 ? "Max" : sliderValue === 0 ? "Off" : undefined
      }
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
  const setAutoHealToolCalls = useChatRuntimeStore(
    (s) => s.setAutoHealToolCalls,
  );

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

function HfTokenField() {
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const setHfToken = useChatRuntimeStore((s) => s.setHfToken);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="min-w-0">
        <div className="text-xs font-medium">Hugging Face Token</div>
        <div className="text-[11px] text-muted-foreground">
          For downloading gated or private models.
        </div>
      </div>
      <Input
        type="password"
        value={hfToken}
        placeholder="hf_..."
        className="h-7 text-xs font-mono"
        onChange={(e) => setHfToken(e.target.value)}
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
          className="min-h-32 font-mono text-[8px] leading-relaxed md:text-[8px] corner-squircle"
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
