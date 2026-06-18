// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { estimateKvCache } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useGpuInfo } from "@/hooks";
import { Alert02Icon, InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  clearRememberedLoadSettings,
  loadRememberedLoadSettings,
  saveRememberedLoadSettings,
} from "./remembered-load-settings";

// Fraction of total memory we treat as usable before warning, matching the
// recommended list's OOM threshold (0.7 of VRAM + system RAM).
const MEM_BUDGET_FRACTION = 0.7;

// Filled pill controls, all the same height/width so the rows line up.
const FIELD_CLASS =
  "h-8 w-[132px] shrink-0 rounded-full border-0 bg-black/[0.04] dark:bg-white/[0.05] px-3 text-[13px]";
const NUMBER_FIELD_CLASS = `${FIELD_CLASS} [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none`;

// Slider range. The ceiling is the model's native max when known (passed in via
// maxContext), else this generous fallback; the numeric box stays unbounded.
const CTX_MIN = 2048;
const CTX_MAX = 131072;
const CTX_STEP = 1024;

function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          className="text-muted-foreground/50 transition-colors hover:text-muted-foreground"
        >
          <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-[220px]">{children}</TooltipContent>
    </Tooltip>
  );
}

function Setting({
  label,
  hint,
  control,
  children,
}: {
  label: string;
  hint: ReactNode;
  control?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="text-[13px] font-medium text-foreground">
            {label}
          </span>
          <InfoHint>{hint}</InfoHint>
        </div>
        {control}
      </div>
      {children}
    </div>
  );
}

/** Pre-load inference settings. Writes the chat runtime store fields the load
 * call reads, and can remember them per model. */
export function InferenceLoadSettingsDialog({
  open,
  onOpenChange,
  repoId,
  quant,
  maxContext,
  onLoad,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repoId: string;
  quant: string;
  maxContext?: number | null;
  onLoad: () => void;
}) {
  const setCustomContextLength = useChatRuntimeStore(
    (s) => s.setCustomContextLength,
  );
  const setKvCacheDtype = useChatRuntimeStore((s) => s.setKvCacheDtype);
  const setSpeculativeType = useChatRuntimeStore((s) => s.setSpeculativeType);
  const setSpecDraftNMax = useChatRuntimeStore((s) => s.setSpecDraftNMax);
  const setTensorParallel = useChatRuntimeStore((s) => s.setTensorParallel);

  // Seed once from remembered settings for this model, else the live store.
  const initial = useRef<{
    ctx: number | null;
    kv: string | null;
    spec: string | null;
    draft: number | null;
    tp: boolean;
    remember: boolean;
  }>(null);
  if (initial.current === null) {
    const saved = loadRememberedLoadSettings(repoId);
    if (saved) {
      initial.current = {
        ctx: saved.contextLength,
        kv: saved.kvCacheDtype,
        spec: saved.speculativeType,
        draft: saved.specDraftNMax,
        tp: saved.tensorParallel,
        remember: true,
      };
    } else {
      const s = useChatRuntimeStore.getState();
      initial.current = {
        ctx: s.customContextLength,
        kv: s.kvCacheDtype,
        spec: s.speculativeType,
        draft: s.specDraftNMax,
        tp: s.tensorParallel,
        remember: false,
      };
    }
  }

  const [ctx, setCtx] = useState(initial.current.ctx);
  const [kv, setKv] = useState(initial.current.kv);
  const [spec, setSpec] = useState(initial.current.spec ?? "auto");
  const [draft, setDraft] = useState(initial.current.draft);
  const [tp, setTp] = useState(initial.current.tp);
  const [remember, setRemember] = useState(initial.current.remember);

  const showDraftTokens = spec === "mtp" || spec === "mtp+ngram";
  const modelName = repoId.split("/").pop() ?? repoId;

  // Use the model's native max when known, else the fallback ceiling.
  const ctxMax = maxContext && maxContext > CTX_MIN ? maxContext : CTX_MAX;

  // Warn when weights + KV cache at the chosen context exceed device memory.
  // The KV size is estimated by the backend (architecture-aware); the budget
  // is VRAM plus system RAM, which also covers Mac unified memory (VRAM 0).
  const gpu = useGpuInfo();
  const budgetGb =
    MEM_BUDGET_FRACTION * (gpu.memoryTotalGb + gpu.systemRamAvailableGb);
  const [memWarning, setMemWarning] = useState<{
    neededGb: number;
    budgetGb: number;
  } | null>(null);
  useEffect(() => {
    // Auto (no explicit context) lets the backend fit it; nothing to warn about.
    if (!open || ctx == null || budgetGb <= 0) {
      setMemWarning(null);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      estimateKvCache(repoId, quant, ctx, kv ?? "f16", controller.signal)
        .then((res) => {
          if (res.kv_bytes == null) {
            setMemWarning(null);
            return;
          }
          const neededBytes = (res.weights_bytes ?? 0) + res.kv_bytes;
          const neededGb = neededBytes / 1024 ** 3;
          setMemWarning(neededGb > budgetGb ? { neededGb, budgetGb } : null);
        })
        .catch(() => {
          /* best-effort: no estimate, no warning */
        });
    }, 350);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [open, ctx, kv, repoId, quant, budgetGb]);

  const handleLoad = () => {
    setCustomContextLength(ctx);
    setKvCacheDtype(kv);
    setSpeculativeType(spec);
    setSpecDraftNMax(showDraftTokens ? draft : null);
    setTensorParallel(tp);
    if (remember) {
      saveRememberedLoadSettings(repoId, {
        contextLength: ctx,
        kvCacheDtype: kv,
        speculativeType: spec,
        specDraftNMax: showDraftTokens ? draft : null,
        tensorParallel: tp,
      });
    } else {
      clearRememberedLoadSettings(repoId);
    }
    onLoad();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="corner-squircle dialog-soft-surface dark:!bg-[#202125] sm:max-w-[424px] overflow-hidden p-0 gap-0"
        overlayClassName="bg-black/20 backdrop-blur-none"
      >
        <DialogHeader className="px-8 pt-6 pb-3">
          <DialogTitle className="truncate pr-8 text-[15px]">
            {modelName}{" "}
            <span className="font-normal text-muted-foreground">({quant})</span>
          </DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-3.5 border-t border-border/50 px-8 py-5">
          <Setting
            label="Context Length"
            hint="Max tokens kept in context. Higher needs more memory. By default, Unsloth auto fits on your device."
            control={
              <Input
                type="number"
                min={128}
                step={1}
                value={ctx ?? ""}
                placeholder="auto"
                onChange={(e) => {
                  const raw = e.target.value;
                  if (raw === "") {
                    setCtx(null);
                    return;
                  }
                  const parsed = Number.parseInt(raw, 10);
                  if (Number.isFinite(parsed) && parsed > 0) setCtx(parsed);
                }}
                aria-label="Context Length"
                className={NUMBER_FIELD_CLASS}
              />
            }
          >
            <Slider
              min={CTX_MIN}
              max={ctxMax}
              step={CTX_STEP}
              value={[Math.min(Math.max(ctx ?? CTX_MIN, CTX_MIN), ctxMax)]}
              onValueChange={([v]) => setCtx(Math.round(v))}
              aria-label="Context Length slider"
            />
            <p className="text-[11px] leading-snug text-muted-foreground">
              {maxContext
                ? `Model supports up to ${maxContext.toLocaleString()} tokens.`
                : "Higher limits use more memory."}
            </p>
            {memWarning && (
              <p className="flex items-start gap-1.5 text-[11px] leading-snug text-amber-600 dark:text-amber-500">
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="mt-px size-3.5 shrink-0"
                />
                <span>
                  Needs about {memWarning.neededGb.toFixed(1)} GB, more than the
                  ~{memWarning.budgetGb.toFixed(1)} GB available. Loading may fail
                  or run slowly.
                </span>
              </p>
            )}
          </Setting>

          <Setting
            label="KV Cache Dtype"
            hint="Lower precision saves memory at a small quality cost."
            control={
              <Select
                value={kv ?? "f16"}
                onValueChange={(v) => setKv(v === "f16" ? null : v)}
              >
                <SelectTrigger className={FIELD_CLASS}>
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
            }
          />

          <Setting
            label="Speculative Decoding"
            hint="Faster output, no quality loss. Auto picks the method."
            control={
              <Select
                value={spec}
                onValueChange={(v) => {
                  setSpec(v);
                  if (v !== "mtp" && v !== "mtp+ngram") setDraft(null);
                }}
              >
                <SelectTrigger className={FIELD_CLASS}>
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
            }
          />

          {showDraftTokens && (
            <Setting
              label="Draft Tokens"
              hint="Max MTP draft tokens per step."
              control={
                <Input
                  type="number"
                  min={1}
                  max={16}
                  step={1}
                  value={draft ?? ""}
                  placeholder="auto"
                  onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === "") {
                      setDraft(null);
                      return;
                    }
                    const parsed = Number.parseInt(raw, 10);
                    if (Number.isFinite(parsed)) {
                      setDraft(Math.max(1, Math.min(16, parsed)));
                    }
                  }}
                  aria-label="Draft Tokens"
                  className={NUMBER_FIELD_CLASS}
                />
              }
            />
          )}

          <Setting
            label="Tensor Parallelism"
            hint="Splits dense models across GPUs. No effect on one GPU."
            control={<Switch checked={tp} onCheckedChange={setTp} />}
          />
        </div>

        <label className="flex min-w-0 cursor-pointer items-center gap-2 border-t border-border/50 px-8 py-2.5">
          <Checkbox
            checked={remember}
            onCheckedChange={(v) => setRemember(v === true)}
          />
          <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">
            Remember for{" "}
            <span className="font-mono text-[11px] text-foreground">
              {modelName}
            </span>
          </span>
        </label>

        <div className="flex items-center justify-end gap-2 px-8 pb-6 pt-1">
          <DialogClose asChild={true}>
            <Button variant="ghost" size="sm">
              Cancel
            </Button>
          </DialogClose>
          <Button size="sm" onClick={handleLoad}>
            Load Model
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
