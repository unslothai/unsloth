// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat's model load settings for one card. Chat's own controls live inline in its run-settings page, bound to
// that page's per-model draft, so they are rebuilt here with the same labels, ranges, help text and classes.
// Values are kept in the saved-override shape (`max_seq_length`, `kv_cache_dtype`, ...), which is what Chat
// remembers per model and what the engine loads with.

import { InfoHint } from "@/components/ui/info-hint";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
// eslint-disable-next-line no-restricted-imports -- constants only, not exported by the barrel.
import {
  formatExtraArgs,
  parseExtraArgs,
} from "@/features/model-picker/model-config/llama-extra-args";
// eslint-disable-next-line no-restricted-imports -- constants only, not exported by the barrel.
import {
  CACHE_RAM_LLAMA_DEFAULT,
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CTX_CHECKPOINTS_LLAMA_DEFAULT,
  CTX_CHECKPOINTS_MAX,
  CTX_CHECKPOINTS_MIN,
  KV_CACHE_DTYPES,
  LOAD_MODES,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MLX_KV_QUANTS,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
} from "@/features/model-picker/model-config/per-model-config";
import {
  DRAFT_N_MAX_SPEC_TYPES,
  SEPARATE_DRAFT_MODEL_SPEC_TYPES,
  SPECULATIVE_TYPES,
} from "@/lib/speculative-modes";
import { type ReactNode, useEffect, useState } from "react";
import { CONTEXT_PIN, type LoadSettings, contextPin } from "./llm-settings";

const ROW_CLASS = "flex min-h-8 items-center justify-between gap-3";
const LABEL_CLASS =
  "min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground";
const INPUT_CLASS =
  "panel-field h-8 w-[calc(84px*var(--ui-space-scale,1))] shrink-0";
const SELECT_TRIGGER_CLASS =
  "panel-select-trigger grid h-8! w-auto max-w-full min-w-0 shrink-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 [&_[data-slot=select-value]]:min-w-0 [&_[data-slot=select-value]]:truncate [&>svg]:shrink-0";

const SPECULATIVE_LABELS: Record<string, string> = {
  auto: "Auto",
  mtp: "MTP",
  dspark: "DSpark",
  dflash: "DFlash",
  ngram: "Ngram",
  "mtp+ngram": "MTP+Ngram",
  off: "Off",
};
const LOAD_MODE_LABELS: Record<string, string> = {
  auto: "Auto",
  none: "None",
  mmap: "mmap",
  mlock: "mlock",
  "mmap+mlock": "mmap+mlock",
  dio: "DirectIO",
};
const DEFAULT = "__default__";

function Row({
  label,
  info,
  children,
}: {
  label: string;
  info?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={ROW_CLASS}>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className={LABEL_CLASS}>{label}</span>
        {info && <InfoHint>{info}</InfoHint>}
      </div>
      {children}
    </div>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section aria-label={title} className="space-y-3 pt-2">
      <h3 className="text-ui-10 font-semibold uppercase leading-none tracking-wider text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

/** Blank is the default; a typed value is clamped on blur. */
function NumberField({
  label,
  value,
  min,
  max,
  placeholder,
  onChange,
}: {
  label: string;
  value: unknown;
  min: number;
  max: number;
  placeholder: string;
  onChange: (value: number | undefined) => void;
}) {
  const shown = typeof value === "number" ? String(value) : "";
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <input
      type="text"
      inputMode="numeric"
      aria-label={label}
      value={draft ?? shown}
      placeholder={placeholder}
      onChange={(e) => setDraft(e.target.value.replace(/[^\d-]/g, ""))}
      onBlur={() => {
        if (draft === null) return;
        setDraft(null);
        const n = Number.parseInt(draft, 10);
        onChange(
          Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : undefined,
        );
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
      }}
      className={INPUT_CLASS}
    />
  );
}

function Choice({
  label,
  value,
  options,
  defaultLabel,
  onChange,
}: {
  label: string;
  value: unknown;
  options: readonly (readonly [string, string])[];
  defaultLabel: string;
  onChange: (value: string | undefined) => void;
}) {
  return (
    <Select
      value={typeof value === "string" ? value : DEFAULT}
      onValueChange={(next) => onChange(next === DEFAULT ? undefined : next)}
    >
      <SelectTrigger aria-label={label} className={SELECT_TRIGGER_CLASS}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={DEFAULT}>{defaultLabel}</SelectItem>
        {options.map(([v, text]) => (
          <SelectItem key={v} value={v}>
            {text}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ExtraArgsField({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (value: string[] | undefined) => void;
}) {
  const stored = Array.isArray(value) ? (value as string[]) : null;
  const [text, setText] = useState(formatExtraArgs(stored));
  useEffect(() => setText(formatExtraArgs(stored)), [stored]);
  const parsed = parseExtraArgs(text);
  return (
    <div className="flex flex-col gap-1">
      <textarea
        value={text}
        rows={2}
        spellCheck={false}
        aria-label="Extra llama-server arguments"
        placeholder="--flag value"
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          if (parsed.unterminatedQuote) return;
          onChange(parsed.tokens.length > 0 ? parsed.tokens : undefined);
        }}
        className="panel-field min-h-16 w-full resize-y rounded-xl px-3 py-2 font-mono text-ui-12"
      />
      {parsed.unterminatedQuote && (
        <p className="text-ui-11 text-destructive">
          Close the {parsed.unterminatedQuote} quote.
        </p>
      )}
    </div>
  );
}

export function LlmLoadSettings({
  load,
  isGguf,
  isMac,
  onChange,
}: {
  load: LoadSettings;
  /** null when the card's model is not known yet: every GGUF control shows. */
  isGguf: boolean | null;
  isMac: boolean;
  onChange: (key: string, value: unknown) => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const set = (key: string) => (value: unknown) => onChange(key, value);
  const gguf = isGguf !== false;
  const manual = load.gpu_memory_mode === "manual";
  const spec = typeof load.speculative_type === "string" ? load.speculative_type : "";
  const kvOptions = KV_CACHE_DTYPES.map((d) => [d, d] as const);

  return (
    <div className="flex flex-col gap-5">
      <Row
        label="Context Length"
        info="Maximum context window in tokens. Applies on load."
      >
        <NumberField
          label="Context Length"
          value={contextPin(load)}
          min={MAX_SEQ_LENGTH_MIN}
          max={MAX_SEQ_LENGTH_MAX}
          placeholder="Auto"
          onChange={(value) => onChange(CONTEXT_PIN, value)}
        />
      </Row>
      {gguf && (
        <Row
          label="GPU Memory"
          info={
            <div className="flex flex-col gap-1.5">
              <div>
                <span className="font-medium">Default:</span> Unsloth fits the
                model and context to your GPUs.
              </div>
              <div>
                <span className="font-medium">Manual:</span> set GPU Layers
                yourself.
              </div>
            </div>
          }
        >
          <Choice
            label="GPU Memory"
            value={load.gpu_memory_mode}
            defaultLabel="Default"
            options={[["manual", "Manual"]]}
            onChange={set("gpu_memory_mode")}
          />
        </Row>
      )}
      {gguf && manual && (
        <>
          <Row
            label="GPU Layers"
            info="Layers to keep on the GPU (--gpu-layers); the rest run on CPU. Auto sizes the split to fit VRAM."
          >
            <NumberField
              label="GPU Layers"
              value={load.gpu_layers}
              min={0}
              max={1024}
              placeholder="Auto"
              onChange={set("gpu_layers")}
            />
          </Row>
          <Row
            label="MoE Layers on CPU"
            info="Keep the experts of this many MoE layers on the CPU (--n-cpu-moe) to save VRAM. 0 keeps every expert on the GPU."
          >
            <NumberField
              label="MoE Layers on CPU"
              value={load.n_cpu_moe}
              min={0}
              max={1024}
              placeholder="0"
              onChange={(v) => onChange("n_cpu_moe", v ? v : undefined)}
            />
          </Row>
        </>
      )}
      {isGguf === false && (
        <Row
          label="Load in 4-bit"
          info="Quantizes the model to 4 bits as it loads, using about a quarter of the memory. Turn it off to run the model in 16-bit, the same precision as a 16-bit LoRA or full fine-tune."
        >
          <Switch
            className="panel-switch shrink-0"
            aria-label="Load in 4-bit"
            checked={load.load_in_4bit !== false}
            onCheckedChange={(on) =>
              onChange("load_in_4bit", on ? undefined : false)
            }
          />
        </Row>
      )}

      <div className={`${ROW_CLASS} border-t border-border pt-5`}>
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-muted-foreground">
            Advanced settings
          </span>
          <InfoHint>
            Extra options for how the model loads. Unsloth already picks the
            best settings for your device, so most setups don't need these.
          </InfoHint>
        </div>
        <Switch
          className="panel-switch shrink-0"
          checked={advanced}
          onCheckedChange={setAdvanced}
          aria-label="Show advanced settings"
        />
      </div>

      {advanced && (
        <>
          <Group title="Memory">
            {gguf && (
              <Row
                label="KV Cache Dtype"
                info="Lower KV cache precision to save VRAM, at some cost to quality. f16 is the default; q8_0 through iq4_nl are quantized."
              >
                <Choice
                  label="KV Cache Dtype"
                  value={load.kv_cache_dtype}
                  defaultLabel="f16"
                  options={kvOptions}
                  onChange={set("kv_cache_dtype")}
                />
              </Row>
            )}
            {isMac && (
              <Row
                label="MLX KV Cache"
                info="Lower KV cache precision to save memory, at some cost to quality. Auto keeps full precision; 8-bit is the safest step down."
              >
                <Choice
                  label="MLX KV Cache"
                  value={load.mlx_kv_quant}
                  defaultLabel="Auto"
                  options={MLX_KV_QUANTS.map((q) => [q, q] as const)}
                  onChange={set("mlx_kv_quant")}
                />
              </Row>
            )}
            {gguf && (
              <Row
                label="Mmap/Mlock"
                info="How the weights are read off disk (--load-mode). Auto is the default and lets Unsloth pick. mmap maps the file, mlock keeps the model in RAM, DirectIO streams it, and None asks for no special mode."
              >
                <Choice
                  label="Mmap/Mlock"
                  value={load.load_mode}
                  defaultLabel="Auto"
                  options={LOAD_MODES.filter((m) => m !== "auto").map(
                    (m) => [m, LOAD_MODE_LABELS[m] ?? m] as const,
                  )}
                  onChange={set("load_mode")}
                />
              </Row>
            )}
            <Row
              label="Vision"
              info="Loads the vision projector so the model can read images. Turning it off frees that VRAM for more layers on the GPU. Text generation is unaffected either way."
            >
              <Switch
                className="panel-switch shrink-0"
                aria-label="Vision"
                checked={load.disable_vision !== true}
                onCheckedChange={(on) =>
                  onChange("disable_vision", on ? undefined : true)
                }
              />
            </Row>
          </Group>

          <Group title="Speed">
            <Row
              label="Speculative Decoding"
              info="Faster generation. Auto picks the best strategy for the model and platform, or choose one to force it. DSpark and DFlash download a drafter sidecar (about 11 GB and 1.5 GB) and trade VRAM for speed; MTP and ngram do not change output."
            >
              <Choice
                label="Speculative Decoding"
                value={load.speculative_type}
                defaultLabel="Auto"
                options={SPECULATIVE_TYPES.filter((t) => t !== "auto").map(
                  (t) => [t, SPECULATIVE_LABELS[t] ?? t] as const,
                )}
                onChange={set("speculative_type")}
              />
            </Row>
            {DRAFT_N_MAX_SPEC_TYPES.has(spec) && (
              <Row
                label="Draft Tokens"
                info="Max draft tokens per step. Leave blank for the default (2 or 3, depending on the strategy and device)."
              >
                <NumberField
                  label="Draft Tokens"
                  value={load.spec_draft_n_max}
                  min={1}
                  max={16}
                  placeholder="Auto"
                  onChange={set("spec_draft_n_max")}
                />
              </Row>
            )}
            {SEPARATE_DRAFT_MODEL_SPEC_TYPES.has(spec) && (
              <Row
                label="Spec Decoding KV Cache Dtype"
                info="KV cache precision for the draft model's own context, separate from the KV Cache Dtype above. f16 is the default; quantizing it saves VRAM on a drafter the target verifies anyway."
              >
                <Choice
                  label="Spec Decoding KV Cache Dtype"
                  value={load.spec_draft_cache_type}
                  defaultLabel="f16"
                  options={kvOptions}
                  onChange={set("spec_draft_cache_type")}
                />
              </Row>
            )}
            <Row
              label="Parallel Slots"
              info="Decode slots (--parallel) for concurrent requests. Leave blank for the server default. More slots share the context pool and use more VRAM."
            >
              <NumberField
                label="Parallel Slots"
                value={load.n_parallel}
                min={N_PARALLEL_MIN}
                max={N_PARALLEL_MAX}
                placeholder="Auto"
                onChange={set("n_parallel")}
              />
            </Row>
            {gguf && (
              <>
                <Row
                  label="Batch Size"
                  info="Logical prompt batch size (--batch-size). Leave blank for the default (2048). The micro-batch below usually matters more."
                >
                  <NumberField
                    label="Batch Size"
                    value={load.n_batch}
                    min={N_BATCH_MIN}
                    max={N_BATCH_MAX}
                    placeholder="2048"
                    onChange={set("n_batch")}
                  />
                </Row>
                <Row
                  label="Micro-batch Size"
                  info="Physical prompt micro-batch size (--ubatch-size). Leave blank for the default (512, or 1120 on Gemma 4 vision models). Larger values speed up prompt processing but use more VRAM; capped at the batch size."
                >
                  <NumberField
                    label="Micro-batch Size"
                    value={load.n_ubatch}
                    min={N_BATCH_MIN}
                    max={N_BATCH_MAX}
                    placeholder="512"
                    onChange={set("n_ubatch")}
                  />
                </Row>
                <Row
                  label="Tensor Parallelism"
                  info="Speeds up dense models across multiple GPUs. No effect on a single GPU, and MoE models don't benefit."
                >
                  <Switch
                    className="panel-switch shrink-0"
                    aria-label="Tensor Parallelism"
                    checked={load.tensor_parallel === true}
                    onCheckedChange={(on) =>
                      onChange("tensor_parallel", on ? true : undefined)
                    }
                  />
                </Row>
              </>
            )}
            {isMac && (
              <Row
                label="Int8 Prefill"
                info="Reads long prompts faster by doing part of the math at lower precision. Answers can change and may be less accurate."
              >
                <Switch
                  className="panel-switch shrink-0"
                  aria-label="Int8 Prefill"
                  checked={load.mlx_int8_prefill === true}
                  onCheckedChange={(on) =>
                    onChange("mlx_int8_prefill", on ? true : undefined)
                  }
                />
              </Row>
            )}
          </Group>

          <Group title="Output">
            <Row
              label="Reasoning Budget"
              info="Maximum thinking tokens. Leave blank for unlimited; 0 turns reasoning off."
            >
              <NumberField
                label="Reasoning Budget"
                value={load.reasoning_budget === -1 ? undefined : load.reasoning_budget}
                min={0}
                max={2_147_483_647}
                placeholder="unlimited"
                onChange={set("reasoning_budget")}
              />
            </Row>
            <Row
              label="Reasoning Budget Message"
              info="Optional text added before the end-of-thinking tag when the budget runs out."
            >
              <input
                type="text"
                aria-label="Reasoning Budget Message"
                value={String(load.reasoning_budget_message ?? "")}
                placeholder="None"
                onChange={(e) =>
                  onChange("reasoning_budget_message", e.target.value)
                }
                className={`${INPUT_CLASS} min-w-0`}
              />
            </Row>
            <div className="flex flex-col gap-2">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className={LABEL_CLASS}>Chat Template</span>
                <InfoHint>
                  Replace the model's chat template with custom Jinja. Applies
                  on load.
                </InfoHint>
              </div>
              <textarea
                value={String(load.chat_template_override ?? "")}
                rows={3}
                spellCheck={false}
                aria-label="Chat Template"
                placeholder="The model's own template"
                onChange={(e) =>
                  onChange("chat_template_override", e.target.value)
                }
                className="panel-field min-h-16 w-full resize-y rounded-xl px-3 py-2 font-mono text-ui-12"
              />
            </div>
          </Group>

          {gguf && (
            <Group title="Server">
              <Row
                label="Checkpoints"
                info={`Checkpoints kept per slot (--ctx-checkpoints), which let a sliding-window model rewind instead of re-processing the prompt. Leave blank for the default (${CTX_CHECKPOINTS_LLAMA_DEFAULT}); 0 disables them. Each one costs host memory.`}
              >
                <NumberField
                  label="Context checkpoints per slot"
                  value={load.ctx_checkpoints}
                  min={CTX_CHECKPOINTS_MIN}
                  max={CTX_CHECKPOINTS_MAX}
                  placeholder={String(CTX_CHECKPOINTS_LLAMA_DEFAULT)}
                  onChange={set("ctx_checkpoints")}
                />
              </Row>
              <Row
                label="Cache RAM"
                info={`Host memory in MiB for caching prompt state evicted from a slot (--cache-ram), so a returning conversation is not re-processed. Leave blank for the default (${CACHE_RAM_LLAMA_DEFAULT}); 0 disables it and -1 lifts the limit.`}
              >
                <NumberField
                  label="Host prompt cache size in MiB"
                  value={load.cache_ram}
                  min={CACHE_RAM_MIN}
                  max={CACHE_RAM_MAX}
                  placeholder={String(CACHE_RAM_LLAMA_DEFAULT)}
                  onChange={set("cache_ram")}
                />
              </Row>
              <div className="flex flex-col gap-2">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className={LABEL_CLASS}>Extra Arguments</span>
                  <InfoHint>
                    Passed straight to llama-server after the settings above, so
                    anything set in both is taken from here.
                  </InfoHint>
                </div>
                <ExtraArgsField
                  value={load.llama_extra_args}
                  onChange={set("llama_extra_args")}
                />
              </div>
            </Group>
          )}
        </>
      )}
    </div>
  );
}
