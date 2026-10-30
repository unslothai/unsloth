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
import { usePlatformStore } from "@/config/env";
import {
  ParamSlider,
  defaultInferenceParams,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  type ApiModelOverrides,
  fetchModelOverrides,
  modelOverrideKey,
  syncModelOverride,
} from "@/features/model-picker";
// eslint-disable-next-line no-restricted-imports -- the override <-> config mapping Chat's own panel uses.
import {
  fromApiOverride,
  resolveStoredOverride,
} from "@/features/model-picker/api/model-overrides";
// eslint-disable-next-line no-restricted-imports -- the per-model store Chat's run settings write.
import { savePerModelConfig } from "@/features/model-picker/model-config/per-model-config";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useId, useState } from "react";
import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import type { FlowNode, ModelChoice, NodeSpec } from "../types";
import { LlmLoadSettings } from "./llm-load-settings";
import {
  type LoadSettings,
  editPatch,
  effectiveLoad,
  effectiveParams,
  findSaved,
  isOn,
  implicitModel,
  loadPatch,
  modelKey,
  paramsAsSaved,
  reloadHint,
  togglePatch,
  withLoadField,
} from "./llm-settings";
import { ParamField } from "./param-fields";

// Chat's own section header, so the panel reads as the same one.
function Section({
  label,
  children,
  defaultOpen = true,
  first = false,
}: {
  label: string;
  children: ReactNode;
  defaultOpen?: boolean;
  first?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      className={cn(
        !first &&
          "border-t border-[rgb(0_0_0_/_calc(0.13*var(--contrast-edge-gain,1)))] dark:border-[rgb(255_255_255_/_calc(0.09*var(--contrast-edge-gain,1)))]",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className={cn(
          "flex w-full cursor-pointer items-center justify-between text-ui-12 font-medium normal-case tracking-[0.04em] text-nav-fg-muted transition-colors hover:text-nav-fg",
          first ? "pt-1 pb-4" : "py-4",
        )}
      >
        <span className="leading-none">{label}</span>
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={cn("size-3.5", open ? "rotate-0" : "-rotate-90")}
        />
      </button>
      {open && <div className="pb-4">{children}</div>}
    </div>
  );
}

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function shortModel(key: string): string {
  return key.split("/").pop() ?? key;
}

const SAMPLING_KEYS = [
  "top_p",
  "top_k",
  "min_p",
  "repetition_penalty",
  "presence_penalty",
] as const;

const EFFORTS = [
  ["low", "Low"],
  ["medium", "Medium"],
  ["high", "High"],
  ["max", "Max"],
] as const;
const EFFORT_DEFAULT = "__model__";

function cardModel(node: FlowNode): ModelChoice | null {
  const choice = node.params.model as ModelChoice | null | undefined;
  return choice?.repo?.trim() ? choice : null;
}

// One read shared by every card on the canvas; a save drops it.
let overridesRead: Promise<ApiModelOverrides> | null = null;
const overrideListeners = new Set<() => void>();

function readOverrides(): Promise<ApiModelOverrides> {
  overridesRead ??= fetchModelOverrides().catch(() => {
    overridesRead = null;
    return {};
  });
  return overridesRead;
}

function overridesChanged() {
  overridesRead = null;
  for (const listener of overrideListeners) listener();
}

/** Chat's saved launch settings for the model, as the engine reads them. */
function useSavedLoad(key: string | null, variant: string | null) {
  // Tagged with the model it was read for, so a switch never shows the last model's while the read is out.
  const at = `${key ?? ""}\u0000${variant ?? ""}`;
  const [saved, setSaved] = useState<{
    at: string;
    load: LoadSettings | null;
  } | null>(null);
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((v) => v + 1);
    overrideListeners.add(bump);
    return () => {
      overrideListeners.delete(bump);
    };
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: version re-reads after a save
  useEffect(() => {
    if (!key) {
      setSaved(null);
      return;
    }
    let alive = true;
    void readOverrides().then((overrides) => {
      const keys = variant ? [modelOverrideKey(key, variant), key] : [key];
      const found = resolveStoredOverride(overrides, keys);
      if (alive) setSaved({ at, load: found ? { ...found } : null });
    });
    return () => {
      alive = false;
    };
  }, [key, variant, version]);
  return saved?.at === at ? saved.load : null;
}

/** The loaded model as Chat's status reports it: the resident one, not Chat's picker. */
function useLoadedModel() {
  const checkpoint = useChatRuntimeStore((s) =>
    implicitModel(s.residentCheckpoint, s.params.checkpoint),
  );
  const ggufVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const contextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const kvCacheDtype = useChatRuntimeStore((s) => s.loadedKvCacheDtype);
  const nParallel = useChatRuntimeStore((s) => s.loadedNParallel);
  const speculativeType = useChatRuntimeStore((s) => s.loadedSpeculativeType);
  return {
    checkpoint: checkpoint || null,
    ggufVariant,
    contextLength,
    kvCacheDtype,
    nParallel,
    speculativeType,
  };
}

/** The card face's one quiet line when running it would reload the model. */
export function LlmReloadHint({ node }: { node: FlowNode }) {
  const loaded = useLoadedModel();
  const loadedVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const choice = cardModel(node);
  const key = isOn(node.params) ? modelKey(node.params, loaded.checkpoint) : null;
  const savedLoad = useSavedLoad(
    key,
    choice ? (choice.variant ?? null) : loadedVariant,
  );
  const hint = reloadHint(
    { ...node.params, load: effectiveLoad(node.params, savedLoad) },
    loaded,
  );
  if (!hint) return null;
  return (
    <p className="flow-detail px-3 text-ui-10p5 leading-snug text-muted-foreground">
      {hint}
    </p>
  );
}

export function LlmSettingsPanel({
  node,
  spec,
}: {
  node: FlowNode;
  spec: NodeSpec;
}) {
  const loadedModel = useLoadedModel();
  const loadedVariant = useChatRuntimeStore((s) => s.activeGgufVariant);
  const loadedIsGguf = useChatRuntimeStore((s) => s.loadedIsGguf);
  const isMac = usePlatformStore((s) => s.deviceType === "mac");
  const choice = cardModel(node);
  const key = modelKey(node.params, loadedModel.checkpoint);
  const variant = choice ? (choice.variant ?? null) : loadedVariant;
  const saved = useChatRuntimeStore((s) => findSaved(s.paramsByModel, key));
  const savedLoad = useSavedLoad(key, variant);
  const switchId = useId();
  const isGguf = choice
    ? Boolean(choice.variant) || /gguf/i.test(choice.repo)
    : loadedIsGguf;

  useEffect(() => {
    void useChatRuntimeStore.getState().hydratePersistedSettings();
  }, []);

  const on = isOn(node.params);
  const shown = effectiveParams(node.params, saved, spec, savedLoad);
  const shownLoad = (shown.load as LoadSettings | null) ?? {};
  const hint = reloadHint(shown, loadedModel);
  const has = (id: string) => spec.params.some((p) => p.id === id);
  const paramSpec = (id: string) => spec.params.find((p) => p.id === id);
  const d = defaultInferenceParams;

  const update = (patch: Record<string, unknown>, label: string) =>
    useDocStore
      .getState()
      .updateParams(node.id, patch, `param:${node.id}:${label}`);
  const edit = (id: string) => (value: unknown) =>
    update(editPatch(node.params, saved, spec, { [id]: value }, savedLoad), id);
  const editLoad = (field: string, value: unknown) =>
    update(
      editPatch(
        node.params,
        saved,
        spec,
        { load: withLoadField(shownLoad, field, value) },
        savedLoad,
      ),
      `load:${field}`,
    );
  const hasSaved = !!saved || !!savedLoad;

  const status = !key
    ? "No model picked or loaded, so this card's own values apply."
    : hasSaved
      ? on
        ? `Following Chat's saved settings for ${shortModel(key)}.`
        : `Chat has saved settings for ${shortModel(key)}.`
      : `Nothing saved for ${shortModel(key)} yet, so this card's own values apply.`;

  // Auto: unset, so the server uses the model's recommended value.
  const auto = (id: string) =>
    shown[id] === null || shown[id] === undefined ? "Auto" : undefined;

  const maxTokens = paramSpec("max_tokens");
  const seed = paramSpec("seed");
  const schema = paramSpec("json_schema");

  return (
    <section
      className="flex flex-col border-border/60 border-t px-4 pt-3"
      data-testid="flows-llm-settings"
    >
      <Section label="Model settings" first={true}>
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-1.5">
              <label
                htmlFor={switchId}
                className="cursor-pointer text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg"
              >
                Use the model's saved settings
              </label>
              <InfoHint>
                Reads the settings Chat saved for this model, live. Change any
                value below to give this card its own copy.
              </InfoHint>
            </div>
            <Switch
              id={switchId}
              checked={on}
              onCheckedChange={(checked) =>
                update(
                  togglePatch(
                    node.params,
                    saved,
                    spec,
                    checked === true,
                    savedLoad,
                  ),
                  "use-saved",
                )
              }
            />
          </div>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {status}
          </p>
          {hint && (
            <p className="text-ui-11 leading-snug text-amber-600 dark:text-amber-400">
              {hint}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={!key || on}
              title="Save this card's settings as Chat's defaults for the model"
              onClick={() => {
                if (!key) return;
                useChatRuntimeStore
                  .getState()
                  .setRememberedParamsForModel(key, paramsAsSaved(node.params));
                const own = node.params.load as LoadSettings | null;
                if (own && Object.keys(own).length > 0) {
                  // Over what is saved, so a field this card leaves unset keeps Chat's value.
                  const config = fromApiOverride({ ...savedLoad, ...own });
                  if (savePerModelConfig(key, variant, config))
                    syncModelOverride(key, variant, config);
                  setTimeout(overridesChanged, 500);
                }
                toast.success(`Saved as defaults for ${shortModel(key)}`);
              }}
            >
              Save as model defaults
            </Button>
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={!hasSaved || on}
              title="Copy Chat's saved settings for the model onto this card"
              onClick={() =>
                update(loadPatch(saved, spec, savedLoad), "load-saved")
              }
            >
              Load model defaults
            </Button>
          </div>
        </div>
      </Section>

      {has("system") && (
        <Section label="System Prompt">
          <div className="panel-text-surface h-24 w-full overflow-hidden corner-squircle">
            <textarea
              value={String(shown.system ?? "")}
              onChange={(e) => edit("system")(e.target.value)}
              placeholder="Example: You are a helpful assistant..."
              aria-label="System prompt"
              className="block size-full resize-none bg-transparent px-3.5 py-2.5 text-left text-ui-13 font-medium leading-relaxed text-nav-fg outline-none placeholder:text-muted-foreground"
            />
          </div>
        </Section>
      )}

      <Section label="Sampling">
        <div className="flex flex-col gap-5">
          <ParamSlider
            label="Temperature"
            value={num(shown.temperature, 0.8)}
            min={0}
            max={2}
            step={0.01}
            onChange={edit("temperature")}
            info="Controls randomness. Lower values make output focused and deterministic; higher values increase variety and creativity."
          />
          <ParamSlider
            label="Top P"
            value={num(shown.top_p, d.topP)}
            min={0}
            max={1}
            step={0.05}
            onChange={edit("top_p")}
            format={(v) => auto("top_p") ?? (v === 1 ? "Off" : undefined)}
            info="Nucleus sampling. Restricts choices to the smallest set of tokens whose cumulative probability reaches this threshold. 1.0 = off."
          />
          <ParamSlider
            label="Top K"
            value={num(shown.top_k, d.topK)}
            min={0}
            max={100}
            step={1}
            onChange={edit("top_k")}
            format={(v) => auto("top_k") ?? (v === 0 ? "Off" : undefined)}
            info="Limits sampling to the K most likely tokens at each step. 0 = off."
          />
          <ParamSlider
            label="Min P"
            value={num(shown.min_p, d.minP)}
            min={0}
            max={1}
            step={0.01}
            onChange={edit("min_p")}
            format={() => auto("min_p")}
            info="Drops tokens whose probability is below this fraction of the top token's probability. Filters unlikely candidates."
          />
          <ParamSlider
            label="Repetition Penalty"
            value={num(shown.repetition_penalty, d.repetitionPenalty)}
            min={1}
            max={2}
            step={0.05}
            onChange={edit("repetition_penalty")}
            format={(v) =>
              auto("repetition_penalty") ?? (v === 1 ? "Off" : undefined)
            }
            info="Down-weights tokens that have already appeared, reducing repetition. 1.0 = off; higher values penalize more strongly."
          />
          <ParamSlider
            label="Presence Penalty"
            value={num(shown.presence_penalty, d.presencePenalty)}
            min={0}
            max={2}
            step={0.1}
            onChange={edit("presence_penalty")}
            format={(v) =>
              auto("presence_penalty") ?? (v === 0 ? "Off" : undefined)
            }
            info="Penalizes any token that has already appeared at least once, encouraging the model to introduce new topics. 0 = off."
          />
          {maxTokens && (
            <ParamSlider
              label="Max Tokens"
              value={num(shown.max_tokens, num(maxTokens.default, 1024))}
              min={64}
              max={maxTokens.max ?? 32768}
              step={64}
              valueSize={5}
              onChange={edit("max_tokens")}
              info="Maximum number of tokens to generate. Generation stops at this limit or when the model emits an end-of-sequence token."
            />
          )}
          {seed && (
            <div className="flex flex-col gap-2">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
                  Seed
                </span>
                <InfoHint>
                  Blank follows the flow's seed, so Re-roll draws a new one.
                  Kept on the card, never read from or saved to the model.
                </InfoHint>
              </div>
              <ParamField
                spec={seed}
                value={node.params.seed}
                onChange={(value) =>
                  update({ seed: value }, "seed")
                }
              />
            </div>
          )}
          {has("thinking") && (
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
                  Thinking
                </span>
                <InfoHint>
                  Let the model reason before it answers, as the Thinking
                  button in Chat's composer does. Off for JSON output.
                </InfoHint>
              </div>
              <Switch
                aria-label="Thinking"
                checked={node.params.thinking === true}
                onCheckedChange={(checked) =>
                  update({ thinking: checked === true }, "thinking")
                }
              />
            </div>
          )}
          {has("reasoning_effort") && node.params.thinking === true && (
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
                  Reasoning Effort
                </span>
                <InfoHint>
                  How hard the model thinks, for models that take a level
                  (gpt-oss, GLM). Others ignore it.
                </InfoHint>
              </div>
              <Select
                value={
                  typeof node.params.reasoning_effort === "string" &&
                  node.params.reasoning_effort
                    ? node.params.reasoning_effort
                    : EFFORT_DEFAULT
                }
                onValueChange={(next) =>
                  update(
                    {
                      reasoning_effort: next === EFFORT_DEFAULT ? null : next,
                    },
                    "reasoning_effort",
                  )
                }
              >
                <SelectTrigger
                  aria-label="Reasoning Effort"
                  className="panel-select-trigger h-8! w-auto"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={EFFORT_DEFAULT}>Model default</SelectItem>
                  {EFFORTS.map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="text-muted-foreground"
              title="Back to Auto: the model's saved or recommended sampling"
              onClick={() =>
                update(
                  editPatch(
                    node.params,
                    saved,
                    spec,
                    Object.fromEntries(SAMPLING_KEYS.map((k) => [k, null])),
                    savedLoad,
                  ),
                  "reset-sampling",
                )
              }
            >
              Reset
            </Button>
          </div>
        </div>
      </Section>

      {has("load") && (
        <Section label="Model Load" defaultOpen={false}>
          <LlmLoadSettings
            load={shownLoad}
            isGguf={isGguf}
            isMac={isMac}
            onChange={editLoad}
          />
        </Section>
      )}

      {schema && (
        <Section label="Structured Output" defaultOpen={!!node.params.json_schema}>
          <div className="flex flex-col gap-1.5">
            <p className="text-ui-11 leading-snug text-muted-foreground">
              A JSON schema makes the model answer with matching JSON on the
              JSON output.
            </p>
            <ParamField
              spec={schema}
              value={node.params.json_schema}
              onChange={(value) => update({ json_schema: value }, "json_schema")}
            />
          </div>
        </Section>
      )}
    </section>
  );
}
