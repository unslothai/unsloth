// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Switch } from "@/components/ui/switch";
import {
  ParamSlider,
  defaultInferenceParams,
  useChatRuntimeStore,
} from "@/features/chat";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useId, useState } from "react";
import { toast } from "sonner";
import { useDocStore } from "../stores/doc-store";
import type { FlowNode, NodeSpec } from "../types";
import {
  editPatch,
  effectiveParams,
  findSaved,
  isOn,
  implicitModel,
  loadPatch,
  modelKey,
  paramsAsSaved,
  togglePatch,
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

export function LlmSettingsPanel({
  node,
  spec,
}: {
  node: FlowNode;
  spec: NodeSpec;
}) {
  const loaded = useChatRuntimeStore((s) =>
    implicitModel(s.residentCheckpoint, s.params.checkpoint),
  );
  const key = modelKey(node.params, loaded);
  const saved = useChatRuntimeStore((s) => findSaved(s.paramsByModel, key));
  const switchId = useId();

  useEffect(() => {
    void useChatRuntimeStore.getState().hydratePersistedSettings();
  }, []);

  const on = isOn(node.params);
  const shown = effectiveParams(node.params, saved, spec);
  const has = (id: string) => spec.params.some((p) => p.id === id);
  const paramSpec = (id: string) => spec.params.find((p) => p.id === id);
  const d = defaultInferenceParams;

  const update = (patch: Record<string, unknown>, label: string) =>
    useDocStore
      .getState()
      .updateParams(node.id, patch, `param:${node.id}:${label}`);
  const edit = (id: string) => (value: unknown) =>
    update(editPatch(node.params, saved, spec, { [id]: value }), id);

  const status = !key
    ? "No model picked or loaded, so this card's own values apply."
    : saved
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
                update(togglePatch(saved, spec, checked === true), "use-saved")
              }
            />
          </div>
          <p className="text-ui-11 leading-snug text-muted-foreground">
            {status}
          </p>
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
                toast.success(`Saved as defaults for ${shortModel(key)}`);
              }}
            >
              Save as model defaults
            </Button>
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={!saved || on}
              title="Copy Chat's saved settings for the model onto this card"
              onClick={() => update(loadPatch(saved, spec), "load-saved")}
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
              <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
                Allow thinking
              </span>
              <Switch
                checked={node.params.thinking === true}
                onCheckedChange={(checked) =>
                  update({ thinking: checked === true }, "thinking")
                }
              />
            </div>
          )}
        </div>
      </Section>

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
