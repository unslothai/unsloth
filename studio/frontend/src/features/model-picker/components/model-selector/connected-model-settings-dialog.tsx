// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// System prompt and output cap reuse Chat's per-model memory under the same checkpoint id.

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ProviderApiType } from "@/features/chat/api/providers-api";
// eslint-disable-next-line no-restricted-imports -- Connection contract has no React dependencies.
import type { CustomReasoningConfig } from "@/features/chat/custom-reasoning";
import {
  modelCatalogVersion,
  reconcilePinnedReasoningEffort,
  subscribeModelCatalog,
  useChatRuntimeStore,
} from "@/features/chat";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  externalReasoningTakesEffort,
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  getExternalReasoningCapabilities,
} from "@/features/chat/provider-capabilities";
import { useState, useSyncExternalStore } from "react";
import { useModelReasoningEffortStore } from "./model-reasoning-effort";

/** A Select cannot carry an empty value, so absence needs a name. */
const FOLLOW_CHAT = "__follow_chat__";

export function ConnectedModelSettingsDialog({
  open,
  onOpenChange,
  checkpointId,
  displayName,
  modelId,
  providerType,
  apiType,
  baseUrl,
  isReasoningProvider,
  reasoningConfig,
  connectionMaxOutputTokens,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The `external::` id, which is what both memories key on. */
  checkpointId: string;
  displayName: string;
  modelId: string;
  providerType: string;
  apiType?: ProviderApiType;
  baseUrl?: string | null;
  isReasoningProvider?: boolean;
  reasoningConfig?: CustomReasoningConfig;
  /** The connection's own output cap, which lowers the model's documented one. */
  connectionMaxOutputTokens?: number | null;
}) {
  const remembered = useChatRuntimeStore(
    (state) => state.paramsByModel[checkpointId],
  );
  const rememberParamsPerModel = useChatRuntimeStore(
    (state) => state.rememberParamsPerModel,
  );
  const setRememberedParamsForModel = useChatRuntimeStore(
    (state) => state.setRememberedParamsForModel,
  );
  const chatMaxTokens = useChatRuntimeStore((state) => state.params.maxTokens);
  // A live model's edits must reach the live settings as well as the memory.
  const isLiveModel = useChatRuntimeStore(
    (state) => state.params.checkpoint === checkpointId,
  );
  const pinnedEffort = useModelReasoningEffortStore(
    (state) => state.effortByModel[checkpointId],
  );
  const setModelReasoningEffort = useModelReasoningEffortStore(
    (state) => state.setModelReasoningEffort,
  );

  const reasoning = getExternalReasoningCapabilities(providerType, modelId, {
    isReasoningProvider,
    reasoningConfig,
    baseUrl,
    apiType,
  });
  // Only where a level is actually sent: the default ladder exists even for on/off-only styles.
  const efforts = externalReasoningTakesEffort(reasoning)
    ? reasoning.reasoningEffortLevels.filter((level) => level !== "none")
    : [];

  // The OpenRouter cap comes from the live catalogue, which can land after this renders.
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  // Same bounds the adapter clamps every request to.
  const minCap = getExternalMinOutputTokens(providerType);
  const maxCap = getExternalMaxOutputTokens(
    providerType,
    modelId,
    connectionMaxOutputTokens,
  );
  const clampCap = (value: number) =>
    Math.min(Math.max(value, minCap), maxCap);

  // null means untouched: keeps reading the store and Save writes nothing, so a pre-hydration
  // open or another tab's change is not overwritten.
  const [promptDraft, setPromptDraft] = useState<string | null>(null);
  const [capDraft, setCapDraft] = useState<string | null>(null);
  const [effortDraft, setEffortDraft] = useState<string | null>(null);
  const systemPrompt = promptDraft ?? remembered?.systemPrompt ?? "";
  // A level the ladder dropped reverts to stored / reads as Follow chat, so Save never writes it.
  const offered = (level: string): boolean =>
    (efforts as readonly string[]).includes(level);
  const liveEffortDraft =
    effortDraft !== null && (effortDraft === FOLLOW_CHAT || offered(effortDraft))
      ? effortDraft
      : null;
  const effort =
    liveEffortDraft ??
    (pinnedEffort && offered(pinnedEffort) ? pinnedEffort : FOLLOW_CHAT);
  // Always a real number: per-key merges cannot express clearing the cap.
  const maxTokens =
    capDraft ?? String(clampCap(remembered?.maxTokens ?? chatMaxTokens));

  function save() {
    // Number, not parseInt: parseInt reads 1e5 as 1.
    const typedCap = Math.round(Number(maxTokens.trim()));
    // Patch only touched fields.
    setRememberedParamsForModel(checkpointId, {
      ...(promptDraft !== null ? { systemPrompt: promptDraft } : {}),
      ...(capDraft !== null && Number.isFinite(typedCap) && typedCap > 0
        ? { maxTokens: clampCap(typedCap) }
        : {}),
    });
    // Untouched writes nothing, so another tab's pin is not overwritten.
    if (liveEffortDraft !== null) {
      const pinned = liveEffortDraft === FOLLOW_CHAT ? null : liveEffortDraft;
      setModelReasoningEffort(checkpointId, pinned);
      if (isLiveModel) {
        reconcilePinnedReasoningEffort({
          checkpoint: checkpointId,
          caps: reasoning,
          providerType,
          apiType,
        });
      }
    }
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{displayName} settings</DialogTitle>
          <DialogDescription>
            Model defaults. Chats with their own system prompt keep it.
          </DialogDescription>
        </DialogHeader>

        {rememberParamsPerModel ? null : (
          <p className="rounded-md border border-border bg-muted px-3 py-2 text-xs leading-relaxed text-muted-foreground">
            "Remember settings per model" is off in Settings → Chat, so the
            prompt and output cap below are stored but never restored. Reasoning
            effort is kept separately and still applies.
          </p>
        )}

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="connected-model-system-prompt">System prompt</Label>
            <Textarea
              id="connected-model-system-prompt"
              value={systemPrompt}
              onChange={(event) => setPromptDraft(event.target.value)}
              rows={5}
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <Label htmlFor="connected-model-max-tokens">
              Max output tokens
              <span className="ml-1 font-normal text-muted-foreground">
                ({minCap.toLocaleString()} to {maxCap.toLocaleString()})
              </span>
            </Label>
            <Input
              id="connected-model-max-tokens"
              type="number"
              min={minCap}
              max={maxCap}
              inputMode="numeric"
              value={maxTokens}
              onChange={(event) => setCapDraft(event.target.value)}
              className="w-28"
            />
          </div>

          {efforts.length > 0 ? (
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="connected-model-effort">Reasoning effort</Label>
              <Select value={effort} onValueChange={setEffortDraft}>
                <SelectTrigger id="connected-model-effort" className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={FOLLOW_CHAT}>Follow chat</SelectItem>
                  {efforts.map((level) => (
                    <SelectItem key={level} value={level}>
                      {level}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
