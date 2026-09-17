// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One connected model's own settings. The system prompt and output cap are not a new store: Chat
// has remembered both per model for as long as "Remember settings per model" has been on, under
// the same checkpoint id. This just makes that memory visible and editable from the row.
//
// Reasoning effort is the one new value here; the chat's own is a single global level.

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
import {
  modelCatalogVersion,
  subscribeModelCatalog,
  useChatRuntimeStore,
} from "@/features/chat";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  getExternalReasoningCapabilities,
  resolveExternalReasoningEffort,
} from "@/features/chat/provider-capabilities";
import { useState, useSyncExternalStore } from "react";
import { useModelReasoningEffortStore } from "./model-reasoning-effort";

/** Follows the chat's own level. A Select cannot carry an empty value, so absence needs a name. */
const FOLLOW_CHAT = "__follow_chat__";

export function ConnectedModelSettingsDialog({
  open,
  onOpenChange,
  checkpointId,
  displayName,
  modelId,
  providerType,
  baseUrl,
  isReasoningProvider,
  connectionMaxOutputTokens,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The `external::` id, which is what both memories key on. */
  checkpointId: string;
  displayName: string;
  /** The provider's own id, for the catalogue lookup. */
  modelId: string;
  providerType: string;
  baseUrl?: string | null;
  /** A vLLM connection flagged as serving a reasoning model. */
  isReasoningProvider?: boolean;
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
  // The chat's own cap, which is what a model with none of its own runs at.
  const chatMaxTokens = useChatRuntimeStore((state) => state.params.maxTokens);
  // Whether this is the model currently loaded, which decides whether an edit has to reach the
  // live settings as well as the memory.
  const isLiveModel = useChatRuntimeStore(
    (state) => state.params.checkpoint === checkpointId,
  );
  const setReasoningEffort = useChatRuntimeStore(
    (state) => state.setReasoningEffort,
  );
  const chatEffort = useChatRuntimeStore((state) => state.reasoningEffort);
  const pinnedEffort = useModelReasoningEffortStore(
    (state) => state.effortByModel[checkpointId],
  );
  const setModelReasoningEffort = useModelReasoningEffortStore(
    (state) => state.setModelReasoningEffort,
  );

  // The resolver behind the composer's own Thinking control, not the catalogue alone, so the
  // levels offered here are the ones the provider actually accepts. "none" is the off switch.
  const reasoning = getExternalReasoningCapabilities(providerType, modelId, {
    isReasoningProvider,
    baseUrl,
  });
  const efforts = reasoning.supportsReasoning
    ? reasoning.reasoningEffortLevels.filter((level) => level !== "none")
    : [];

  // An OpenRouter cap comes from the live catalogue, which can land after this renders, and the
  // bounds below are read from it.
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  // The bounds the chat's own Max Tokens control uses and the adapter clamps every request to, so
  // a value stored here cannot differ from the one the provider is actually sent.
  const minCap = getExternalMinOutputTokens(providerType);
  const maxCap = getExternalMaxOutputTokens(
    providerType,
    modelId,
    connectionMaxOutputTokens,
  );
  const clampCap = (value: number) =>
    Math.min(Math.max(value, minCap), maxCap);

  // null is untouched, so an untouched field keeps reading the store. The dialog can open before
  // chat settings hydrate, and a useState seeded from the empty store then held that emptiness
  // after the response landed -- and Save wrote it back over the prompt the server had.
  const [promptDraft, setPromptDraft] = useState<string | null>(null);
  const [capDraft, setCapDraft] = useState<string | null>(null);
  const [effort, setEffort] = useState(pinnedEffort ?? FOLLOW_CHAT);
  const systemPrompt = promptDraft ?? remembered?.systemPrompt ?? "";
  // Always a real number. A blank would have to mean "forget the cap", and nothing can express
  // that: paramsByModel merges per key here and the settings row deep-merges on the server, so an
  // omitted key keeps the old value and the clear would be dropped without saying so.
  const maxTokens =
    capDraft ?? String(clampCap(remembered?.maxTokens ?? chatMaxTokens));

  function save() {
    const typedCap = Number.parseInt(maxTokens, 10);
    // Only what the user actually touched: this is a patch, and writing an untouched field would
    // put whatever the dialog happened to be showing over the stored value.
    setRememberedParamsForModel(checkpointId, {
      ...(promptDraft !== null ? { systemPrompt: promptDraft } : {}),
      // A blank or junk field leaves the cap alone rather than sending a zero as the limit.
      ...(capDraft !== null && Number.isFinite(typedCap) && typedCap > 0
        ? { maxTokens: clampCap(typedCap) }
        : {}),
    });
    setModelReasoningEffort(
      checkpointId,
      effort === FOLLOW_CHAT ? null : effort,
    );
    // The pin is read on a model switch, and a switch to the model already loaded returns before
    // that, so the live model's level has to be set here or the edit would not apply until the
    // user switched away and back. Through the same resolver the switch uses, so clearing the pin
    // falls back to the default rather than leaving the cleared level in force.
    if (isLiveModel) {
      setReasoningEffort(
        resolveExternalReasoningEffort({
          caps: reasoning,
          providerType,
          current: chatEffort,
          pinned: effort === FOLLOW_CHAT ? null : effort,
        }),
      );
    }
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{displayName} settings</DialogTitle>
          <DialogDescription>
            Used whenever you chat with this model, in place of the chat's own.
          </DialogDescription>
        </DialogHeader>

        {rememberParamsPerModel ? null : (
          <p className="rounded-md border border-border/70 bg-muted/30 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
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
              <Select value={effort} onValueChange={setEffort}>
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
