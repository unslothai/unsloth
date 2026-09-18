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
  // Offered only where a level is actually sent: the default low/medium/high ladder is present
  // even for a model whose style carries a bare thinking on/off, so gating on supportsReasoning
  // alone let a pin be set on Kimi that no request could ever carry.
  const efforts = externalReasoningTakesEffort(reasoning)
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

  // null is untouched, so an untouched field keeps reading the store and Save writes nothing for
  // it. The dialog can open before chat settings hydrate, and a useState seeded from the empty
  // store then held that emptiness after the response landed, so Save put it back over the prompt
  // the server had. A pin another tab changes while this is open reads the same way.
  const [promptDraft, setPromptDraft] = useState<string | null>(null);
  const [capDraft, setCapDraft] = useState<string | null>(null);
  const [effortDraft, setEffortDraft] = useState<string | null>(null);
  const systemPrompt = promptDraft ?? remembered?.systemPrompt ?? "";
  // Against the ladder as it stands now, since the catalogue subscription below can withdraw a
  // level while this is open. A draft the model no longer offers reverts to what is stored and
  // Save leaves the pin alone, rather than writing a level every resolver then refuses. A stored
  // pin the ladder dropped reads as Follow chat for the same reason: that is what it now does.
  const offered = (level: string): boolean =>
    (efforts as readonly string[]).includes(level);
  const liveEffortDraft =
    effortDraft !== null && (effortDraft === FOLLOW_CHAT || offered(effortDraft))
      ? effortDraft
      : null;
  const effort =
    liveEffortDraft ??
    (pinnedEffort && offered(pinnedEffort) ? pinnedEffort : FOLLOW_CHAT);
  // Always a real number. A blank would have to mean "forget the cap", and nothing can express
  // that: paramsByModel merges per key here and the settings row deep-merges on the server, so an
  // omitted key keeps the old value and the clear would be dropped without saying so.
  const maxTokens =
    capDraft ?? String(clampCap(remembered?.maxTokens ?? chatMaxTokens));

  function save() {
    // Number, not parseInt: a number field accepts scientific notation, and parseInt stops at the
    // "e", so 1e5 read as 1 and was saved clamped to the provider minimum. Rounded because the
    // cap is a token count and the field admits a decimal.
    const typedCap = Math.round(Number(maxTokens.trim()));
    // Only what the user actually touched: this is a patch, and writing an untouched field would
    // put whatever the dialog happened to be showing over the stored value.
    setRememberedParamsForModel(checkpointId, {
      ...(promptDraft !== null ? { systemPrompt: promptDraft } : {}),
      // A blank or junk field leaves the cap alone rather than sending a zero as the limit.
      ...(capDraft !== null && Number.isFinite(typedCap) && typedCap > 0
        ? { maxTokens: clampCap(typedCap) }
        : {}),
    });
    // Untouched writes nothing at all: another tab can change this pin while the dialog is open,
    // and a save of an unrelated field would otherwise put the value this select opened with back
    // over it. It would also reset the composer's own Think level for no reason.
    if (liveEffortDraft !== null) {
      const pinned = liveEffortDraft === FOLLOW_CHAT ? null : liveEffortDraft;
      setModelReasoningEffort(checkpointId, pinned);
      if (isLiveModel) {
        reconcilePinnedReasoningEffort({
          checkpoint: checkpointId,
          caps: reasoning,
          providerType,
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
