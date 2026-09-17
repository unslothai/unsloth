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
import { useChatRuntimeStore } from "@/features/chat";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { getExternalReasoningCapabilities } from "@/features/chat/provider-capabilities";
import { useState } from "react";
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

  const [systemPrompt, setSystemPrompt] = useState(
    remembered?.systemPrompt ?? "",
  );
  // A string, not a number: blank is a real state ("no cap of its own") a number cannot hold.
  const [maxTokens, setMaxTokens] = useState(
    remembered?.maxTokens ? String(remembered.maxTokens) : "",
  );
  const [effort, setEffort] = useState(pinnedEffort ?? FOLLOW_CHAT);

  function save() {
    const cap = Number.parseInt(maxTokens, 10);
    setRememberedParamsForModel(checkpointId, {
      systemPrompt,
      // Blank is an absence, not a zero the request would then send as the cap.
      ...(Number.isFinite(cap) && cap > 0 ? { maxTokens: cap } : {}),
    });
    setModelReasoningEffort(
      checkpointId,
      effort === FOLLOW_CHAT ? null : effort,
    );
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
              onChange={(event) => setSystemPrompt(event.target.value)}
              rows={5}
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <Label htmlFor="connected-model-max-tokens">
              Max output tokens
              <span className="ml-1 font-normal text-muted-foreground">
                (blank follows the connection)
              </span>
            </Label>
            <Input
              id="connected-model-max-tokens"
              type="number"
              min={1}
              inputMode="numeric"
              value={maxTokens}
              onChange={(event) => setMaxTokens(event.target.value)}
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
