// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the app knows about one connected model. A local row answers "what is this" with its size,
// quant and path; a connected row has none of those, so everything comes from the provider
// registry and the connection's cached catalogue.
//
// Each figure is read through the same resolver the rest of the app uses for it, so this cannot
// disagree with the composer's own chips.

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useChatRuntimeStore } from "@/features/chat";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { resolveModelCatalogEntry } from "@/features/chat/model-catalog";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  getExternalReasoningCapabilities,
  getPublishedExternalMaxOutputTokens,
} from "@/features/chat/provider-capabilities";
import type { ReactNode } from "react";
import { connectedModelMarks } from "./connected-model-meta";
import { useModelReasoningEffortStore } from "./model-reasoning-effort";

const MODALITY_LABELS: Record<string, string> = {
  text: "Text",
  image: "Images",
  audio: "Audio",
  video: "Video",
  pdf: "PDF",
};

function tokens(count: number): string {
  return `${count.toLocaleString()} tokens`;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 py-2">
      <span className="shrink-0 text-xs text-muted-foreground">{label}</span>
      <span className="min-w-0 text-right text-xs text-foreground">
        {children}
      </span>
    </div>
  );
}

function Unset({ children }: { children: string }) {
  return <span className="text-muted-foreground">{children}</span>;
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="pt-4 pb-1 text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

export function ConnectedModelInfoDialog({
  open,
  onOpenChange,
  modelId,
  checkpointId,
  displayName,
  providerName,
  providerType,
  baseUrl,
  isReasoningProvider,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The provider's own id, which is what every lookup here keys on. */
  modelId: string;
  /** The `external::` id, which is what the per-model memories key on instead. */
  checkpointId: string;
  displayName: string;
  providerName: string;
  providerType: string;
  baseUrl?: string | null;
  /** A vLLM connection flagged as serving a reasoning model: the only signal a self-host gives. */
  isReasoningProvider?: boolean;
}) {
  const entry = resolveModelCatalogEntry(providerType, modelId);
  const marks = connectedModelMarks({ providerType, modelId, baseUrl });
  // The resolver behind the composer's Thinking chip, not the catalogue alone.
  const reasoning = getExternalReasoningCapabilities(providerType, modelId, {
    isReasoningProvider,
    baseUrl,
  });
  // "none" is the off switch, not a rung.
  const effortLevels = reasoning.reasoningEffortLevels.filter(
    (level) => level !== "none",
  );
  const publishedMaxOutput =
    getPublishedExternalMaxOutputTokens(providerType, modelId) ??
    entry?.maxOutputTokens ??
    null;

  // What this model carries of its own. Chat has remembered these per model all along, but
  // nothing ever said so, which is how a prompt set weeks ago comes back as a surprise.
  const remembered = useChatRuntimeStore(
    (state) => state.paramsByModel[checkpointId],
  );
  const pinnedEffort = useModelReasoningEffortStore(
    (state) => state.effortByModel[checkpointId],
  );
  const ownPrompt = remembered?.systemPrompt?.trim();

  const inputs = entry?.inputModalities?.length
    ? entry.inputModalities
        .map((modality) => MODALITY_LABELS[modality] ?? modality)
        .join(" · ")
    : marks.vision
      ? "Text · Images"
      : "Text";
  const generates = [
    marks.capabilities.imageGen ? "Images" : null,
    marks.capabilities.videoGen ? "Video" : null,
  ].filter(Boolean);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{displayName}</DialogTitle>
        </DialogHeader>

        <div className="-mt-2 divide-y divide-border/50">
          <Field label="Model ID">
            {/* break-all, not break-words: an id has no spaces to break at. */}
            <span className="break-all font-mono">{modelId}</span>
          </Field>
          <Field label="Connection">{providerName}</Field>
          {baseUrl ? (
            <Field label="Endpoint">
              <span className="break-all font-mono">{baseUrl}</span>
            </Field>
          ) : null}
          <Field label="Accepts">{inputs}</Field>
          {generates.length > 0 ? (
            <Field label="Generates">{generates.join(" · ")}</Field>
          ) : null}
          <Field label="Context window">
            {entry?.contextLength ? (
              <span className="tabular-nums">
                {tokens(entry.contextLength)}
              </span>
            ) : (
              <Unset>Not published</Unset>
            )}
          </Field>
          <Field label="Max output">
            {publishedMaxOutput ? (
              <span className="tabular-nums">{tokens(publishedMaxOutput)}</span>
            ) : (
              <Unset>Follows the connection</Unset>
            )}
          </Field>
          <Field label="Reasoning">
            {reasoning.supportsReasoning ? (
              <>
                {effortLevels.length > 0
                  ? effortLevels.join(" · ")
                  : "Supported"}
                {reasoning.reasoningAlwaysOn ? " (always on)" : null}
              </>
            ) : (
              <Unset>No</Unset>
            )}
          </Field>
        </div>

        <div className="divide-y divide-border/50">
          <SectionLabel>This model's own settings</SectionLabel>
          <Field label="System prompt">
            {ownPrompt ? (
              <span className="line-clamp-3 text-left">{ownPrompt}</span>
            ) : (
              <Unset>None</Unset>
            )}
          </Field>
          <Field label="Max output">
            {remembered?.maxTokens ? (
              <span className="tabular-nums">
                {tokens(remembered.maxTokens)}
              </span>
            ) : (
              <Unset>Follows the chat</Unset>
            )}
          </Field>
          <Field label="Reasoning effort">
            {pinnedEffort ?? <Unset>Follows the chat</Unset>}
          </Field>
        </div>
      </DialogContent>
    </Dialog>
  );
}
