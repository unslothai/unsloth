// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the app knows about one connected model. A local row answers "what is this" with its size,
// quant and path; a connected row has none of those, so everything comes from the provider
// registry and the connection's cached catalogue.
//
// Each figure is read through the same resolver the rest of the app uses for it, so this cannot
// disagree with the composer's own chips.
//
// What the model is, not what it is set to: the gear beside this opens the three settings it
// keeps of its own and shows what each holds, so printing them here was that dialog again.

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { modelCatalogVersion, subscribeModelCatalog } from "@/features/chat";
import type { ProviderApiType } from "@/features/chat/api/providers-api";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import { resolveModelCatalogEntry } from "@/features/chat/model-catalog";
// eslint-disable-next-line no-restricted-imports -- Avoid the chat barrel's React exports.
import {
  externalReasoningTakesEffort,
  getExternalReasoningCapabilities,
  getPublishedExternalMaxOutputTokens,
} from "@/features/chat/provider-capabilities";
import { type ReactNode, useSyncExternalStore } from "react";
import { connectedModelMarks } from "./connected-model-meta";

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

export function ConnectedModelInfoDialog({
  open,
  onOpenChange,
  modelId,
  displayName,
  providerName,
  providerType,
  apiType,
  baseUrl,
  isReasoningProvider,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The provider's own id, which is what every lookup here keys on. */
  modelId: string;
  displayName: string;
  providerName: string;
  providerType: string;
  apiType?: ProviderApiType;
  baseUrl?: string | null;
  /** A vLLM connection flagged as serving a reasoning model: the only signal a self-host gives. */
  isReasoningProvider?: boolean;
}) {
  // Every figure below is read from the catalogue, which can land after this renders.
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  const entry = resolveModelCatalogEntry(providerType, modelId);
  const marks = connectedModelMarks({
    providerType,
    modelId,
    baseUrl,
    apiType,
  });
  // The resolver behind the composer's Thinking chip, not the catalogue alone.
  const reasoning = getExternalReasoningCapabilities(providerType, modelId, {
    isReasoningProvider,
    baseUrl,
    apiType,
  });
  // "none" is the off switch, not a rung. Only where a level is sent at all: the default ladder
  // is there even for a style that carries a bare thinking on/off, and listing it read as a
  // choice this model takes.
  const takesEffort = externalReasoningTakesEffort(reasoning);
  const effortLevels = takesEffort
    ? reasoning.reasoningEffortLevels.filter((level) => level !== "none")
    : [];
  const publishedMaxOutput =
    getPublishedExternalMaxOutputTokens(providerType, modelId) ??
    entry?.maxOutputTokens ??
    null;

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
      </DialogContent>
    </Dialog>
  );
}
