// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { formatBytes, useHubModelSearch } from "@/features/hub";
import { useDebouncedValue, useWheelScrollRef } from "@/hooks";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { PipelineType } from "@huggingface/hub";
import { type ReactElement, useMemo, useState } from "react";

// HF pipeline filter for embedding models; matches the backend's
// is_embedding_model signals (sentence-similarity / feature-extraction).
const EMBEDDING_TASKS: readonly PipelineType[] = [
  "sentence-similarity",
  "feature-extraction",
];
type EmbeddingModelPickerProps = {
  value: string;
  /** Fires once, on a pick. Typing is a search, not a selection. */
  onSelect: (model: string) => void;
  /** The env/default model, marked "Recommended" so it is findable again. */
  defaultModel?: string;
  /** Repos already on disk, for the on-device dot. */
  cachedModels?: ReadonlySet<string>;
  accessToken?: string;
  disabled?: boolean;
  /** Held open with a spinner while the pick is resolved and saved. */
  busy?: boolean;
  className?: string;
};

/**
 * Embedding model picker for Settings -> Documents & RAG.
 *
 * Modelled on the dictation picker (voice-tab.tsx `SttModelPicker`): a button
 * trigger, so the saved model never reads as a half-typed query, and one field
 * that reaches the whole Hub. Empty lists unsloth's embedders, which the global
 * top-downloads page would bury.
 */
/** Repo ids an on-device copy of `model` can be filed under.
 *
 * The inventory records what was fetched, not what was picked, and the two differ
 * by the conventions the backend resolves through: llama-server opens the `-GGUF`
 * companion, and a slashless alias resolves under `sentence-transformers/`.
 *
 * An off-convention mirror is not derivable here and still shows no dot; the
 * row's own status line covers the selected model. */
export function cachedRepoCandidates(model: string): string[] {
  const id = model.trim();
  if (!id) return [];
  const candidates = [id, `${id}-GGUF`];
  if (!id.includes("/")) candidates.push(`sentence-transformers/${id}`);
  // Mirrors the backend's _QUANT_SUFFIX_RE: an unquantized re-upload's GGUF sits
  // on the base name, so embeddinggemma-300m-qat-q8_0-unquantized resolves to
  // embeddinggemma-300m-GGUF and would otherwise never light up.
  const slash = id.lastIndexOf("/");
  const owner = slash === -1 ? "" : id.slice(0, slash + 1);
  const name = id.slice(slash + 1);
  const base = name.replace(/(?:-qat)?(?:-q\d+_\d+[a-z]*)?-unquantized$/i, "");
  if (base !== name) candidates.push(`${owner}${base}-GGUF`);
  return candidates;
}

function isOnDevice(
  cached: ReadonlySet<string> | undefined,
  model: string,
): boolean {
  if (!cached) return false;
  return cachedRepoCandidates(model).some((repo) => cached.has(repo));
}

export function EmbeddingModelPicker({
  value,
  onSelect,
  defaultModel,
  cachedModels,
  accessToken,
  disabled,
  busy,
  className,
}: EmbeddingModelPickerProps): ReactElement {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const resultsRef = useWheelScrollRef<HTMLDivElement>();
  const debouncedQuery = useDebouncedValue(query.trim());

  const { results, isLoading } = useHubModelSearch(debouncedQuery, {
    task: EMBEDDING_TASKS,
    accessToken,
    excludeGguf: true,
    enabled: open && !disabled,
    ownerScope: debouncedQuery ? "all" : "unsloth",
  });

  const items = useMemo(() => {
    const rows = results.map((result) => ({
      id: result.id,
      sizeBytes: result.estimatedSizeBytes ?? result.curatedSizeBytes ?? null,
    }));
    // Keep the saved model reachable when the listing drops it: a local path, or
    // a repo the query does not match.
    const selected = value.trim();
    if (selected && !rows.some((row) => row.id === selected)) {
      rows.push({ id: selected, sizeBytes: null });
    }
    // The configured default the same way: the empty search is scoped to
    // `unsloth`, so a private, other-owner or local default had no row, and the
    // old "Reset to default" button is gone.
    const fallback = defaultModel?.trim();
    if (fallback && !rows.some((row) => row.id === fallback)) {
      rows.push({ id: fallback, sizeBytes: null });
    }
    return rows;
  }, [results, value, defaultModel]);

  const pick = (model: string) => {
    setOpen(false);
    setQuery("");
    // Reselecting is a retry: a previous transfer may have been cancelled or
    // its cache evicted while the setting still names this model.
    onSelect(model);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (disabled || busy) return;
        setOpen(next);
        if (!next) setQuery("");
      }}
    >
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          data-testid="embedding-model-trigger"
          aria-label={t("settings.general.rag.embeddingModel")}
          disabled={disabled || busy}
          className={`border-border bg-background hover:bg-accent/50 dark:border-transparent dark:bg-white/[0.06] dark:hover:bg-white/10 focus-visible:border-ring flex h-8 w-full cursor-pointer items-center justify-between gap-1.5 rounded-full border px-3.5 font-mono text-ui-11 outline-none transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${className ?? ""}`}
        >
          <span className="truncate">{value}</span>
          {busy ? (
            <Spinner className="size-3.5 shrink-0" />
          ) : (
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={2}
              className="text-muted-foreground pointer-events-none size-4 shrink-0"
            />
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" sideOffset={4} className="w-80 gap-0 p-0">
        <div className="relative p-1.5 pb-0.5">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={2}
            className="text-muted-foreground pointer-events-none absolute top-[calc(50%+2px)] left-4 size-3.5 -translate-y-1/2"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            data-testid="embedding-model-search"
            placeholder={t("settings.general.rag.searchPlaceholder")}
            className="h-8 pl-8 text-sm"
            autoFocus={true}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              event.preventDefault();
              // Preserve arbitrary typed submission: the backend can recognize
              // existing relative paths such as "embedder" that have no slash.
              const typed = query.trim();
              if (typed) {
                pick(typed);
              } else if (items.length > 0) {
                pick(items[0].id);
              }
            }}
          />
        </div>
        <div
          ref={resultsRef}
          data-testid="embedding-model-results"
          className="max-h-64 overflow-y-auto p-1"
        >
          {isLoading && debouncedQuery ? (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              {t("settings.general.rag.searching")}
            </div>
          ) : items.length === 0 ? (
            <div className="px-3 py-3 text-xs text-muted-foreground">
              {t("settings.general.rag.noResults")}
            </div>
          ) : (
            items.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => pick(item.id)}
                aria-selected={item.id === value}
                className={`flex w-full items-center justify-between gap-3 rounded-full px-2.5 py-1.5 text-left transition-colors hover:bg-muted ${
                  item.id === value ? "bg-accent font-medium" : ""
                }`}
              >
                <span className="flex min-w-0 flex-1 items-center gap-1.5">
                  {/* Same green dot the Hub marks an on-device row with. */}
                  {isOnDevice(cachedModels, item.id) ? (
                    <span
                      // A bare span is generic, and ARIA-in-HTML forbids naming
                      // one, so Safari and Firefox drop the label and the dot goes
                      // unannounced. Same role the Hub's own on-device dot carries.
                      role="img"
                      aria-label={t("settings.general.rag.onDevice")}
                      className="size-[5px] shrink-0 rounded-full bg-status-success"
                    />
                  ) : null}
                  <span className="truncate font-mono text-ui-11">
                    {item.id}
                  </span>
                  {item.id === defaultModel ? (
                    <span className="shrink-0 rounded-full bg-emerald-500/12 px-1.5 py-px text-ui-9 font-medium text-emerald-600 dark:bg-emerald-400/15 dark:text-emerald-400">
                      {t("settings.general.rag.recommended")}
                    </span>
                  ) : null}
                </span>
                {item.sizeBytes ? (
                  <span className="shrink-0 text-ui-10 tabular-nums text-muted-foreground">
                    {formatBytes(item.sizeBytes)}
                  </span>
                ) : null}
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
