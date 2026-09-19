// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turns Hub metadata into the rows the picker's model-info panel renders (issue #11017).
// Pure: no React/DOM/network deps, so the panel stays a thin renderer over this and the
// decisions about what is worth showing can be tested directly.

import { formatBytes } from "@/features/hub/lib/format";
import { detectLicense } from "@/features/hub/lib/model-capabilities";

// Not imported from @/features/hub/lib/view-models: that barrel reaches as far as the auth
// pages, and ten lines of arithmetic are not worth the coupling. Same output as its
// formatParamCount for the values this panel sees.
const BILLION = 1_000_000_000;
const MILLION = 1_000_000;

function formatParamCount(totalParams: number): string {
  const billions = totalParams / BILLION;
  if (billions >= 1) {
    const rounded =
      billions >= 100
        ? Math.round(billions)
        : Number(billions.toFixed(billions >= 10 ? 0 : 1));
    return `${rounded}B`;
  }
  const millions = totalParams / MILLION;
  // Whole millions print "0M" under 500K, reporting a small embedding model as having no
  // parameters. Keep a decimal below 1M, and a plain grouped count below 100K.
  if (millions >= 1) return `${Math.round(millions)}M`;
  if (millions >= 0.1) return `${millions.toFixed(1)}M`;
  return `${Math.round(totalParams).toLocaleString()}`;
}

/** Panel reading order. The panel maps over what `modelInfoFacts` returns, so this is the order. */
export const MODEL_INFO_FIELDS = [
  "license",
  "params",
  "size",
  "library",
  "task",
  "languages",
  "created",
  "updated",
  "access",
] as const;

export type ModelInfoField = (typeof MODEL_INFO_FIELDS)[number];

export interface ModelInfoFact {
  key: ModelInfoField;
  label: string;
  value: string;
  /** Longer explanation for a tooltip, when the bare value understates the row. */
  detail?: string;
}

export interface ModelInfoMeta {
  id: string;
  license?: string | null;
  downloads?: number;
  likes?: number;
  totalParams?: number;
  sizeBytes?: number;
  /** `sizeBytes` is the full-precision checkpoint, not the size of a quantized download. */
  sizeIsFullPrecision?: boolean;
  createdAt?: string | Date;
  lastModified?: string | Date;
  library?: string;
  pipelineTag?: string;
  tags?: string[];
  languages?: string[];
  gated?: boolean;
  isPrivate?: boolean;
}

// The API returns dates as ISO strings or Date objects, and occasionally something that
// parses to NaN. "Invalid Date" in a facts panel reads as a bug, so those drop out.
function formatDate(value: string | Date | undefined): string | undefined {
  if (!value) return undefined;
  const date = value instanceof Date ? value : new Date(value);
  const ms = date.getTime();
  if (Number.isNaN(ms)) return undefined;
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/** HF pipeline tags are kebab-case ("text-generation"); the panel shows prose. */
function formatTask(tag: string): string {
  return titleCase(tag.replace(/-/g, " "));
}

function formatLanguages(languages: string[]): string {
  const named = languages.map((l) => l.toUpperCase());
  if (named.length <= 4) return named.join(", ");
  return `${named.slice(0, 4).join(", ")} +${named.length - 4}`;
}

/** Omit unavailable metadata, except for the explicit missing-licence row. */
export function modelInfoFacts(meta: ModelInfoMeta): ModelInfoFact[] {
  const facts: ModelInfoFact[] = [];

  facts.push({
    key: "license",
    label: "License",
    value: meta.license?.trim() || "Not specified",
  });

  if (meta.totalParams && meta.totalParams > 0) {
    facts.push({
      key: "params",
      label: "Parameters",
      value: formatParamCount(meta.totalParams),
    });
  }

  if (meta.sizeBytes && meta.sizeBytes > 0) {
    facts.push({
      key: "size",
      label: meta.sizeIsFullPrecision ? "Size (full precision)" : "Size",
      value: formatBytes(meta.sizeBytes),
      ...(meta.sizeIsFullPrecision
        ? {
            detail:
              "The unquantized checkpoint on the Hub. A quantized download is smaller — the size on each quant row is the one you will actually fetch.",
          }
        : {}),
    });
  }

  if (meta.library) {
    facts.push({
      key: "library",
      label: "Library",
      value: meta.library,
    });
  }

  if (meta.pipelineTag) {
    facts.push({
      key: "task",
      label: "Task",
      value: formatTask(meta.pipelineTag),
    });
  }

  if (meta.languages && meta.languages.length > 0) {
    facts.push({
      key: "languages",
      label: "Languages",
      value: formatLanguages(meta.languages),
      ...(meta.languages.length > 4
        ? { detail: meta.languages.map((l) => l.toUpperCase()).join(", ") }
        : {}),
    });
  }

  const created = formatDate(meta.createdAt);
  if (created) {
    facts.push({ key: "created", label: "Created", value: created });
  }

  const updated = formatDate(meta.lastModified);
  if (updated) {
    facts.push({ key: "updated", label: "Updated", value: updated });
  }

  // Only worth a row when it is true: every ordinary public repo would otherwise carry a
  // "Public" row that tells the user nothing.
  if (meta.gated || meta.isPrivate) {
    const labels = [
      meta.gated ? "Gated" : null,
      meta.isPrivate ? "Private" : null,
    ].filter((l): l is string => l !== null);
    facts.push({
      key: "access",
      label: "Access",
      value: labels.join(" · "),
      detail: meta.gated
        ? "Access must be requested on Hugging Face before these weights can be downloaded."
        : "This repository is private to your account.",
    });
  }

  return facts;
}

/** The subset of `HfModelResult` this panel reads. Structural, so the Hub's type can grow
 *  without dragging its module graph into the picker. */
export interface HfResultLike {
  id: string;
  downloads?: number;
  likes?: number;
  private?: boolean;
  gated?: false | "auto" | "manual";
  totalParams?: number;
  estimatedSizeBytes?: number;
  curatedSizeBytes?: number;
  tags?: string[];
  pipelineTag?: string;
  updatedAt?: string;
  createdAt?: string;
  downloadsAllTime?: number;
  libraryName?: string;
}

const LANGUAGE_TAG_PREFIX = "language:";

// HF flattens card languages into tags; recognize bare ISO codes as well as language: tags.
const ISO_639_1_CODES: ReadonlySet<string> = new Set(
  (
    "aa ab ae af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch " +
    "co cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy " +
    "ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it " +
    "iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo " +
    "lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny " +
    "oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl " +
    "sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty " +
    "ug uk ur uz ve vi vo wa wo xh yi yo za zh zu"
  ).split(" "),
);

// Accept a language code with an optional region or title-case script, not arbitrary compound tags.
const LANGUAGE_TAG_SHAPE = /^[A-Za-z]{2}(-([A-Za-z]{2}|[A-Z][a-z]{3}))?$/;

/** `en`, `zh-CN` and `pt-br` all carry a base code; the region suffix is not a language. */
function baseLanguageCode(tag: string): string {
  const [base] = tag.split("-", 1);
  return (base ?? "").toLowerCase();
}

function languagesFromTags(tags: string[] | undefined): string[] {
  const langs: string[] = [];
  // Deduplicate prefixed and bare tags without losing region or script information.
  const seen = new Set<string>();
  const add = (code: string) => {
    const key = code.toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    langs.push(code);
  };
  for (const tag of tags ?? []) {
    if (tag.startsWith(LANGUAGE_TAG_PREFIX)) {
      // The prefix states the intent, but it does not make the value a language: HF cards
      // carry `language:multilingual` and similar. Hold it to the same shape and the same set.
      const value = tag.slice(LANGUAGE_TAG_PREFIX.length);
      if (
        LANGUAGE_TAG_SHAPE.test(value) &&
        ISO_639_1_CODES.has(baseLanguageCode(value))
      )
        add(value);
      continue;
    }
    // Keep the tag's own spelling (`pt-br`, not `pt`): the region is information the reader
    // wants, even though only the base code decides whether this is a language at all.
    if (
      LANGUAGE_TAG_SHAPE.test(tag) &&
      ISO_639_1_CODES.has(baseLanguageCode(tag))
    )
      add(tag);
  }
  return langs;
}

/**
 * Adapts a Hub search/info result to this panel's input.
 *
 * Licence and language live in the `tags` array rather than in fields of their own, so both
 * are parsed out here; `gated` arrives as `false | "auto" | "manual"`, and both modes gate.
 */
export function metaFromHfResult(
  result: HfResultLike | null | undefined,
): ModelInfoMeta | null {
  if (!result) return null;
  const languages = languagesFromTags(result.tags);
  return {
    id: result.id,
    license: detectLicense(result.tags),
    // `downloads` counts a 30-day window, so the all-time figure is the fairer headline for
    // an established repo when the API reports it.
    downloads: result.downloadsAllTime ?? result.downloads,
    likes: result.likes,
    totalParams: result.totalParams,
    // Prefer the curated size; label a fallback checkpoint estimate as full precision.
    sizeBytes: result.curatedSizeBytes ?? result.estimatedSizeBytes,
    sizeIsFullPrecision:
      result.curatedSizeBytes === undefined &&
      result.estimatedSizeBytes !== undefined,
    createdAt: result.createdAt,
    lastModified: result.updatedAt,
    library: result.libraryName,
    pipelineTag: result.pipelineTag,
    tags: result.tags,
    ...(languages.length > 0 ? { languages } : {}),
    gated: result.gated !== undefined ? result.gated !== false : undefined,
    isPrivate: result.private,
  };
}
