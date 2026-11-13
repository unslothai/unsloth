// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hub metadata -> model-info panel rows. Pure (no React/DOM/network) so it is testable directly.

// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import { formatBytes } from "@/features/hub/lib/format";
// eslint-disable-next-line no-restricted-imports -- pure helpers; the barrel pulls React into node tests.
import { detectLicense } from "@/features/hub/lib/model-capabilities";

// Local copy of view-models' formatParamCount: that module reaches the auth pages.
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
  // Keep small models from rounding to "0M".
  if (millions >= 1) return `${Math.round(millions)}M`;
  if (millions >= 0.1) return `${millions.toFixed(1)}M`;
  return `${Math.round(totalParams).toLocaleString()}`;
}

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
  detail?: string;
}

export interface ModelInfoMeta {
  id: string;
  license?: string | null;
  downloads?: number;
  likes?: number;
  totalParams?: number;
  sizeBytes?: number;
  /** Summed from the stored tensor dtypes, so it is the checkpoint as published (FP8/BNB stay small). */
  sizeIsEstimate?: boolean;
  createdAt?: string | Date;
  lastModified?: string | Date;
  library?: string;
  pipelineTag?: string;
  tags?: string[];
  languages?: string[];
  gated?: boolean;
  isPrivate?: boolean;
}

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
      label: meta.sizeIsEstimate ? "Size (estimated)" : "Size",
      value: formatBytes(meta.sizeBytes),
      ...(meta.sizeIsEstimate
        ? {
            detail:
              "Estimated from the tensor types stored on the Hub. For GGUFs, the size on each quant row is the one you will fetch.",
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

/** Structural subset of `HfModelResult`. */
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

const LANGUAGE_TAG_SHAPE = /^[A-Za-z]{2}(-([A-Za-z]{2}|[A-Z][a-z]{3}))?$/;

function baseLanguageCode(tag: string): string {
  const [base] = tag.split("-", 1);
  return (base ?? "").toLowerCase();
}

function languagesFromTags(tags: string[] | undefined): string[] {
  const langs: string[] = [];
  const seen = new Set<string>();
  const add = (code: string) => {
    const key = code.toLowerCase();
    if (!key || seen.has(key)) return;
    seen.add(key);
    langs.push(code);
  };
  for (const tag of tags ?? []) {
    if (tag.startsWith(LANGUAGE_TAG_PREFIX)) {
      // `language:multilingual` exists, so the prefix alone is not proof.
      const value = tag.slice(LANGUAGE_TAG_PREFIX.length);
      if (
        LANGUAGE_TAG_SHAPE.test(value) &&
        ISO_639_1_CODES.has(baseLanguageCode(value))
      )
        add(value);
      continue;
    }
    if (
      LANGUAGE_TAG_SHAPE.test(tag) &&
      ISO_639_1_CODES.has(baseLanguageCode(tag))
    )
      add(tag);
  }
  return langs;
}

export function metaFromHfResult(
  result: HfResultLike | null | undefined,
): ModelInfoMeta | null {
  if (!result) return null;
  const languages = languagesFromTags(result.tags);
  return {
    id: result.id,
    license: detectLicense(result.tags),
    downloads: result.downloadsAllTime ?? result.downloads,
    likes: result.likes,
    totalParams: result.totalParams,
    sizeBytes: result.curatedSizeBytes ?? result.estimatedSizeBytes,
    sizeIsEstimate:
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
