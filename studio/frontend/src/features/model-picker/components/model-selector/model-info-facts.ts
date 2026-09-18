// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turns Hub metadata into the rows the picker's model-info panel renders (issue #11017).
// Pure: no React/DOM/network deps, so the panel stays a thin renderer over this and the
// decisions about what is worth showing can be tested directly.

import { formatBytes } from "@/features/hub/lib/format";
import { detectLicense } from "@/features/hub/lib/model-capabilities";
import { type LicenseOpenness, classifyLicense } from "./license-openness";

// Deliberately not imported from @/features/hub/lib/view-models: that barrel re-exports the
// Hub inventory and reaches as far as the auth pages, and pulling it in for ten lines of
// arithmetic would tie the picker's info panel to all of it. Same output as its
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
  // Rounding straight to whole millions prints "0M" for anything under 500K — a tokenizer or a
  // small embedding model reported as having no parameters. Below 1M, keep a decimal, and below
  // 100K fall back to a plain grouped count, which is the only honest rendering left.
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
  /** Set on the licence row only, so the panel can tone the chip by verdict. */
  openness?: LicenseOpenness;
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

/**
 * The rows worth showing for `meta`, in `MODEL_INFO_FIELDS` order.
 *
 * A field whose data HF did not return is omitted rather than rendered as a placeholder:
 * the panel exists to state facts about a model before the user loads it, and an invented
 * row is worse than a shorter panel. The licence is the deliberate exception — "not stated"
 * is itself the answer when someone is checking whether a model is open source.
 */
export function modelInfoFacts(meta: ModelInfoMeta): ModelInfoFact[] {
  const facts: ModelInfoFact[] = [];

  // Always present: silence here would read as "no restrictions", which is the opposite
  // of what an unlicensed repo means.
  const license = classifyLicense(meta.license);
  facts.push({
    key: "license",
    label: "License",
    value: license.label,
    openness: license.openness,
    detail: license.summary,
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

// HF emits a model's languages both ways: the dataset-style `language:en`, and — far more
// commonly for models, since a card's `language:` frontmatter list is flattened into `tags` —
// the bare code `en`. Reading only the prefixed form drops the languages of most model repos.
//
// A bare code cannot be recognised by shape: a length test would sweep in `rl`, `ai` and any
// two-letter library name alongside the real codes. So bare codes are matched against this
// explicit ISO 639-1 set; anything outside it stays a plain tag. Prefixed codes need no such
// guard, as the prefix already states the intent.
//
// The set cannot resolve every collision: a bare `ml` is Malayalam's code and also how a repo
// might tag "machine learning". It reads as the language, since that is what the tag means in
// HF's language vocabulary, and the cost of being wrong is one stray chip against losing the
// Languages row for every repo that tags bare codes.
const ISO_639_1_CODES: ReadonlySet<string> = new Set([
  "aa",
  "ab",
  "ae",
  "af",
  "ak",
  "am",
  "an",
  "ar",
  "as",
  "av",
  "ay",
  "az",
  "ba",
  "be",
  "bg",
  "bi",
  "bm",
  "bn",
  "bo",
  "br",
  "bs",
  "ca",
  "ce",
  "ch",
  "co",
  "cr",
  "cs",
  "cu",
  "cv",
  "cy",
  "da",
  "de",
  "dv",
  "dz",
  "ee",
  "el",
  "en",
  "eo",
  "es",
  "et",
  "eu",
  "fa",
  "ff",
  "fi",
  "fj",
  "fo",
  "fr",
  "fy",
  "ga",
  "gd",
  "gl",
  "gn",
  "gu",
  "gv",
  "ha",
  "he",
  "hi",
  "ho",
  "hr",
  "ht",
  "hu",
  "hy",
  "hz",
  "ia",
  "id",
  "ie",
  "ig",
  "ii",
  "ik",
  "io",
  "is",
  "it",
  "iu",
  "ja",
  "jv",
  "ka",
  "kg",
  "ki",
  "kj",
  "kk",
  "kl",
  "km",
  "kn",
  "ko",
  "kr",
  "ks",
  "ku",
  "kv",
  "kw",
  "ky",
  "la",
  "lb",
  "lg",
  "li",
  "ln",
  "lo",
  "lt",
  "lu",
  "lv",
  "mg",
  "mh",
  "mi",
  "mk",
  "ml",
  "mn",
  "mr",
  "ms",
  "mt",
  "my",
  "na",
  "nb",
  "nd",
  "ne",
  "ng",
  "nl",
  "nn",
  "no",
  "nr",
  "nv",
  "ny",
  "oc",
  "oj",
  "om",
  "or",
  "os",
  "pa",
  "pi",
  "pl",
  "ps",
  "pt",
  "qu",
  "rm",
  "rn",
  "ro",
  "ru",
  "rw",
  "sa",
  "sc",
  "sd",
  "se",
  "sg",
  "si",
  "sk",
  "sl",
  "sm",
  "sn",
  "so",
  "sq",
  "sr",
  "ss",
  "st",
  "su",
  "sv",
  "sw",
  "ta",
  "te",
  "tg",
  "th",
  "ti",
  "tk",
  "tl",
  "tn",
  "to",
  "tr",
  "ts",
  "tt",
  "tw",
  "ty",
  "ug",
  "uk",
  "ur",
  "uz",
  "ve",
  "vi",
  "vo",
  "wa",
  "wo",
  "xh",
  "yi",
  "yo",
  "za",
  "zh",
  "zu",
]);

// A language tag is a 2-letter ISO 639-1 code, optionally followed by ONE region or script
// subtag: `en`, `zh-CN`, `pt-br`, `sr-Latn`. Testing only the segment before the first hyphen is
// not enough — that accepts the WHOLE of any tag merely beginning with two letters that happen
// to spell a code, and the Hub carries those: `ml-agents` (the Unity toolkit, thousands of
// repos) read as Malayalam, and `mt-bench` as Maltese. The tag has to match end to end for the
// ISO set to mean anything.
//
// The region alternative is two letters in either case, since HF cards write both `zh-CN` and
// `pt-br`. The script alternative is four letters in BCP-47's title case (`Latn`, `Hans`), which
// is what keeps an English compound like `no-code` or `or-else` out: lowercase four-letter
// suffixes are not script codes.
//
// One residual, stated rather than papered over: a two-letter suffix cannot be told from a real
// region, so `as-is` (Assamese + Iceland) and `to-do` (Tongan + Dominican Republic) are
// structurally valid language tags and would still be accepted. Neither exists on the Hub — I
// checked — and no structural rule can reject them, since they are well-formed BCP-47.
const LANGUAGE_TAG_SHAPE = /^[A-Za-z]{2}(-([A-Za-z]{2}|[A-Z][a-z]{3}))?$/;

/** `en`, `zh-CN` and `pt-br` all carry a base code; the region suffix is not a language. */
function baseLanguageCode(tag: string): string {
  const [base] = tag.split("-", 1);
  return (base ?? "").toLowerCase();
}

function languagesFromTags(tags: string[] | undefined): string[] {
  const langs: string[] = [];
  // Dedup on the lowercased spelling, because HF emits a model's languages both ways and the
  // two spellings differ in case as often as not: `["language:EN", "en"]` is one language, and
  // listing it as "EN, EN" is the panel contradicting itself in a single row. The first
  // spelling seen wins, so a repo that only says `zh-CN` still keeps its region.
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
      if (LANGUAGE_TAG_SHAPE.test(value) && ISO_639_1_CODES.has(baseLanguageCode(value)))
        add(value);
      continue;
    }
    // Keep the tag's own spelling (`pt-br`, not `pt`): the region is information the reader
    // wants, even though only the base code decides whether this is a language at all.
    if (LANGUAGE_TAG_SHAPE.test(tag) && ISO_639_1_CODES.has(baseLanguageCode(tag)))
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
    // `curatedSizeBytes` first, matching `recommended-fit.ts`, whose comment states the rule and
    // the reason: "Safetensors / MLX always use the params-based estimate ... since their
    // estimatedSizeBytes is the full-precision checkpoint. `curatedSizeBytes` outranks both."
    // Reading it the other way round made this panel quote the BF16 checkpoint — often four
    // times the download — while the size badge on the very same row quoted the quantized load.
    //
    // When only the estimate is available the row still shows it, but `sizeIsFullPrecision`
    // marks where it came from so the panel can say so rather than let the reader take a
    // checkpoint size for a download size. No params-based guess is substituted: the one
    // constant the codebase has, `MIN_QUANT_BYTES_PER_PARAM`, is the SMALLEST practical quant
    // for a can-it-run check, and printing that as a download size would be a new wrong number
    // in the opposite direction.
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
