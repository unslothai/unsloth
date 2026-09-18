// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Classifies a Hugging Face `license` slug into how freely the weights may be used, so the
// picker's info panel can answer "is this open source?" instead of echoing a bare slug
// (issue #11017). Pure: no React/DOM/network deps, so the picker and the Hub can both call it.

/**
 * - `open`        OSI-style terms: use, modify and redistribute, commercial use included.
 * - `restricted`  Public weights carrying conditions an OSI licence would not impose:
 *   acceptable-use policies, user-count ceilings, non-commercial or research-only grants.
 *   Usable, but the condition is what a user needs to see before shipping.
 * - `proprietary` Terms that do not grant redistribution at all.
 * - `unknown`     Nothing stated, or a slug we do not recognise. Absence of a licence is not
 *   permission, so this never collapses into `open`.
 */
export type LicenseOpenness = "open" | "restricted" | "proprietary" | "unknown";

export interface LicenseVerdict {
  openness: LicenseOpenness;
  /** Display label for the licence itself, e.g. "Apache 2.0"; the original text when unrecognised. */
  label: string;
  /** One short sentence for a tooltip: what this licence lets the user do. Never empty. */
  summary: string;
}

// `obligation` is the condition the grant carries, not a catch in the RESTRICTED sense: these
// licences do permit commercial use. Omitting it renders copyleft and attribution as absent to
// the reader about to ship. AGPL is the expensive case — §13 obliges anyone offering network
// access to offer the source, so "permits commercial use" alone misleads a hosted-inference
// user.
const OPEN_LICENSES: Readonly<
  Record<string, { label: string; obligation?: string }>
> = {
  "apache-2.0": { label: "Apache 2.0" },
  mit: { label: "MIT" },
  bsd: { label: "BSD" },
  "bsd-2-clause": { label: "BSD 2-Clause" },
  "bsd-3-clause": { label: "BSD 3-Clause" },
  "bsd-3-clause-clear": { label: "BSD 3-Clause Clear" },
  "cc0-1.0": { label: "CC0 1.0" },
  "cc-by-4.0": { label: "CC BY 4.0", obligation: "you credit the author" },
  "cc-by-3.0": { label: "CC BY 3.0", obligation: "you credit the author" },
  "cc-by-2.0": { label: "CC BY 2.0", obligation: "you credit the author" },
  "cc-by-sa-4.0": {
    label: "CC BY-SA 4.0",
    obligation: "you credit the author and derivatives carry the same licence",
  },
  "cc-by-sa-3.0": {
    label: "CC BY-SA 3.0",
    obligation: "you credit the author and derivatives carry the same licence",
  },
  "artistic-2.0": { label: "Artistic 2.0" },
  isc: { label: "ISC" },
  zlib: { label: "zlib" },
  wtfpl: { label: "WTFPL" },
  "bsl-1.0": { label: "Boost 1.0" },
  "afl-3.0": { label: "Academic Free 3.0" },
  "ms-pl": { label: "Microsoft Public" },
  ncsa: { label: "NCSA" },
  postgresql: { label: "PostgreSQL" },
  "ecl-2.0": { label: "Educational Community 2.0" },
  "odc-by": { label: "ODC-By", obligation: "you credit the source" },
  "cdla-permissive-2.0": { label: "CDLA-Permissive 2.0" },
  "mpl-2.0": {
    label: "MPL 2.0",
    obligation: "modified files stay under the MPL",
  },
  "epl-2.0": {
    label: "EPL 2.0",
    obligation: "modified source stays under the EPL",
  },
  "epl-1.0": {
    label: "EPL 1.0",
    obligation: "modified source stays under the EPL",
  },
  "osl-3.0": {
    label: "OSL 3.0",
    obligation: "derivatives stay under the OSL",
  },
  "eupl-1.2": {
    label: "EUPL 1.2",
    obligation: "derivatives stay under the EUPL",
  },
  "lgpl-3.0": {
    label: "LGPL 3.0",
    obligation: "changes to the library itself stay under the LGPL",
  },
  "lgpl-2.1": {
    label: "LGPL 2.1",
    obligation: "changes to the library itself stay under the LGPL",
  },
  lgpl: {
    label: "LGPL",
    obligation: "changes to the library itself stay under the LGPL",
  },
  "gpl-3.0": {
    label: "GPL 3.0",
    obligation: "derivatives stay under the GPL",
  },
  "gpl-2.0": {
    label: "GPL 2.0",
    obligation: "derivatives stay under the GPL",
  },
  gpl: { label: "GPL", obligation: "derivatives stay under the GPL" },
  "agpl-3.0": {
    label: "AGPL 3.0",
    obligation:
      "derivatives stay under the AGPL and anyone using it over a network is offered the source",
  },
  // Permissive despite the name: the Apple SAMPLE CODE licence grants use, modification and
  // redistribution, conditioned only on keeping the notice and not implying endorsement. The
  // research-only sibling is `apple-amlr`, in RESTRICTED_LICENSES.
  "apple-ascl": {
    label: "Apple Sample Code",
    obligation: "you keep Apple's notice and do not imply Apple's endorsement",
  },
  unlicense: { label: "Unlicense" },
};

// Weights anyone may download, under terms an OSI licence would not impose. Each entry says
// what the catch is, because "restricted" alone tells a user nothing actionable.
const RESTRICTED_LICENSES: Readonly<
  Record<string, { label: string; catch: string; note?: string }>
> = {
  llama2: {
    label: "Llama 2 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  llama3: {
    label: "Llama 3 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  "llama3.1": {
    label: "Llama 3.1 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  // 3.2 and 4 carry a clause the other Llama releases do not, and it is not a condition on the
  // grant — it withholds it. Meta's Acceptable Use Policy for both: "With respect to any
  // multimodal models included in Llama <v>, the rights granted under Section 1(a) ... are not
  // being granted to you if you are an individual domiciled in, or a company with a principal
  // place of business in, the European Union." (llama-models/models/llama4/USE_POLICY.md, and
  // the identically worded llama3_2/USE_POLICY.md.) Every Llama 4 model is multimodal, so for
  // `llama4` it applies to the whole tag; for `llama3.2` it reaches the Vision models only, and
  // a slug cannot tell those apart, so the wording says "vision models".
  "llama3.2": {
    label: "Llama 3.2 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
    note: "Its vision models are also not licensed at all to individuals domiciled in, or companies with a principal place of business in, the European Union.",
  },
  "llama3.3": {
    label: "Llama 3.3 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  llama4: {
    label: "Llama 4 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
    note: "Every Llama 4 model is multimodal, so none of them is licensed at all to individuals domiciled in, or companies with a principal place of business in, the European Union.",
  },
  gemma: { label: "Gemma Terms of Use", catch: "a prohibited-use policy" },
  "apple-amlr": { label: "Apple ML Research", catch: "research-only terms" },
  "fair-noncommercial-research-license": {
    label: "FAIR Non-Commercial Research",
    catch: "a non-commercial research-only grant",
  },
  "deepfloyd-if-license": {
    label: "DeepFloyd IF Research",
    catch: "a non-commercial research-only grant",
  },
  "intel-research": {
    label: "Intel Research",
    catch: "a research-only grant",
  },
  "h-research": { label: "H Research", catch: "a research-only grant" },
  "creativeml-openrail-m": {
    label: "CreativeML OpenRAIL-M",
    catch: "use-based restrictions",
  },
  "openrail++": { label: "OpenRAIL++", catch: "use-based restrictions" },
  openrail: { label: "OpenRAIL", catch: "use-based restrictions" },
  "bigscience-bloom-rail-1.0": {
    label: "BigScience BLOOM RAIL 1.0",
    catch: "use-based restrictions",
  },
  "bigcode-openrail-m": {
    label: "BigCode OpenRAIL-M",
    catch: "use-based restrictions",
  },
  "bigscience-openrail-m": {
    label: "BigScience OpenRAIL-M",
    catch: "use-based restrictions",
  },
  // Every non-commercial spelling HF actually emits, not just the 4.0 ones. A missing NC slug
  // degrades to "Licence unclear", which drops the warning for exactly the licences where
  // getting it wrong costs the most, so the older point releases are listed too.
  "cc-by-nc-4.0": { label: "CC BY-NC 4.0", catch: "a non-commercial grant" },
  "cc-by-nc-3.0": { label: "CC BY-NC 3.0", catch: "a non-commercial grant" },
  "cc-by-nc-2.0": { label: "CC BY-NC 2.0", catch: "a non-commercial grant" },
  "cc-by-nc-sa-4.0": {
    label: "CC BY-NC-SA 4.0",
    catch: "a non-commercial share-alike grant",
  },
  "cc-by-nc-sa-3.0": {
    label: "CC BY-NC-SA 3.0",
    catch: "a non-commercial share-alike grant",
  },
  "cc-by-nc-sa-2.0": {
    label: "CC BY-NC-SA 2.0",
    catch: "a non-commercial share-alike grant",
  },
  "cc-by-nc-nd-4.0": {
    label: "CC BY-NC-ND 4.0",
    catch: "a non-commercial, no-derivatives grant",
  },
  "cc-by-nc-nd-3.0": {
    label: "CC BY-NC-ND 3.0",
    catch: "a non-commercial, no-derivatives grant",
  },
  "cc-by-nd-4.0": { label: "CC BY-ND 4.0", catch: "a no-derivatives grant" },
};

// `proprietary` is NOT in Hugging Face's licence vocabulary — no Hub repo carries it, so this
// branch cannot fire for a Hub-sourced model, and the red chip it drives is unreachable today.
// It is kept, not deleted, because the classifier takes a bare slug and a non-Hub catalog may
// legitimately supply one. What it must not become is a dumping ground: a vendor EULA arrives
// tagged `other`, which is deliberately routed to `unknown` below rather than here, since the
// repository never said redistribution was refused.
const PROPRIETARY_LICENSES: Readonly<Record<string, string>> = {
  proprietary: "Proprietary",
};

// HF tags that name the absence of a verdict rather than a set of terms. Neither establishes
// that redistribution is refused: `unknown` states nothing at all, and `other` is the catch-all
// for terms that did not fit HF's list — sometimes a vendor EULA, sometimes a custom licence
// that grants everything Apache does. Calling either "proprietary" would assert a restriction
// the repository never stated, which is the same invention this module avoids for slugs it does
// not recognise, so they resolve to `unknown` and send the reader to the model card.
const UNCLEAR_LICENSES: Readonly<
  Record<string, { label: string; summary: string }>
> = {
  unknown: {
    label: "Not stated",
    summary:
      "This repository tags its licence as unknown. No usage rights are granted by default — check the model card before use.",
  },
  other: {
    label: "Custom terms",
    summary:
      "This repository uses custom licence terms that Hugging Face does not classify. Read the model card before use.",
  },
  // HF's generic Creative Commons family tag. It names a family that spans CC0 through
  // CC BY-NC-ND, so it can be anything from public domain to non-commercial and
  // no-derivatives. Unclear is the honest verdict, but say WHY rather than reporting it as a
  // slug nobody recognised.
  cc: {
    label: "Creative Commons (unspecified)",
    summary:
      "Tagged only as Creative Commons, which spans everything from CC0 to non-commercial, no-derivatives terms. Check the model card for which one applies.",
  },
};

// These tables are plain objects, so a bare index also resolves Object.prototype: a repo
// tagged `license:__proto__` or `license:constructor` read as open, with a non-string label
// that is not a valid React child. Own keys only.
function own<T>(table: Readonly<Record<string, T>>, slug: string): T | undefined {
  return Object.hasOwn(table, slug) ? table[slug] : undefined;
}

export function classifyLicense(
  license: string | null | undefined,
): LicenseVerdict {
  const slug = (license ?? "").trim().toLowerCase();

  if (!slug) {
    return {
      openness: "unknown",
      label: "Not stated",
      summary:
        "This repository states no licence. No usage rights are granted by default — check the model card before use.",
    };
  }

  const open = own(OPEN_LICENSES, slug);
  if (open) {
    return {
      openness: "open",
      label: open.label,
      summary: open.obligation
        ? `${open.label} permits commercial use, modification and redistribution, provided ${open.obligation}.`
        : `${open.label} permits commercial use, modification and redistribution.`,
    };
  }

  const restricted = own(RESTRICTED_LICENSES, slug);
  if (restricted) {
    // "Downloadable" rather than "public": a licence slug says what the terms are, not whether
    // the repository will hand you the files. meta-llama/*, google/gemma-* and most apple-amlr
    // repos are gated, and this function is given the slug alone.
    return {
      openness: "restricted",
      label: restricted.label,
      summary: `These weights are downloadable, but ${restricted.catch} applies.${restricted.note ? ` ${restricted.note}` : ""} Read the licence before shipping.`,
    };
  }

  const unclear = own(UNCLEAR_LICENSES, slug);
  if (unclear) {
    return {
      openness: "unknown",
      label: unclear.label,
      summary: unclear.summary,
    };
  }

  const proprietaryLabel = own(PROPRIETARY_LICENSES, slug);
  if (proprietaryLabel) {
    return {
      openness: "proprietary",
      label: proprietaryLabel,
      summary:
        "These terms do not grant redistribution. Check the model card for what is permitted.",
    };
  }

  // Keep the repository's own text: a slug we do not know is still the most accurate label
  // we can show, and inventing a verdict for it would be worse than admitting ignorance.
  return {
    openness: "unknown",
    label: license?.trim() || slug,
    summary: `Unrecognised licence "${license?.trim() || slug}". Check the model card for its terms.`,
  };
}
