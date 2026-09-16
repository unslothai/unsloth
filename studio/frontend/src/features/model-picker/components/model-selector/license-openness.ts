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

const OPEN_LICENSES: Readonly<Record<string, string>> = {
  "apache-2.0": "Apache 2.0",
  mit: "MIT",
  bsd: "BSD",
  "bsd-2-clause": "BSD 2-Clause",
  "bsd-3-clause": "BSD 3-Clause",
  "cc0-1.0": "CC0 1.0",
  "cc-by-4.0": "CC BY 4.0",
  "cc-by-sa-4.0": "CC BY-SA 4.0",
  "artistic-2.0": "Artistic 2.0",
  isc: "ISC",
  "mpl-2.0": "MPL 2.0",
  "lgpl-3.0": "LGPL 3.0",
  "gpl-3.0": "GPL 3.0",
  "agpl-3.0": "AGPL 3.0",
  unlicense: "Unlicense",
};

// Weights anyone may download, under terms an OSI licence would not impose. Each entry says
// what the catch is, because "restricted" alone tells a user nothing actionable.
const RESTRICTED_LICENSES: Readonly<
  Record<string, { label: string; catch: string }>
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
  "llama3.2": {
    label: "Llama 3.2 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  "llama3.3": {
    label: "Llama 3.3 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  llama4: {
    label: "Llama 4 Community",
    catch: "an acceptable-use policy and a 700M monthly-user ceiling",
  },
  gemma: { label: "Gemma Terms of Use", catch: "a prohibited-use policy" },
  "apple-ascl": { label: "Apple ASCL", catch: "Apple's sample-code terms" },
  "apple-amlr": { label: "Apple ML Research", catch: "research-only terms" },
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
  "cc-by-nc-4.0": { label: "CC BY-NC 4.0", catch: "a non-commercial grant" },
  "cc-by-nc-sa-4.0": {
    label: "CC BY-NC-SA 4.0",
    catch: "a non-commercial share-alike grant",
  },
  "cc-by-nc-nd-4.0": {
    label: "CC BY-NC-ND 4.0",
    catch: "a non-commercial, no-derivatives grant",
  },
  "cc-by-nc-3.0": { label: "CC BY-NC 3.0", catch: "a non-commercial grant" },
  "cc-by-nd-4.0": { label: "CC BY-ND 4.0", catch: "a no-derivatives grant" },
};

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
};

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

  const openLabel = OPEN_LICENSES[slug];
  if (openLabel) {
    return {
      openness: "open",
      label: openLabel,
      summary: `${openLabel} permits commercial use, modification and redistribution.`,
    };
  }

  const restricted = RESTRICTED_LICENSES[slug];
  if (restricted) {
    return {
      openness: "restricted",
      label: restricted.label,
      summary: `Weights are public, but ${restricted.catch} applies. Read the licence before shipping.`,
    };
  }

  const unclear = UNCLEAR_LICENSES[slug];
  if (unclear) {
    return {
      openness: "unknown",
      label: unclear.label,
      summary: unclear.summary,
    };
  }

  const proprietaryLabel = PROPRIETARY_LICENSES[slug];
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
