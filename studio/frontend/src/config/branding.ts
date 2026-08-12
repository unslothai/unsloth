/**
 * The single source of the product's user-visible identity.
 *
 * Per ADR 0000 nothing else may hardcode a product name. `__root.tsx`'s document
 * title and `i18n/locales/en.ts`'s `shell.brand` / `shell.product` both read from
 * here, and `scripts/rag-platform/branding-scan.mjs` fails the build on any
 * user-visible vendor-name occurrence outside its allowlist.
 *
 * The values are frozen by the integration plan §1.2. Changing one needs a
 * superseding ADR, not an edit here.
 *
 * What deliberately does NOT belong in this file: Hugging Face repo ids,
 * `UNSLOTH_*` environment variable names, `unsloth …` CLI invocations and
 * `unsloth_*` localStorage keys. Those are protocol and identifier values —
 * renaming them breaks the call or signs users out. See ADR 0000 decision 3.
 */

/** Displayed product name. Used wherever a user reads the product's name. */
export const PRODUCT_NAME = "Rag Platform";

/**
 * Lowercase wordmark form, for the sidebar mark and anywhere the design calls
 * for the name set in lower case. Not a slug — see `PRODUCT_SLUG`.
 */
export const PRODUCT_WORDMARK = "rag platform";

/** URL / package slug. Safe in paths, ids and file names. */
export const PRODUCT_SLUG = "rag-platform";

/** Default `document.title`, used before a route supplies its own. */
export const DEFAULT_DOCUMENT_TITLE = PRODUCT_NAME;

/**
 * Prefix for exported file names, e.g. `RagPlatform_video_20260624-143005.mp4`.
 * Separate from `PRODUCT_SLUG` because exports use a capitalised,
 * underscore-joined form and users recognise their own files by it.
 */
export const EXPORT_FILE_PREFIX = "RagPlatform";

/** Default `<meta name="description">` content. */
export const PRODUCT_DESCRIPTION =
  "Rag Platform — retrieval-augmented generation over your own documents.";

export const branding = {
  productName: PRODUCT_NAME,
  productWordmark: PRODUCT_WORDMARK,
  productSlug: PRODUCT_SLUG,
  documentTitle: DEFAULT_DOCUMENT_TITLE,
  exportFilePrefix: EXPORT_FILE_PREFIX,
  description: PRODUCT_DESCRIPTION,
} as const;
