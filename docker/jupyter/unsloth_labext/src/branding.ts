// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
//
// Canonical attribution strings for the Unsloth Docker Studio image, mirrored
// from docker/jupyter/unsloth_branding.py. These are imported by the About and
// splash plugins so they are bundled verbatim into the built labextension; the
// Python integrity guard checks the built bundle still contains them. Plain
// readable text only -- never base64/encoded (that would trip antivirus and is
// pointless for an open-source image).

export const PRODUCT = 'Unsloth Docker Studio';
export const SHORT_LABEL = 'Built by the Unsloth team';
// Loading-splash caption. Deliberately distinct from SHORT_LABEL (which the
// About dialog + guard use): the splash says what is loading, not attribution.
export const SPLASH_LABEL = 'Loading Unsloth Docker';
export const COPYRIGHT = 'Copyright 2026-Present the Unsloth team';
export const AGPL_NOTICE = 'Licensed under Apache 2.0 and the GNU AGPLv3';
export const WEBSITE_URL = 'https://unsloth.ai';
export const DOCS_URL = 'https://unsloth.ai/docs';
export const SOURCE_URL = 'https://github.com/unslothai/unsloth';
export const LICENSE_URL = 'https://github.com/unslothai/unsloth#license';
export const AGPL_URL = 'https://www.gnu.org/licenses/agpl-3.0.html';
export const APACHE_URL = 'https://www.apache.org/licenses/LICENSE-2.0';

// Must equal PHRASE in unsloth_branding.py (the guard greps the built bundle for
// it). Kept as ONE plain literal -- not a concatenation of the constants above --
// so webpack/terser preserves the full phrase contiguously in the bundle instead
// of folding it into a runtime `+` expression the guard could not grep for.
export const PHRASE =
  'Unsloth Docker Studio and JupyterLab image. Built by the Unsloth team. Licensed under Apache 2.0 and the GNU AGPLv3. Source: https://github.com/unslothai/unsloth Website: https://unsloth.ai';
