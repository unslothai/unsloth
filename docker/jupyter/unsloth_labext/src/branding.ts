// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
//
// Canonical attribution strings, mirrored from unsloth_branding.py. Imported by
// the About and splash plugins so they're bundled verbatim; the Python guard
// checks the built bundle still contains them. Plain text only, never encoded.

export const PRODUCT = 'Unsloth Docker Studio';
export const SHORT_LABEL = 'Built by the Unsloth team';
// Loading-splash caption; distinct from SHORT_LABEL (says what's loading).
export const SPLASH_LABEL = 'Loading Unsloth Docker';
export const COPYRIGHT = 'Copyright 2026-Present the Unsloth team';
export const AGPL_NOTICE = 'Licensed under Apache 2.0 and the GNU AGPLv3';
export const WEBSITE_URL = 'https://unsloth.ai';
export const DOCS_URL = 'https://unsloth.ai/docs';
export const SOURCE_URL = 'https://github.com/unslothai/unsloth';
export const LICENSE_URL = 'https://github.com/unslothai/unsloth#license';
export const AGPL_URL = 'https://www.gnu.org/licenses/agpl-3.0.html';
export const APACHE_URL = 'https://www.apache.org/licenses/LICENSE-2.0';

// Must equal PHRASE in unsloth_branding.py (the guard greps the bundle for it).
// ONE plain literal, not a concatenation, so webpack keeps it contiguous.
export const PHRASE =
  'Unsloth Docker Studio and JupyterLab image. Built by the Unsloth team. Licensed under Apache 2.0 and the GNU AGPLv3. Source: https://github.com/unslothai/unsloth Website: https://unsloth.ai';
