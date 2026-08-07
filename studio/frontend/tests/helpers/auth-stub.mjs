// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for src/features/auth/index.ts. The real barrel re-exports
// login-page.tsx, and node --experimental-strip-types cannot parse JSX, so a
// settings API unit test would pull in the whole React tree. Same cut as
// export-api-stub.mjs. Forwards to globalThis.fetch, which the test replaces.

export async function authFetch(input, init) {
  return globalThis.fetch(input, init);
}
