// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Real logic, re-exported: only the barrel it normally comes from needs stubbing.
export { EMBEDDING_TAGS } from "../../../src/features/hub/lib/hf-model-meta.ts";
export { normalizeModelIdentity } from "../../../src/features/hub/lib/model-identity.ts";

let token = "";

export function getHfToken(): string {
  return token;
}

export function hubTokenHeader(): Record<string, string> {
  return {};
}

export function mirrorHfTokenInto(): void {
  // Token state is local to this stub.
}

export const useHfTokenStore = {
  getState: () => ({
    setToken: (next: string) => {
      token = next;
    },
  }),
};
