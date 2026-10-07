// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type AuthFetchHandler = (
  input: string,
  init?: RequestInit,
) => Response | Promise<Response>;

let handler: AuthFetchHandler | null = null;

/** Answer authFetch from `next`; null restores the failing default. */
export function setAuthFetchHandler(next: AuthFetchHandler | null): void {
  handler = next;
}

export function authFetch(input: string, init?: RequestInit): Promise<Response> {
  if (!handler) throw new Error("authFetch: no network in tests");
  return Promise.resolve(handler(input, init));
}

export async function prepareHfTokenForUse(): Promise<null> {
  return null;
}

export function useIsAccountOwner(): boolean {
  return true;
}
