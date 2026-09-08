// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The one native-intents export the memory-estimate API module imports. The real one
// reaches @tauri-apps/api, which bare Node cannot load and which would answer nothing
// useful outside the desktop shell anyway.

export interface NativePathLeaseResponse {
  nativePathLease: string;
}

type Handler = (
  token: string,
  operation: string,
) => NativePathLeaseResponse | Promise<NativePathLeaseResponse>;

let handler: Handler | null = null;

/** Answer the next lease exchange from `next`. Pass null to restore the default,
 *  which rejects the way an expired or revoked lease does. */
export function setNativePathHandler(next: Handler | null): void {
  handler = next;
}

export async function consumeNativePathToken(
  token: string,
  operation: string,
): Promise<NativePathLeaseResponse> {
  if (!handler) {
    throw new Error("consumeNativePathToken: no native shell in tests");
  }
  return handler(token, operation);
}
