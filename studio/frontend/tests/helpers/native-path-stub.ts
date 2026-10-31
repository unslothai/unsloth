// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The real export reaches @tauri-apps/api, which bare Node cannot load.

export interface NativePathLeaseResponse {
  nativePathLease: string;
}

type Handler = (
  token: string,
  operation: string,
) => NativePathLeaseResponse | Promise<NativePathLeaseResponse>;

let handler: Handler | null = null;

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
