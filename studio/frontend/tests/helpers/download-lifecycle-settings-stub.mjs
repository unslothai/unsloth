// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export async function loadDownloadTransportSettings() {
  return { mode: "auto" };
}

export function subscribeDownloadTransportSettings() {
  return () => undefined;
}

export async function updateDownloadTransportSettings() {
  return { mode: "auto" };
}
