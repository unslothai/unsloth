// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let panelAvailable = false;

export function setBrowserPanelAvailable(available: boolean): void {
  panelAvailable = available;
}

export function browserPanelAvailable(): boolean {
  return panelAvailable;
}
