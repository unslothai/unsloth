// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Set by the header's cast button, so the run settings open on the Characters section even if it was collapsed. */
export const useCastReveal = create<{ pending: boolean }>(() => ({ pending: false }));

export function requestCastReveal(): void {
  useCastReveal.setState({ pending: true });
}

export function consumeCastReveal(): void {
  useCastReveal.setState({ pending: false });
}
