// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useRef } from "react";
import { create } from "zustand";

export interface UpdateOffer {
  source: "desktop" | "web";
  currentVersion: string;
  latestVersion: string;
  notesUrl: string | null;
  actionLabel: string;
  actionDisabled: boolean;
  note: string | null;
  runAction: () => void;
  later: () => void;
}

export type UpdatePillSlot = "titlebar" | "header";

interface UpdateOfferState {
  offer: UpdateOffer | null;
  hosts: Record<UpdatePillSlot, number>;
  setOffer: (offer: UpdateOffer | null) => void;
  mountHost: (slot: UpdatePillSlot) => () => void;
}

export const useUpdateOffer = create<UpdateOfferState>((set) => ({
  offer: null,
  hosts: { titlebar: 0, header: 0 },
  setOffer: (offer) => set({ offer }),
  mountHost: (slot) => {
    set((s) => ({ hosts: { ...s.hosts, [slot]: s.hosts[slot] + 1 } }));
    return () =>
      set((s) => ({ hosts: { ...s.hosts, [slot]: s.hosts[slot] - 1 } }));
  },
}));

export function displayVersion(version: string): string {
  return version.replace(/^v/, "").replace(/-beta$/, "");
}

export function usePillHosted(): boolean {
  return useUpdateOffer((s) => s.hosts.titlebar + s.hosts.header > 0);
}

export function usePillHost(slot: UpdatePillSlot, enabled = true): void {
  const mountHost = useUpdateOffer((s) => s.mountHost);
  useEffect(
    () => (enabled ? mountHost(slot) : undefined),
    [enabled, mountHost, slot],
  );
}

/** Publishes `offer` while mounted; callbacks may change identity every render. */
export function usePublishUpdateOffer(offer: UpdateOffer | null): void {
  const setOffer = useUpdateOffer((s) => s.setOffer);
  const key = offer
    ? [
        offer.source,
        offer.currentVersion,
        offer.latestVersion,
        offer.notesUrl,
        offer.actionLabel,
        offer.actionDisabled,
        offer.note,
      ].join("\u0000")
    : null;
  const latest = useLatestRef(offer);
  useEffect(() => {
    const current = latest.current;
    if (key === null || !current) {
      return;
    }
    setOffer({
      ...current,
      runAction: () => latest.current?.runAction(),
      later: () => latest.current?.later(),
    });
    return () => setOffer(null);
  }, [key, latest, setOffer]);
}

function useLatestRef<T>(value: T): { current: T } {
  const ref = useRef(value);
  // Runs before the publishing effect below, so it always sees this render's value.
  useLayoutEffect(() => {
    ref.current = value;
  });
  return ref;
}
