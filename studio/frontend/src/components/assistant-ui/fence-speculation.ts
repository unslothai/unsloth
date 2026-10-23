// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SpeculationGate = {
  chars: number;
  speculate: (settle: (seeded: boolean) => void) => (() => void) | null;
};

export type FenceSpeculatorOptions<G extends SpeculationGate> = {
  idle: (callback: () => void, timeout: number) => () => void;
  wait: (callback: () => void, ms: number) => () => void;
  enabled: () => boolean;
  busy: () => boolean;
  eligible: (gate: G) => boolean;
  rank: (gates: readonly G[]) => number[];
  maxChars: number;
  budgetChars: number;
  budgetFences: number;
  maxInFlight?: number;
  rankEvery?: number;
};

export type FenceSpeculator<G> = {
  add: (gate: G) => void;
  remove: (gate: G) => void;
  rerank: () => void;
  snapshot: () => {
    candidates: number;
    seeded: number;
    seededChars: number;
    inFlight: boolean;
    inFlightCount: number;
  };
};

export function createFenceSpeculator<G extends SpeculationGate>(
  options: FenceSpeculatorOptions<G>,
): FenceSpeculator<G> {
  const rankEvery = options.rankEvery ?? 16;
  const maxInFlight = Math.max(1, options.maxInFlight ?? 1);
  const candidates = new Set<G>();
  const seeded = new Map<G, number>();
  let seededChars = 0;
  let ranked: G[] | null = null;
  let picks = 0;
  const inFlight = new Map<G, { cancel: (() => void) | null }>();
  let inFlightChars = 0;
  let pending: (() => void) | null = null;

  const schedule = (): void => {
    if (pending || inFlight.size >= maxInFlight || candidates.size === 0) return;
    pending = options.idle(run, 2000);
  };

  const order = (): G[] => {
    if (ranked) return ranked;
    const gates = [...candidates];
    const distances = options.rank(gates);
    ranked = gates
      .map((gate, index) => ({ gate, at: distances[index] ?? Infinity }))
      .sort((a, b) => a.at - b.at)
      .map((entry) => entry.gate);
    picks = 0;
    return ranked;
  };

  const pick = (): { gate: G | null; waiting: boolean } => {
    if (seeded.size + inFlight.size >= options.budgetFences)
      return { gate: null, waiting: false };
    let waiting = false;
    for (const gate of order()) {
      if (!candidates.has(gate)) continue;
      if (seededChars + inFlightChars + gate.chars > options.budgetChars) continue;
      if (!options.eligible(gate)) {
        waiting = true;
        continue;
      }
      return { gate, waiting: false };
    }
    return { gate: null, waiting };
  };

  const finish = (gate: G): boolean => {
    if (!inFlight.delete(gate)) return false;
    inFlightChars -= gate.chars;
    return true;
  };

  const settle = (gate: G, ok: boolean): void => {
    if (!finish(gate)) return;
    if (ok) {
      seeded.set(gate, gate.chars);
      seededChars += gate.chars;
    }
    picks += 1;
    if (picks >= rankEvery) ranked = null;
    schedule();
  };

  const launch = (gate: G): void => {
    candidates.delete(gate);
    const job = { cancel: null as (() => void) | null };
    inFlight.set(gate, job);
    inFlightChars += gate.chars;
    let cancel: (() => void) | null = null;
    try {
      cancel = gate.speculate((ok) => settle(gate, ok));
    } catch {
      settle(gate, false);
      return;
    }
    if (inFlight.get(gate) === job) job.cancel = cancel;
  };

  const run = (): void => {
    pending = null;
    if (!options.enabled()) {
      candidates.clear();
      ranked = null;
      return;
    }
    if (options.busy()) {
      pending = options.wait(run, 1000);
      return;
    }
    while (inFlight.size < maxInFlight) {
      const { gate, waiting } = pick();
      if (!gate) {
        if (waiting && inFlight.size === 0) pending = options.wait(run, 1000);
        return;
      }
      launch(gate);
    }
  };

  return {
    add: (gate) => {
      if (gate.chars === 0 || gate.chars > options.maxChars) return;
      candidates.add(gate);
      ranked = null;
      schedule();
    },
    remove: (gate) => {
      candidates.delete(gate);
      ranked = null;
      const share = seeded.get(gate);
      if (share !== undefined) {
        seeded.delete(gate);
        seededChars -= share;
      }
      const job = inFlight.get(gate);
      if (job) {
        finish(gate);
        job.cancel?.();
      }
      schedule();
    },
    rerank: () => {
      ranked = null;
    },
    snapshot: () => ({
      candidates: candidates.size,
      seeded: seeded.size,
      seededChars,
      inFlight: inFlight.size > 0,
      inFlightCount: inFlight.size,
    }),
  };
}
