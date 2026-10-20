// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The clips on the playback timeline. onended used to be the only way a clip left
// the set, so a source started into a context whose clock never moved (WebKit parks
// it around a mic reopen) stayed "playing" forever: the orb stuck on speaking and
// every later aside, the approval prompt included, was dropped behind it.

export const CLIP_WATCHDOG_GRACE_MS = 2000;

export interface LedgerContext {
  readonly currentTime: number;
  readonly state: string;
  resume(): Promise<void>;
}

export interface LedgerSource {
  onended: ((ev: Event) => unknown) | null;
  start(when: number): void;
  stop(): void;
}

export interface LedgerClip {
  source: LedgerSource;
  start: number;
  duration: number;
  text: string;
  env: Float32Array;
}

interface Entry extends LedgerClip {
  addedAt: number;
  seenTime: number;
  timer: unknown;
}

export interface ClipLedgerDeps {
  ctx(): LedgerContext | null;
  /** Paused or held on purpose: a frozen clock is expected, not a fault. */
  frozen(): boolean;
  /** The first clip went on the timeline. */
  onBusy(): void;
  /** The last clip left it, by ending or being reaped. */
  onIdle(): void;
  trace(event: string, detail: Record<string, unknown>): void;
  now(): number;
  setTimer(fn: () => void, ms: number): unknown;
  clearTimer(timer: unknown): void;
}

export interface ClipLedger {
  readonly size: number;
  clips(): Iterable<LedgerClip>;
  add(clip: LedgerClip): void;
  /** A clip is on the speaker right now: started, not finished, clock running. */
  audible(): boolean;
  /** Force out clips that cannot be playing (clock parked, or past their end). */
  reapStale(): number;
  clear(): void;
}

export function createClipLedger(deps: ClipLedgerDeps): ClipLedger {
  const entries = new Set<Entry>();

  const remove = (e: Entry) => {
    if (!entries.delete(e)) return;
    deps.clearTimer(e.timer);
    e.source.onended = null;
    if (entries.size === 0) deps.onIdle();
  };

  const forceEnd = (e: Entry) => {
    const ctx = deps.ctx();
    deps.trace("tts.clipWatchdog", {
      text: e.text,
      ms: Math.round(deps.now() - e.addedAt),
      ctx: ctx?.state,
    });
    try {
      e.source.stop();
    } catch {
      // never started
    }
    remove(e);
    if (ctx && ctx.state !== "running" && !deps.frozen()) void ctx.resume().catch(() => {});
  };

  const budgetMs = (e: Entry) => {
    const t = deps.ctx()?.currentTime ?? 0;
    const remaining = Math.max(0, e.start + e.duration - t);
    return remaining * 1000 + CLIP_WATCHDOG_GRACE_MS;
  };

  const arm = (e: Entry) => {
    e.seenTime = deps.ctx()?.currentTime ?? 0;
    e.timer = deps.setTimer(() => check(e), budgetMs(e));
  };

  const check = (e: Entry) => {
    if (!entries.has(e)) return;
    const ctx = deps.ctx();
    if (deps.frozen()) return arm(e);
    // Still playing on a clock that moves (a late start, a short interruption): wait it out.
    if (ctx && ctx.state === "running" && ctx.currentTime > e.seenTime && ctx.currentTime < e.start + e.duration) {
      return arm(e);
    }
    forceEnd(e);
  };

  const isAudible = (e: Entry, ctx: LedgerContext) =>
    ctx.currentTime >= e.start && ctx.currentTime < e.start + e.duration;

  return {
    get size() {
      return entries.size;
    },
    clips: () => entries,
    add(clip) {
      const e: Entry = { ...clip, addedAt: deps.now(), seenTime: 0, timer: null };
      const wasEmpty = entries.size === 0;
      entries.add(e);
      e.source.onended = () => remove(e);
      e.source.start(e.start);
      arm(e);
      if (wasEmpty) deps.onBusy();
    },
    audible() {
      const ctx = deps.ctx();
      if (!ctx || ctx.state !== "running" || deps.frozen()) return false;
      for (const e of entries) if (isAudible(e, ctx)) return true;
      return false;
    },
    reapStale() {
      const ctx = deps.ctx();
      if (deps.frozen()) return 0;
      let n = 0;
      for (const e of [...entries]) {
        const stale = !ctx || ctx.state !== "running" || ctx.currentTime >= e.start + e.duration;
        if (stale) {
          forceEnd(e);
          n++;
        }
      }
      return n;
    },
    clear() {
      for (const e of entries) {
        deps.clearTimer(e.timer);
        e.source.onended = null;
        try {
          e.source.stop();
        } catch {
          // never started
        }
      }
      entries.clear();
    },
  };
}

/**
 * Whether an aside may play now. A patient one (the approval prompt) is dropped only
 * for something actually audible; a stuck or stale clip is reaped instead.
 */
export function asideBlocked(ledger: ClipLedger, opts: { patient: boolean; replySynth: boolean }): boolean {
  if (!opts.patient) return ledger.size > 0 || opts.replySynth;
  if (ledger.audible()) return true;
  ledger.reapStale();
  return false;
}
