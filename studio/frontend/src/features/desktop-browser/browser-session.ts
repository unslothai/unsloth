// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type BrowserRect = { x: number; y: number; width: number; height: number };
export type BrowserSnapshot = {
  sessionId: string;
  url: string;
  title: string;
  loading: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  visible: boolean;
  bounds: BrowserRect;
  actualBounds: BrowserRect | null;
  popupUrl: string | null;
  error: string | null;
};
export type BrowserAction = "back" | "forward" | "reload" | "stop" | "focus" | "dismiss-popup";
export type BrowserTransport = {
  open: (sessionId: string, url: string, rect: BrowserRect) => Promise<BrowserSnapshot>;
  close: (sessionId: string) => Promise<void>;
  setBounds: (sessionId: string, revision: number, rect: BrowserRect, visible: boolean) => Promise<void>;
  snapshot: (sessionId: string) => Promise<BrowserSnapshot>;
  navigate: (sessionId: string, url: string) => Promise<void>;
  action: (sessionId: string, action: BrowserAction) => Promise<void>;
};

export function resolveAddress(input: string): string {
  const value = input.trim();
  if (!value) throw new Error("Enter a website or search term");
  // A bare host with a numeric port looks like a custom scheme to URL.parse.
  // Recognize public dotted hosts first; never reinterpret javascript:/file: as hosts.
  if (/^(?:[\w-]+\.)+[\w-]{2,}:\d{1,5}(?:[/?#].*)?$/i.test(value)) {
    return new URL(`https://${value}`).href;
  }
  if (/^[a-z][a-z\d+.-]*:/i.test(value)) {
    const url = new URL(value);
    if (!(["http:", "https:"].includes(url.protocol)) || url.username || url.password)
      throw new Error("Only public HTTP(S) websites can open here");
    return url.href;
  }
  if (/\s/.test(value) || !/^(?:localhost|(?:[\w-]+\.)+[\w-]{2,}|\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?(?:[/?#].*)?$/i.test(value)) {
    return `https://duckduckgo.com/?q=${encodeURIComponent(value)}`;
  }
  return new URL(`https://${value}`).href;
}

// getBoundingClientRect is CSS pixels; Tauri expects window-relative logical pixels.
// DPR includes page zoom on Chromium, whereas WebKit's CSS-to-point conversion uses
// the applied interface zoom. The monitor scale removes only the physical-pixel factor.
export function logicalRect(rect: BrowserRect, zoom: number, dpr: number, monitorScale: number, chromium: boolean): BrowserRect {
  const factor = chromium && dpr > 0 && monitorScale > 0
    ? dpr / monitorScale
    : (zoom > 0 ? zoom : 1);
  return {
    x: Math.max(0, rect.x * factor),
    y: Math.max(0, rect.y * factor),
    width: Math.max(0, rect.width * factor),
    height: Math.max(0, rect.height * factor),
  };
}

export type SlotRect = Pick<BrowserRect, "x" | "y" | "width" | "height">;
export function shouldSuspendSlot(slot: SlotRect, groupWidth: number, modal: boolean, layers: readonly SlotRect[]): boolean {
  if (groupWidth < 650 || slot.width < 320 || slot.height < 120 || modal) return true;
  return layers.some((layer) => layer.width > 0 && layer.height > 0 &&
    layer.x < slot.x + slot.width && slot.x < layer.x + layer.width &&
    layer.y < slot.y + slot.height && slot.y < layer.y + layer.height);
}

export type BrowserContext = { active: boolean; mode: string; threadId: string | null; selectedThreadId: string | null; newNonce: string | null; projectId: string | null; research: boolean; canvas: boolean };
export function shouldCloseBrowser(previous: BrowserContext, next: BrowserContext): boolean {
  if (!next.active || next.mode !== "single" || next.research || next.canvas) return true;
  if (previous.projectId !== next.projectId || previous.newNonce !== next.newNonce && Boolean(next.newNonce)) return true;
  // First send can adopt an ID while the URL remains ?new= or implicit /chat.
  // Selecting an existing URL thread is a switch even from a fresh composer.
  if (next.selectedThreadId && previous.selectedThreadId !== next.selectedThreadId) return true;
  if (!previous.threadId && next.threadId && !next.selectedThreadId) return false;
  return Boolean(previous.threadId && previous.threadId !== next.threadId);
}

/** One native session; close invalidates every pending async completion immediately. */
export class BrowserSession {
  private id: string;
  private generation = 0;
  private opening = false;
  private ready = false;
  private revision = 0;
  private frame = 0;
  private pending: { rect: BrowserRect; visible: boolean } | null = null;
  private inFlight = false;
  private pollingGeneration: number | null = null;
  private last: { rect: BrowserRect; visible: boolean } | null = null;
  private listeners = new Set<(snapshot: BrowserSnapshot | null, error: string | null) => void>();
  private snapshotValue: BrowserSnapshot | null = null;
  private errorValue: string | null = null;
  private transport: BrowserTransport;
  private raf: (cb: FrameRequestCallback) => number;
  private cancelRaf: (id: number) => void;
  constructor(transport: BrowserTransport, raf: (cb: FrameRequestCallback) => number = (cb) => requestAnimationFrame(cb), cancelRaf: (id: number) => void = (id) => cancelAnimationFrame(id)) {
    this.transport = transport;
    this.raf = raf;
    this.cancelRaf = cancelRaf;
    this.id = crypto.randomUUID();
  }
  get snapshot(): BrowserSnapshot | null { return this.snapshotValue; }
  get error(): string | null { return this.errorValue; }
  subscribe(listener: (snapshot: BrowserSnapshot | null, error: string | null) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  private publish(snapshot: BrowserSnapshot | null = this.snapshotValue, error: string | null = this.errorValue): void {
    this.snapshotValue = snapshot;
    this.errorValue = error;
    for (const listener of this.listeners) listener(snapshot, error);
  }
  async open(url: string, rect: BrowserRect): Promise<void> {
    if (this.opening || this.ready) return;
    this.opening = true;
    const generation = this.generation;
    const id = this.id;
    try {
      const snapshot = await this.transport.open(id, url, rect);
      if (generation !== this.generation) {
        // Native open might complete after close. Native close is serialized behind open.
        await this.transport.close(id);
        return;
      }
      this.ready = true;
      this.publish(snapshot, null);
      if (this.pending) this.schedule();
    } catch (error) {
      if (generation === this.generation) this.publish(null, String(error));
    } finally {
      this.opening = false;
    }
  }
  close(): void {
    const id = this.id;
    this.generation++;
    this.ready = false;
    this.pending = null;
    this.last = null;
    if (this.frame) this.cancelRaf(this.frame);
    this.frame = 0;
    this.publish(null, null);
    void this.transport.close(id).catch(console.error);
  }
  setBounds(rect: BrowserRect, visible: boolean): void {
    const value = { rect, visible };
    if (JSON.stringify(value) === JSON.stringify(this.pending ?? this.last)) return;
    this.pending = value;
    this.schedule();
  }
  private schedule(): void {
    if (!this.ready || this.frame || this.inFlight || !this.pending) return;
    const generation = this.generation;
    this.frame = this.raf(() => {
      this.frame = 0;
      if (!this.ready || generation !== this.generation || !this.pending) return;
      const value = this.pending;
      this.pending = null;
      this.last = value;
      this.inFlight = true;
      void this.transport.setBounds(this.id, ++this.revision, value.rect, value.visible)
        .catch((error: unknown) => {
          if (generation === this.generation) this.publish(this.snapshotValue, String(error));
        })
        .finally(() => {
          this.inFlight = false;
          if (generation === this.generation) this.schedule();
        });
    });
  }
  async poll(): Promise<void> {
    if (!this.ready || this.pollingGeneration === this.generation) return;
    this.pollingGeneration = this.generation;
    const generation = this.generation;
    try {
      const snapshot = await this.transport.snapshot(this.id);
      if (generation === this.generation) this.publish(snapshot, null);
    } catch (error) {
      if (generation === this.generation) this.publish(this.snapshotValue, String(error));
    } finally {
      if (this.pollingGeneration === generation) this.pollingGeneration = null;
    }
  }
  async navigate(url: string): Promise<void> { await this.command(() => this.transport.navigate(this.id, url)); }
  async action(action: BrowserAction): Promise<void> { await this.command(() => this.transport.action(this.id, action)); }
  private async command(run: () => Promise<void>): Promise<void> {
    if (!this.ready) return;
    const generation = this.generation;
    try { await run(); if (generation === this.generation) await this.poll(); }
    catch (error) { if (generation === this.generation) this.publish(this.snapshotValue, String(error)); }
  }
}
