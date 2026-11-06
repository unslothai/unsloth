// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Detach the stable portal target so React retains message state, refs and handlers.
const REACH = 2000;
const FIND_FLAG = "data-find-bar-open";
const controllers = new WeakMap<HTMLElement, MessageWindowController>();

type Row = {
  root: HTMLElement;
  holder: HTMLElement;
  content: HTMLElement;
  text: HTMLElement;
  saved: Array<[string, string, string]> | null;
  frozen: Array<[string, string, string]> | null;
};

class MessageWindowController {
  private rows = new Map<Element, Row>();
  private observer: IntersectionObserver;
  private flags: MutationObserver;
  private media = window.matchMedia("print");
  private printing = this.media.matches;
  private pointerHeld = false;
  private scrollbarHeld = false;
  private frame = 0;
  private settling = false;

  private viewport: HTMLElement;

  constructor(viewport: HTMLElement) {
    this.viewport = viewport;
    viewport.setAttribute("data-windowed-messages", "");
    // IO records may predate a scroll or prepend; read current geometry instead.
    this.observer = new IntersectionObserver(this.update,
      { root: viewport, rootMargin: `${REACH}px 0px` });
    this.flags = new MutationObserver(this.refresh);
    this.flags.observe(document.documentElement, { attributes: true, attributeFilter: [FIND_FLAG] });
    document.addEventListener("selectionchange", this.refresh);
    document.addEventListener("keydown", this.onKeyDown, true);
    viewport.addEventListener("pointerdown", this.onPointerDown, true);
    viewport.addEventListener("focusin", this.onFocus, true);
    viewport.addEventListener("scroll", this.onScroll, { passive: true });
    window.addEventListener("pointerup", this.onPointerUp, true);
    window.addEventListener("pointermove", this.onPointerMove, { capture: true, passive: true });
    window.addEventListener("pointercancel", this.onPointerUp, true);
    window.addEventListener("blur", this.onPointerUp);
    window.addEventListener("beforeprint", this.beforePrint);
    window.addEventListener("afterprint", this.afterPrint);
    this.media.addEventListener("change", this.printChanged);
  }

  add(root: HTMLElement, holder: HTMLElement, content: HTMLElement): () => void {
    const text = document.createElement("span");
    text.className = "sr-only";
    const row: Row = { root, holder, content, text, saved: null, frozen: null };
    this.rows.set(root, row);
    this.observer.observe(root);
    return () => {
      this.restore(row, false);
      this.releaseHeight(row);
      this.observer.unobserve(root);
      this.rows.delete(root);
      if (this.rows.size === 0) this.dispose();
    };
  }

  private restoreHeight(row: Row, saved: Array<[string, string, string]>) {
    for (const [name, value, priority] of saved) {
      if (value) row.root.style.setProperty(name, value, priority);
      else row.root.style.removeProperty(name);
    }
  }

  private releaseHeight(row: Row) {
    if (!row.frozen) return;
    this.restoreHeight(row, row.frozen);
    row.frozen = null;
  }

  private restore(row: Row, freeze = this.scrollbarHeld) {
    if (!row.saved) return;
    row.text.remove();
    row.holder.append(row.content);
    // Hold the scroll range steady until the native thumb is released.
    if (freeze) row.frozen = row.saved;
    else this.restoreHeight(row, row.saved);
    row.saved = null;
    row.root.removeAttribute("data-message-parked");
  }

  private restoreAll = () => {
    for (const row of this.rows.values()) {
      this.restore(row, false);
      this.releaseHeight(row);
    }
  };

  private held() {
    const selection = document.getSelection();
    return this.printing || document.documentElement.hasAttribute(FIND_FLAG)
      || (selection !== null && !selection.isCollapsed && selection.rangeCount > 0
        && selection.getRangeAt(0).intersectsNode(this.viewport));
  }

  private update = () => {
    if (this.held()) {
      this.restoreAll();
      return;
    }
    const wanted = new Set<Row>();
    const bounds = this.viewport.getBoundingClientRect();
    const reach = this.scrollbarHeld ? Math.min(REACH, bounds.height) : REACH;
    // Batch layout reads before reconnecting or detaching any subtree.
    const candidates: Array<[Row, number]> = [];
    for (const row of this.rows.values()) {
      const rect = row.root.getBoundingClientRect();
      const margin = row.saved ? reach : REACH;
      // Hysteresis avoids repeated swaps on small drag reversals.
      if (rect.bottom >= bounds.top - margin && rect.top <= bounds.bottom + margin
        || row.root.contains(document.activeElement)) {
        wanted.add(row);
        continue;
      }
      if (row.saved || this.pointerHeld || this.settling) continue;
      // Embedded media must stay connected, and wide replies own horizontal scroll range.
      if (row.content.querySelector("iframe, video, audio, canvas, textarea, [contenteditable=true]")) continue;
      const height = rect.height;
      if (height > 0 && row.root.scrollWidth <= row.root.clientWidth + 1) candidates.push([row, height]);
    }
    for (const row of this.rows.values()) {
      if (wanted.has(row) || row.root.contains(document.activeElement)) this.restore(row);
    }
    for (const [row, height] of candidates) {
      row.saved = row.frozen ?? ["height"].map((name) =>
        [name, row.root.style.getPropertyValue(name), row.root.style.getPropertyPriority(name)]);
      row.frozen = null;
      // Preserve accessible text; focus restores the rich body.
      row.text.textContent = row.content.textContent;
      row.root.style.height = `${height}px`;
      row.content.remove();
      row.holder.append(row.text);
      row.root.setAttribute("data-message-parked", "");
    }
  };

  private refresh = () => {
    if (this.held()) this.restoreAll();
    if (this.frame) return;
    // Let find/print code-fence layout changes settle before parking bodies.
    this.settling = true;
    this.frame = requestAnimationFrame(() => {
      this.frame = requestAnimationFrame(() => {
        this.frame = 0;
        this.settling = false;
        this.update();
      });
    });
  };

  private onScroll = () => {
    // Restore before paint, including jumps that outrun IO.
    this.update();
  };

  private onFocus = (event: FocusEvent) => {
    for (const row of this.rows.values()) {
      if (row.root.contains(event.target as Node)) this.restore(row);
    }
  };
  private onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    // Native thumb hits target the viewport; text-selection drags target descendants.
    this.scrollbarHeld = event.target === this.viewport && event.pointerType === "mouse";
    this.pointerHeld = !this.scrollbarHeld;
    if (this.scrollbarHeld && !this.held()) {
      const active = [...this.rows.values()].filter((row) => !row.saved && !row.frozen);
      const heights = active.map((row) => row.root.getBoundingClientRect().height);
      active.forEach((row, index) => {
        row.frozen = [["height", row.root.style.getPropertyValue("height"), row.root.style.getPropertyPriority("height")]];
        row.root.style.height = `${heights[index]}px`;
      });
    }
  };
  private onPointerUp = () => {
    this.pointerHeld = false;
    this.scrollbarHeld = false;
    for (const row of this.rows.values()) this.releaseHeight(row);
    this.refresh();
  };
  private onPointerMove = (event: PointerEvent) => {
    if ((this.pointerHeld || this.scrollbarHeld) && event.buttons === 0) this.onPointerUp();
  };
  private onKeyDown = (event: KeyboardEvent) => {
    // Reconnect before the browser's select-all default action chooses its range.
    const target = event.target as Element | null;
    if (target?.closest?.("input, textarea, [contenteditable=true]")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") this.restoreAll();
  };
  private beforePrint = () => { this.printing = true; this.restoreAll(); };
  private afterPrint = () => { this.printing = this.media.matches; this.refresh(); };
  private printChanged = () => { this.printing = this.media.matches; this.refresh(); };

  private dispose() {
    this.observer.disconnect();
    this.flags.disconnect();
    cancelAnimationFrame(this.frame);
    document.removeEventListener("selectionchange", this.refresh);
    document.removeEventListener("keydown", this.onKeyDown, true);
    this.viewport.removeEventListener("pointerdown", this.onPointerDown, true);
    this.viewport.removeEventListener("focusin", this.onFocus, true);
    this.viewport.removeEventListener("scroll", this.onScroll);
    window.removeEventListener("pointerup", this.onPointerUp, true);
    window.removeEventListener("pointermove", this.onPointerMove, true);
    window.removeEventListener("pointercancel", this.onPointerUp, true);
    window.removeEventListener("blur", this.onPointerUp);
    window.removeEventListener("beforeprint", this.beforePrint);
    window.removeEventListener("afterprint", this.afterPrint);
    this.media.removeEventListener("change", this.printChanged);
    this.viewport.removeAttribute("data-windowed-messages");
    controllers.delete(this.viewport);
  }
}

export function observeMessageWindow(root: HTMLElement, holder: HTMLElement, content: HTMLElement): (() => void) | undefined {
  const viewport = root.closest<HTMLElement>(".aui-thread-viewport");
  if (!viewport || typeof IntersectionObserver === "undefined") return;
  let controller = controllers.get(viewport);
  if (!controller) {
    controller = new MessageWindowController(viewport);
    controllers.set(viewport, controller);
  }
  return controller.add(root, holder, content);
}
