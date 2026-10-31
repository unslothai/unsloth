// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Selector adapter for the real Unsloth chat UI, exposing the smoke fixture's method names.
// Compare mode renders the composer twice, so everything scopes to the first thread root; Stop
// is replaced by Queue when the composer has text; Radix keeps collapsed content mounted, so
// "open" is read from data-state on the root.

(() => {
  if (window.__sb && window.__sb.dom) return;
  window.__sb = window.__sb || {};

  const q = (sel, root) => (root || document).querySelector(sel);
  const qa = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // TooltipIconButton renders its label in a visually hidden span, not an aria-label.
  const nameOf = (el) => {
    if (!el) return "";
    const aria = el.getAttribute("aria-label");
    if (aria) return aria.trim();
    const sr = el.querySelector(".aui-sr-only, .sr-only");
    if (sr && sr.textContent) return sr.textContent.trim();
    return (el.textContent || "").trim();
  };

  // Every control ComposerRightControls can put in the run-state slot.
  const RUN_STATE_CONTROLS = [
    "Stop generating",
    "Stop queued message",
    "Stop research",
    "Stopping research",
    "Queue message",
    "Send message",
  ];

  // Progress hooks: data-status on text parts and aria-busy on reasoning content, named once so
  // the streaming scan and its control cannot drift apart.
  const STATUS_HOOK = '[data-status], [data-slot="reasoning-content"][aria-busy]';

  const byName = (sel, name, root) =>
    qa(sel, root).find((el) => nameOf(el) === name) || null;

  const D = {
    threadRoot() {
      return q(".aui-thread-root") || document.body;
    },
    viewport() {
      return q(".aui-thread-viewport");
    },
    composer() {
      return q('textarea[aria-label="Message input"]');
    },
    composerText() {
      const el = D.composer();
      return el ? el.value : null;
    },
    sendButton() {
      return q('button[aria-label="Send message"]');
    },
    stopButton() {
      return q('button[aria-label="Stop generating"]');
    },
    queueButton() {
      return q('button[aria-label="Queue message"]');
    },
    // Rendered under isQueueRunning && !thread.isRunning, where neither stopButton() nor
    // queueButton() matches; without it that transient looks like a settled Send arm.
    stopQueuedButton() {
      return q('button[aria-label="Stop queued message"]');
    },
    isRunning() {
      return Boolean(D.stopButton() || D.queueButton());
    },
    // Exactly one run-state control renders at a time and each is a different subtree, so arms
    // showing different tokens differ in the scaffold for that reason.
    runStateControl() {
      for (const name of RUN_STATE_CONTROLS) {
        if (byName("button", name)) return name;
      }
      return "";
    },
    // Scoped to the thread root so another thread's queue is not read as this one's.
    promptQueue() {
      return q('[aria-label^="Prompt queue,"]', D.threadRoot());
    },
    // Narrower than isRunning(): "Queue message" also renders while a queued prompt waits on an idle
    // thread. Research runs are deliberately invisible here; the seeder cannot start one.
    generating() {
      if (D.stopButton()) return true;
      if (!D.queueButton()) return false;
      return !D.promptQueue();
    },
    // Read from the app's own published state (data-status, aria-busy), never from timers or
    // "the last assistant message", which attributes a stream to the wrong message.
    streamingMessages() {
      return qa("[data-role]").filter(
        (m) => m.querySelector('[data-status="running"], [aria-busy="true"]') !== null,
      );
    },
    // Tells a renamed data-status hook (blind probe) apart from an unmounted row or a row with no
    // parts yet. data-status is rendered for complete parts too.
    statusHookPresent() {
      return D.assistantMessages().some((m) => m.querySelector(STATUS_HOOK) !== null);
    },
    lastAssistantPublishesStatus() {
      const last = D.lastAssistantMessage();
      return Boolean(last && last.querySelector(STATUS_HOOK));
    },
    messages() {
      return qa("[data-role]");
    },
    messageCount() {
      return qa("[data-role]").length;
    },
    // Thread length, not mounted count: differs once an arm windows. Read from aria-setsize, where
    // WAI-ARIA requires a windowed list to publish it.
    threadTotal() {
      // The ordinal may sit on the message or on the virtualizer's row wrapper (same walk as readiness.py).
      const first = q("[data-role]");
      const owner = first ? first.closest("[aria-setsize]") : q("[aria-setsize]");
      if (owner) {
        const n = Number(owner.getAttribute("aria-setsize"));
        if (Number.isFinite(n) && n >= 0) return n;
      }
      return qa("[data-role]").length;
    },
    isWindowed() {
      return D.threadTotal() > qa("[data-role]").length;
    },
    assistantMessages() {
      return qa('[data-role="assistant"]');
    },
    lastAssistantMessage() {
      const all = qa('[data-role="assistant"]');
      return all.length ? all[all.length - 1] : null;
    },

    // Hidden with `invisible` when the app's autoscroll considers itself at the bottom, so the
    // harness and the app cannot disagree about pinning.
    jumpToBottomButton() {
      return q(".aui-thread-scroll-to-bottom");
    },
    appSaysAtBottom() {
      const jump = D.jumpToBottomButton();
      // null, not false, when absent: a build without the control has told us nothing.
      return jump ? jump.classList.contains("invisible") : null;
    },
    distanceFromBottom() {
      const vp = D.viewport();
      if (!vp) return null;
      return Math.round(vp.scrollHeight - vp.clientHeight - vp.scrollTop);
    },

    reasoningRoots() {
      return qa('[data-slot="reasoning-root"]');
    },
    reasoningTriggers() {
      return qa('[data-slot="reasoning-trigger"]');
    },
    reasoningOpenCount() {
      return qa('[data-slot="reasoning-root"][data-state="open"]').length;
    },
    // Collapsed content stays mounted until animationend/transitionend, so a collapse is settled
    // only once the content is gone.
    reasoningContentMounted() {
      return qa('[data-slot="reasoning-content"]').filter((el) => !el.hasAttribute("hidden"))
        .length;
    },

    actionBar(message) {
      const m = message || D.lastAssistantMessage();
      if (!m) return null;
      return q(".aui-assistant-action-bar-root", m) || q(".aui-user-action-bar-root", m);
    },
    actionButton(name, message) {
      const bar = D.actionBar(message);
      if (bar) {
        const inBar = byName("button", name, bar);
        if (inBar) return inBar;
      }
      const m = message || D.lastAssistantMessage();
      return m ? byName("button", name, m) : null;
    },
    // The action bar uses autohide and unmounts on every message that is not hovered.
    hoverLastAssistantMessage() {
      const m = D.lastAssistantMessage();
      if (m) {
        m.dispatchEvent(
          new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" })
        );
      }
      return m;
    },

    // Polls instead of sampling once: hideWhenRunning removes the bar while generating, and at large
    // rungs the reply settles later than the nominal drain arithmetic predicts.
    async waitForActionButton(name, waitMs, everyMs) {
      const started = performance.now();
      const budget = Math.max(0, Number(waitMs) || 0);
      const nextPaint = () =>
        window.__sbNextPaint
          ? window.__sbNextPaint()
          : new Promise((r) => setTimeout(r, Number(everyMs) || 16));
      D.hoverLastAssistantMessage();
      let el = D.actionButton(name);
      while (!el && performance.now() - started < budget) {
        await nextPaint();
        // The bar unmounts whenever the message re-renders, which is every chunk during a stream.
        D.hoverLastAssistantMessage();
        el = D.actionButton(name);
      }
      return {
        el,
        waitedMs: Math.round((performance.now() - started) * 10) / 10,
        // A miss while running is an unsettled reply; a miss while idle is a genuinely absent control.
        running: D.isRunning(),
      };
    },
    openMenu() {
      return q(".aui-action-bar-more-content");
    },
    openMenuItemCount() {
      const menu = D.openMenu();
      return menu ? qa(".aui-action-bar-more-item", menu).length : 0;
    },
    // Delete lives only in the More menu, not on the reply's action bar.
    menuItem(name) {
      const menu = D.openMenu();
      return menu ? byName(".aui-action-bar-more-item", name, menu) : null;
    },
    // Radix opens on pointerdown, so click() leaves the menu shut. The portaled menu is looked up
    // only when body's children change, not per paint.
    async openMenuAndFind(trigger, name, waitMs) {
      const pointer = { bubbles: true, cancelable: true, composed: true, button: 0,
                        pointerId: 1, pointerType: "mouse", isPrimary: true };
      const budget = Math.max(0, Number(waitMs) || 0);
      const nextPaint = () =>
        window.__sbNextPaint ? window.__sbNextPaint() : new Promise((r) => setTimeout(r, 16));
      let bodyChanged = true;
      const watcher = new MutationObserver(() => { bodyChanged = true; });
      watcher.observe(document.body, { childList: true, subtree: false });
      let menu = null;
      let item = null;
      const look = () => {
        if (!menu || !menu.isConnected) {
          if (!bodyChanged) return;
          bodyChanged = false;
          menu = D.openMenu();
        }
        if (menu) item = byName(".aui-action-bar-more-item", name, menu);
      };
      const started = performance.now();
      try {
        trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
        trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
        look();
        while (!item && performance.now() - started < budget) {
          await nextPaint();
          look();
        }
      } finally {
        watcher.disconnect();
      }
      return {
        item,
        opened: Boolean(menu),
        items: menu ? qa(".aui-action-bar-more-item", menu).length : 0,
        openMs: Math.round((performance.now() - started) * 10) / 10,
      };
    },

    settingsTrigger() {
      return q('button[aria-label="Settings"]');
    },
    settingsDialog() {
      return q('[data-slot="dialog-content"].settings-surface');
    },
    settingsScroller() {
      const dlg = D.settingsDialog();
      if (!dlg) return null;
      return q("main > div.hover-scrollbar.overflow-y-auto", dlg) || q("main div.overflow-y-auto", dlg);
    },
    settingsTab(id) {
      return q('[data-testid="settings-tab-' + id + '"]');
    },

    modelTrigger() {
      return q("button.unsloth-model-selector-trigger");
    },
    modelMenu() {
      return q(".unsloth-model-selector-menu");
    },
    modelOptions() {
      const menu = D.modelMenu();
      return menu ? qa("button[data-model-picker-option]", menu) : [];
    },
    currentModelLabel() {
      const t = D.modelTrigger();
      return t ? (t.textContent || "").trim() : null;
    },

    plusButton() {
      return q('button[aria-label="Tools and attachments"]') || q('[data-tour="chat-plus-menu"]');
    },
    menuItemByText(text) {
      return (
        qa('[role="menuitem"], [role="option"], .aui-action-bar-more-item').find((el) =>
          (el.textContent || "").trim().toLowerCase().includes(text.toLowerCase()),
        ) || null
      );
    },

    threadRows() {
      return qa('[data-testid="recent-thread"]');
    },
    threadRow(id) {
      return q('[data-thread-id="' + id + '"]');
    },
    newChatButton() {
      return q('button[aria-label="New chat"].sidebar-header-action') || q('button[aria-label="New chat"]');
    },

    codeCopyButtons() {
      return qa('button[title="Copy code"]');
    },

    counts() {
      const started = performance.now();
      const out = {
        elements: document.getElementsByTagName("*").length,
        messages: qa("[data-role]").length,
        assistant_messages: qa('[data-role="assistant"]').length,
        reasoning_panes: qa('[data-slot="reasoning-root"]').length,
        reasoning_open: qa('[data-slot="reasoning-root"][data-state="open"]').length,
        code_blocks: qa("pre").length,
        // Field capture density was ~5.6 characters per Shiki span; the fixture must reproduce it.
        highlight_spans: qa("pre span").length,
        // Collapsed reasoning panes unmount children, so span location matters. Tools have two
        // renderers (tool-group-root and tool-fallback-root), so count both.
        tool_groups: qa('[data-slot="tool-group-root"]').length
                   + qa('[data-slot="tool-fallback-root"]').length,
        tool_groups_open: qa('[data-slot="tool-group-content"]').length
                        + qa('[data-slot="tool-fallback-content"]').length,
        reasoning_spans: qa('[data-slot="reasoning-root"] pre span').length,
        reasoning_code_blocks: qa('[data-slot="reasoning-root"] pre').length,
        content_spans:
          qa("pre span").length - qa('[data-slot="reasoning-root"] pre span').length,
        content_code_blocks: qa("pre").length - qa('[data-slot="reasoning-root"] pre').length,
        // Read in the same census so occupancy and character count come from one reading.
        assistant_chars: D.assistantChars(),
        viewport_scroll_height: (D.viewport() || {}).scrollHeight || null,
        viewport_client_height: (D.viewport() || {}).clientHeight || null,
        // If the thread stops following, a windowed list unmounts the streamed message and the cost
        // collapses; app_at_bottom is the app's own state, distance_from_bottom tolerates estimates.
        viewport_scroll_top: (D.viewport() || {}).scrollTop || null,
        distance_from_bottom: D.distanceFromBottom(),
        app_at_bottom: D.appSaysAtBottom(),
      };
      out.census_cost_ms = Math.round((performance.now() - started) * 100) / 100;
      return out;
    },

    assistantChars() {
      let n = 0;
      for (const m of qa('[data-role="assistant"]')) n += (m.textContent || "").length;
      return n;
    },
  };

  window.__sb.dom = D;

  // Sampled in-page and read once per cell, outside every window, to keep CDP round trips out of
  // gap windows. Counters persist in sessionStorage because thread_reopen may navigate.
  const FOLLOW_KEY = "__sb_follow_v1";
  const restore = () => {
    try {
      const raw = window.sessionStorage.getItem(FOLLOW_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  };
  const F = Object.assign({
    samples: 0,
    running_samples: 0,
    running_pinned: 0,
    running_unknown: 0,
    max_distance_while_running: 0,
    suspended_samples: 0,
    detached_samples: 0,
    yanked_back_samples: 0,
    stream_samples: 0,
    reattachments: 0,
    // Never cleared: a thread that drifts away and is later pulled back has still failed.
    ever_fell_behind: false,
  }, restore() || {});
  window.addEventListener("pagehide", () => {
    try {
      window.sessionStorage.setItem(FOLLOW_KEY, JSON.stringify(F));
    } catch (e) {
      // A full sessionStorage degrades to "not measured" instead of failing the page.
    }
  });
  const FOLLOW_TICK_MS = 250;
  // Generous because a virtualizer with estimated heights can sit short of the bottom; under two lines.
  const FOLLOW_TOLERANCE_PX = 64;
  // Two phases: attached (does it follow, gated) and detached after a deliberate scroll (it must
  // not be yanked back, recorded separately). One number cannot score both.
  let detached = false;
  let suspended = 0;
  // A run started after the gesture is a fresh intent to be at the end; re-attach only when such
  // a run is seen at the bottom, since the film's scroll gesture never ends at the bottom.
  let runSeq = 0;
  let wasRunning = false;
  let detachedAtRun = 0;
  setInterval(() => {
    F.samples += 1;
    // Read before the suspended early-return, or a run inside a gesture is never counted.
    const running = D.isRunning();
    if (!running) wasRunning = false;
    else if (!wasRunning) { wasRunning = true; runSeq += 1; }
    if (suspended > 0) { F.suspended_samples += 1; return; }
    if (!running) return;
    const app = D.appSaysAtBottom();
    const distance = D.distanceFromBottom();
    // The app's own answer wins; geometry is used only when no jump control renders.
    const atBottom =
      app === true || (app === null && distance !== null && distance <= FOLLOW_TOLERANCE_PX);
    if (detached) {
      if (runSeq !== detachedAtRun && atBottom) {
        detached = false;
        F.reattachments += 1;
      } else {
        F.detached_samples += 1;
        F.stream_samples += 1;
        if (atBottom) F.yanked_back_samples += 1;
        return;
      }
    }
    F.running_samples += 1;
    F.stream_samples += 1;
    if (distance !== null && distance > F.max_distance_while_running) {
      F.max_distance_while_running = distance;
    }
    if (app === null) {
      F.running_unknown += 1;
      if (distance !== null && distance > FOLLOW_TOLERANCE_PX) F.ever_fell_behind = true;
      return;
    }
    if (app) {
      F.running_pinned += 1;
    } else {
      F.ever_fell_behind = true;
    }
  }, FOLLOW_TICK_MS);

  window.__sb.follow = {
    // Nested-safe. Every suspend latches detached and stamps the current run.
    suspend() { suspended += 1; detached = true; detachedAtRun = runSeq; },
    resume() {
      suspended = Math.max(0, suspended - 1);
      // Re-attach if the gesture ended at the bottom; only checked on gesture exit, so the app pulling
      // the viewport down on its own still counts as a yank.
      if (suspended === 0 && detached) {
        const app = D.appSaysAtBottom();
        const distance = D.distanceFromBottom();
        // Either answer suffices: the invisible class updates from an async scroll listener and can be stale.
        if (app === true || (distance !== null && distance <= FOLLOW_TOLERANCE_PX)) {
          detached = false;
          F.reattachments += 1;
        } else {
          // Re-stamp against the current run so a gesture spanning a run boundary is not re-attached early.
          detachedAtRun = runSeq;
        }
      }
    },
    read() {
      const measured = F.running_samples - F.running_unknown;
      return {
        follow_attempted: true,
        samples: F.samples,
        running_samples: F.running_samples,
        running_pinned: F.running_pinned,
        running_unknown: F.running_unknown,
        suspended_samples: F.suspended_samples,
        detached_samples: F.detached_samples,
        yanked_back_samples: F.yanked_back_samples,
        yanked_after_scroll: F.yanked_back_samples > 0,
        // null, not 1.0, when nothing was sampled mid-run, so it cannot read as a pass.
        pinned_fraction: measured > 0 ? F.running_pinned / measured : null,
        pinned_fraction_reason:
          measured > 0 ? null : "no sample was taken while a reply was streaming",
        // pinned_fraction covers attached phases only; this says how much of the stream that was.
        stream_samples: F.stream_samples,
        attached_fraction_of_stream:
          F.stream_samples > 0 ? F.running_samples / F.stream_samples : null,
        reattachments: F.reattachments,
        max_distance_while_running: F.max_distance_while_running,
        ever_fell_behind: F.ever_fell_behind,
        tolerance_px: FOLLOW_TOLERANCE_PX,
        tick_ms: FOLLOW_TICK_MS,
      };
    },
    reset() {
      try { window.sessionStorage.removeItem(FOLLOW_KEY); } catch (e) {}
      F.samples = 0;
      F.running_samples = 0;
      F.running_pinned = 0;
      F.running_unknown = 0;
      F.max_distance_while_running = 0;
      F.suspended_samples = 0;
      F.detached_samples = 0;
      F.yanked_back_samples = 0;
      F.stream_samples = 0;
      F.reattachments = 0;
      F.ever_fell_behind = false;
      suspended = 0;
      detached = false;
      runSeq = 0;
      wasRunning = false;
      detachedAtRun = 0;
    },
  };
})();
