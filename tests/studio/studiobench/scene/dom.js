// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// The selector adapter for the REAL Studio chat UI.
//
// The salvaged action JS from playwright_heavy_thread.py calls `window.__heavyThread`, an API the
// smoke fixture exported. The shipping app exports nothing of the kind, so the actions are ported
// onto this: the same method names, backed by the app's own selectors. One file to edit when a
// class changes, instead of fifteen.
//
// The chat thread has essentially NO test ids -- exactly one, `composer-tool-status`, in the whole
// of components/assistant-ui and features/chat. So the contract is class hooks (`aui-*`,
// `unsloth-*`), `data-role`, `data-slot` and accessible names, all read out of the shipped TSX.
//
// Three selectors carry a trap and are handled here rather than in each action:
//
//   - The composer exists TWICE in compare mode and once on the welcome screen vs once docked, so
//     everything scopes to the first thread root.
//   - The Stop button is REPLACED by a Queue button when the composer has text, because
//     `queueDisabled` depends on `composerText.trim().length > 0`. An action that presses stop
//     with text in the box presses queue and measures nothing.
//   - Radix keeps a collapsed Collapsible's content element mounted for its animation, so
//     "the reasoning pane is open" cannot be read from the content element's presence. It is read
//     from `data-state` on the root, which is what Radix actually toggles.

(() => {
  if (window.__sb && window.__sb.dom) return;
  window.__sb = window.__sb || {};

  const q = (sel, root) => (root || document).querySelector(sel);
  const qa = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // Accessible name for the app's TooltipIconButton, which renders the label in a visually
  // hidden span rather than an aria-label. `getByRole(name:)` on the driver side would do this,
  // but the actions run inside one page.evaluate so they need it here.
  const nameOf = (el) => {
    if (!el) return "";
    const aria = el.getAttribute("aria-label");
    if (aria) return aria.trim();
    const sr = el.querySelector(".aui-sr-only, .sr-only");
    if (sr && sr.textContent) return sr.textContent.trim();
    return (el.textContent || "").trim();
  };

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
      // aria-label="Stop generating" is the chat one; the class also covers stop-queued and
      // stop-research, which are different buttons for different runs.
      return q('button[aria-label="Stop generating"]');
    },
    queueButton() {
      return q('button[aria-label="Queue message"]');
    },
    isRunning() {
      return Boolean(D.stopButton() || D.queueButton());
    },
    messages() {
      return qa("[data-role]");
    },
    messageCount() {
      return qa("[data-role]").length;
    },
    assistantMessages() {
      return qa('[data-role="assistant"]');
    },
    lastAssistantMessage() {
      const all = qa('[data-role="assistant"]');
      return all.length ? all[all.length - 1] : null;
    },

    // ── reasoning ────────────────────────────────────────────────
    reasoningRoots() {
      return qa('[data-slot="reasoning-root"]');
    },
    reasoningTriggers() {
      return qa('[data-slot="reasoning-trigger"]');
    },
    reasoningOpenCount() {
      // data-state on the ROOT, not the presence of the content element: Radix keeps collapsed
      // content mounted for the animation, so a presence check reads every pane as open.
      return qa('[data-slot="reasoning-root"][data-state="open"]').length;
    },

    // ── action bar ───────────────────────────────────────────────
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
    openMenu() {
      return q(".aui-action-bar-more-content");
    },
    openMenuItemCount() {
      const menu = D.openMenu();
      return menu ? qa(".aui-action-bar-more-item", menu).length : 0;
    },

    // ── settings ─────────────────────────────────────────────────
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

    // ── model picker ─────────────────────────────────────────────
    modelTrigger() {
      return q("button.unsloth-model-selector-trigger");
    },
    modelMenu() {
      return q(".unsloth-model-selector-menu");
    },
    modelOptions() {
      const menu = D.modelMenu();
      // No role="option", no data-model-id: the rows are plain buttons with utility classes, so
      // this is the only available handle and it is recorded as the weak point it is.
      return menu ? qa("button", menu) : [];
    },
    currentModelLabel() {
      const t = D.modelTrigger();
      return t ? (t.textContent || "").trim() : null;
    },

    // ── composer plus menu / attachments ─────────────────────────
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

    // ── sidebar / threads ────────────────────────────────────────
    threadRows() {
      return qa('[data-testid="recent-thread"]');
    },
    threadRow(id) {
      return q('[data-thread-id="' + id + '"]');
    },
    newChatButton() {
      return q('button[aria-label="New chat"].sidebar-header-action') || q('button[aria-label="New chat"]');
    },

    // ── code blocks ──────────────────────────────────────────────
    codeCopyButtons() {
      return qa('button[title="Copy code"]');
    },

    // ── census ───────────────────────────────────────────────────
    counts() {
      const started = performance.now();
      const out = {
        elements: document.getElementsByTagName("*").length,
        messages: qa("[data-role]").length,
        assistant_messages: qa('[data-role="assistant"]').length,
        reasoning_panes: qa('[data-slot="reasoning-root"]').length,
        reasoning_open: qa('[data-slot="reasoning-root"][data-state="open"]').length,
        code_blocks: qa("pre").length,
        // Shiki spans. THE span density check: the field capture ran 90,262 characters against
        // 16,186 spans, i.e. 5.6 characters per span, and a fixture that does not reproduce that
        // is not measuring the same highlighter load per character.
        highlight_spans: qa("pre span").length,
        // WHERE the spans live, not just how many. A collapsed reasoning pane UNMOUNTS its
        // children, so a thread with the same text can carry wildly different DOM depending on
        // how that text got there. Without this split, "seeded has 20% fewer spans" is a number
        // with three possible explanations and no way to choose between them.
        reasoning_spans: qa('[data-slot="reasoning-root"] pre span').length,
        reasoning_code_blocks: qa('[data-slot="reasoning-root"] pre').length,
        content_spans:
          qa("pre span").length - qa('[data-slot="reasoning-root"] pre span').length,
        content_code_blocks: qa("pre").length - qa('[data-slot="reasoning-root"] pre').length,
        // Carried in the census so the peak occupancy and the character count come from the
        // SAME reading. Two separate reads, taken either side of a destructive action, disagree.
        assistant_chars: D.assistantChars(),
        viewport_scroll_height: (D.viewport() || {}).scrollHeight || null,
        viewport_client_height: (D.viewport() || {}).clientHeight || null,
      };
      out.census_cost_ms = Math.round((performance.now() - started) * 100) / 100;
      return out;
    },

    // Characters of assistant text currently in the DOM. Used for the seeded-vs-streamed
    // equivalence check and for chars-per-span.
    assistantChars() {
      let n = 0;
      for (const m of qa('[data-role="assistant"]')) n += (m.textContent || "").length;
      return n;
    },
  };

  window.__sb.dom = D;
})();
