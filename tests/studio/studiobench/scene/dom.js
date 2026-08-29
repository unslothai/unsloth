// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// The selector adapter for the REAL Unsloth chat UI.
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

  // The two hooks the app publishes a part's progress through: assistant-ui's `data-status` on a
  // text part (markdown-text.tsx) and `aria-busy` on the reasoning content (reasoning.tsx). Named
  // once because the streaming scan and the control on it must never drift apart.
  // Every control ComposerRightControls can put in the run-state slot, in no significant order.
  // "Stop research" / "Stopping research" are included because they are the same slot, even though
  // nothing in this benchmark can reach a research run today.
  const RUN_STATE_CONTROLS = [
    "Stop generating",
    "Stop queued message",
    "Stop research",
    "Stopping research",
    "Queue message",
    "Send message",
  ];

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
      // aria-label="Stop generating" is the chat one; the class also covers stop-queued and
      // stop-research, which are different buttons for different runs.
      return q('button[aria-label="Stop generating"]');
    },
    queueButton() {
      return q('button[aria-label="Queue message"]');
    },
    // THE DISPATCHED HALF OF A QUEUE RUN. `ComposerRightControls` renders this once the queued
    // entry has been dispatched, under `isQueueRunning && !thread.isRunning` (thread.tsx), so the
    // thread reports itself NOT running and neither `stopButton()` nor `queueButton()` matches.
    // Read on its own, that transient came out as an ordinary idle composer: `streaming` and
    // `queued_idle` both false, exactly as a settled Send arm reads. An arm caught here against a
    // settled one then had a differing composer with no run-state difference to explain it, which
    // is the shape the comparison layer treats as a rendering regression.
    stopQueuedButton() {
      return q('button[aria-label="Stop queued message"]');
    },
    isRunning() {
      return Boolean(D.stopButton() || D.queueButton());
    },
    // WHICH RUN-STATE CONTROL THE COMPOSER IS RENDERING, as a token.
    //
    // `ComposerRightControls` renders exactly one of these at a time and which one is a function
    // of the run state: Send when nothing is happening, Stop while a reply is being written, Queue
    // while one is queued or while text sits in the box mid-reply, and the research pair while a
    // research run is going. They are DIFFERENT SUBTREES, and the composer dock is inside
    // `.aui-thread-root`, so `digest_scaffold` carries whichever one is up.
    //
    // Two arms showing different tokens differ in the scaffold FOR THAT REASON, with no rendering
    // difference between them. `isRunning()` is too coarse to say so: it is true for Stop AND for
    // Queue, so a queued-idle arm and a streaming arm agree on it while rendering different
    // controls. The token is the thing the scaffold actually contains.
    runStateControl() {
      for (const name of RUN_STATE_CONTROLS) {
        if (byName("button", name)) return name;
      }
      return "";
    },
    // THE PROMPT QUEUE'S OWN SURFACE. `PromptQueueStack` renders the waiting prompts inside the
    // composer root with the accessible name "Prompt queue, <n> of <m>", and that is the only
    // place the queue states its own existence in the DOM. Scoped to the thread root so a queue
    // running on ANOTHER thread, which the sidebar also renders, is not read as this one's.
    promptQueue() {
      return q('[aria-label^="Prompt queue,"]', D.threadRoot());
    },
    // A REPLY IS ACTUALLY BEING WRITTEN, which is NOT the question `isRunning()` answers.
    //
    // `isRunning()` has to accept the Queue button: with text in the composer a running thread
    // renders that button and NO Stop button (`queueDisabled` depends on the composer having a
    // queueable prompt), so an action that waited for Stop would wait out its budget on a live
    // stream. Every caller that asks "may I send now" wants exactly that broad reading.
    //
    // It is the wrong reading for a positive control on the streaming probe, because
    // `ComposerRightControls` renders "Queue message" in a SECOND place: under
    // `isQueueRunning && !thread.isRunning`, while a queued prompt waits to be dispatched and
    // nothing at all is generating. `isRunning()` cannot tell that queued-idle interval from a
    // live stream whose `data-status` hooks have gone quiet, and those two need opposite
    // treatment -- one is an ordinary settled thread, the other is an instrument that has stopped
    // working.
    //
    // The queued-idle button always comes with the queue surface: `syncPromptQueueUI` marks the
    // entry `dispatched` from the active item, and `getPromptQueueUIItemsForRun` drops only
    // DISPATCHED items, so an undispatched active item -- the one that renders "Queue message"
    // rather than "Stop queued message" -- is always in the list the stack renders.
    //
    // WHAT THIS GIVES UP, pinned in scene/selftest/test_studiobench_queued_idle_live.py: while a
    // queue run holds further prompts AND a reply is streaming AND the composer has text, the
    // queue surface is up and the only control is the Queue button, so this reads false and the
    // control is not armed for that capture. Under-claiming, and cheap: probe blindness is a
    // renamed selector, which is global, so every other capture in the run still catches it.
    // A RESEARCH RUN IS A GENERATION THIS PREDICATE CANNOT SEE, and deliberately so for now.
    // `ComposerRightControls` renders "Stop research" / "Stopping research" under
    // `isResearchActive`, which neither `stopButton()` nor `queueButton()` matches, and a research
    // report renders through `MarkdownPreview` rather than the assistant text part, so
    // `streamingMessages()` finds no `data-status` for it either. All THREE are blind to it
    // together.
    //
    // Not patched, because nothing here can start one: `ResearchMessage` is gated on
    // `message.metadata` and the seeder writes `None` for every message, and no action, corpus
    // unit or scene touches the composer's research toggle. Fixing one of the three would leave
    // two that still disagree with it. If a research scene is ever added, move all three in the
    // same change, and add the report's own busy hook to `STATUS_HOOK` at the same time.
    generating() {
      if (D.stopButton()) return true;
      if (!D.queueButton()) return false;
      return !D.promptQueue();
    },
    // WHICH MESSAGES ARE STILL BEING WRITTEN, read from the app's own published state rather than
    // guessed at from `isRunning()` plus "it is probably the last one".
    //
    // assistant-ui gives every text part a status and Unsloth serialises it: markdown-text.tsx
    // renders `<div data-status={status.type}>` around the Streamdown tree, so a part that is
    // still arriving reads `data-status="running"` and a finished one reads `"complete"`. The
    // reasoning pane publishes the same fact as `aria-busy` on `[data-slot="reasoning-content"]`
    // (reasoning.tsx), which is a separate part of the same message and can be running while the
    // answer is not, or the other way round. Either one means this message's DOM is mid-flight.
    //
    // Both are the APP's statements about its own state. Nothing here reads a timer, a character
    // count or "the last assistant message", all of which are the benchmark inventing a fact the
    // page already publishes -- and inventing it is how you end up attributing a stream to the
    // wrong message on the arm that renders faster.
    streamingMessages() {
      return qa("[data-role]").filter(
        (m) => m.querySelector('[data-status="running"], [aria-busy="true"]') !== null,
      );
    },
    // ── IS THE STREAMING PROBE BLIND, OR IS THERE SIMPLY NOTHING FOR IT TO SEE ──────────
    //
    // `streamingMessages()` returning nothing has three completely different causes, and the
    // positive control on it is only worth having if they are told apart.
    //
    //   THE HOOK IS GONE          a build renamed or dropped `data-status`. It is one line in
    //                             markdown-text.tsx and it is rendered for `complete` parts as
    //                             well as `running` ones, so on a working build EVERY assistant
    //                             message that has rendered a part carries it, and on a blinded
    //                             build none does.
    //   THE ROW IS NOT MOUNTED    a windowed arm scrolled away from the tail has unmounted the
    //                             message it is writing into. That is what windowing is for.
    //   THE ROW HAS NO PARTS YET  between the send being accepted and the reply's first part
    //                             arriving, the assistant message is mounted with zero content
    //                             parts and thread.tsx renders "Generating..." in its place. It
    //                             publishes no status because it has nothing to publish yet, and
    //                             `send_turn` returns the instant `isRunning()` flips, so a
    //                             capture lands here twice a film.
    //
    // Scoped to ASSISTANT messages throughout: a user message never publishes `data-status` even
    // on a perfectly working build, because only the assistant parts are rendered through
    // `MarkdownText` (thread.tsx `ASSISTANT_PART_COMPONENTS`). A window holding only user rows
    // would otherwise read as a build with no hook.
    statusHookPresent() {
      return D.assistantMessages().some((m) => m.querySelector(STATUS_HOOK) !== null);
    },
    // Whether the message a reply would be written INTO is publishing parts this probe can read.
    // False means it has none yet, which is the third case above and not a broken instrument.
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
    // HOW LONG THE THREAD IS, as opposed to how much of it is mounted.
    //
    // On the shipped build these are the same number and this returns exactly what
    // messageCount() does, so nothing about an existing run changes. They stop being the same
    // number the moment an arm mounts a window, and at that point every before/after assertion in
    // actions.py -- send_turn's "the thread grew", delete's "the count dropped", thread_reopen's
    // "it came back with the same messages" -- is asking about the THREAD and being answered
    // about the window. Under a windowed mount those three all read the wrong answer in the same
    // direction: the window refills as fast as it empties, so a delete that worked reports
    // `after == before` and fails, and a send that worked reports the same and fails.
    //
    // aria-setsize is where a windowed list is already required to publish this: WAI-ARIA says a
    // list whose items are not all in the DOM must carry aria-setsize and aria-posinset. So this
    // is not a private channel invented for the benchmark, it is the accessible name for the
    // quantity, and an arm that does not provide it is one a screen reader cannot navigate.
    threadTotal() {
      // On the message, or on the row wrapper a virtualizer positions it in. See the same walk in
      // runtime/readiness.py: the ordinal belongs on the element that is a member of the set, and
      // for a windowed list that is the positioned row rather than the message inside it.
      const first = q("[data-role]");
      const owner = first ? first.closest("[aria-setsize]") : q("[aria-setsize]");
      if (owner) {
        const n = Number(owner.getAttribute("aria-setsize"));
        if (Number.isFinite(n) && n >= 0) return n;
      }
      return qa("[data-role]").length;
    },
    // True when the thread is publishing a total that is larger than what it has mounted, i.e.
    // this really is a windowed mount and not merely a short thread.
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

    // ── following the stream ─────────────────────────────────────
    //
    // The jump-to-bottom control, which thread.tsx renders permanently and hides with
    // `invisible` when the intent-aware autoscroll reports itself at the bottom. Reading the
    // app's own state rather than recomputing it means the harness and the app cannot disagree
    // about whether the thread is pinned, which is the whole question here.
    jumpToBottomButton() {
      return q(".aui-thread-scroll-to-bottom");
    },
    appSaysAtBottom() {
      const jump = D.jumpToBottomButton();
      // `null`, NOT `false`, when the control is absent. A build that does not render it has not
      // told us it is scrolled up; it has told us nothing, and the two must not be summed.
      return jump ? jump.classList.contains("invisible") : null;
    },
    distanceFromBottom() {
      const vp = D.viewport();
      if (!vp) return null;
      return Math.round(vp.scrollHeight - vp.clientHeight - vp.scrollTop);
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
    // STILL MOUNTED IS NOT THE SAME AS STILL OPEN, and closing needs the other one.
    //
    // `reasoningOpenCount` flips on the click, because `data-state` is the state. The CHILDREN
    // outlive it on both collapse mechanisms and by design: Radix's `Presence` suspends the
    // unmount until `animationend` of `animate-collapsible-up`, and the grid arm's
    // `UnmeasuredCollapsibleContent` renders `present && children` until the `grid-template-rows`
    // `transitionend` or its 250 ms backstop. For that whole window every pane is closed, every
    // span it contributed is still in the document, and a census asked whether it has stopped
    // moving answers yes -- because it has not started.
    //
    // So a collapse is settled when the content is GONE, which is what this counts. Both arms are
    // covered by one selector: Radix removes the element, the grid arm leaves it behind with
    // `hidden`, and neither is a mounted pane.
    reasoningContentMounted() {
      return qa('[data-slot="reasoning-content"]').filter((el) => !el.hasAttribute("hidden"))
        .length;
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
    // Hover the last assistant message, which is what mounts its action bar. `autohide` on the
    // bar unmounts it on every message that is not hovered, so a control read without this is
    // read out of a tree it was never in.
    hoverLastAssistantMessage() {
      const m = D.lastAssistantMessage();
      if (m) {
        m.dispatchEvent(
          new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" })
        );
      }
      return m;
    },

    // WAIT for one of the action bar's controls, up to `waitMs`, instead of sampling for it once.
    //
    // WHY A WAIT AND NOT A SAMPLE, which is the whole point of this method. The assistant action
    // bar is mounted with `hideWhenRunning` (thread.tsx), so while the thread is generating there
    // is no Copy, no Delete and no More ANYWHERE in the tree -- not hidden, not disabled, absent.
    // Every action that needs one of them is therefore scheduled after a `send_turn`, on the
    // arithmetic that the follow-up turn drains in FOLLOW_UP_CHARS / the field cadence. That
    // arithmetic is the NOMINAL drain: it assumes the pacer's cadence is the binding constraint,
    // and at the 100K rung it is not -- the renderer is, so the reply arrives about 25% later
    // than the cadence says, and the slot can open with the last few chunks still in flight.
    //
    // Measured on the CI runner, from the payload of the run that failed the liveness gate: the
    // `message_menu` window opened at 32,000ms, took ONE more SSE chunk inside itself, and the
    // reply stopped growing 71 characters later, inside the same window. The reply settled about
    // a third of a second after the instant the action sampled the DOM. A single sample turns
    // that third of a second into `NOT RUN -- no More button on the last assistant message`,
    // which reads like a missing control and is really a clock being read too early.
    //
    // So the control is waited for. The wait is bounded, it is reported, and it happens BEFORE
    // any measurement clock starts, so a cell that had to wait is not a cell whose open latency
    // was inflated by the waiting. `fixture/selftest/test_studiobench_rung_plan.py` already
    // asserts this behaviour in prose -- "a slot may legitimately open a little before the
    // follow-up finishes and wait inside its own budget" -- and this is the code that makes the
    // sentence true.
    //
    // Polling per paint, not per millisecond, and scoped: `actionButton` searches inside the last
    // assistant message, so this is O(that message) rather than O(the thread), and it stops the
    // instant the control appears (0 iterations on every cell where the reply has settled, which
    // is the normal case).
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
        // Re-hovered every pass: the bar unmounts again whenever the message re-renders, which
        // during a stream is on every chunk.
        D.hoverLastAssistantMessage();
        el = D.actionButton(name);
      }
      return {
        el,
        waitedMs: Math.round((performance.now() - started) * 10) / 10,
        // Recorded whether the control was found or not. A miss with `running: true` is the
        // reply not having settled; a miss with `running: false` is a control that is genuinely
        // not there, and those are different bugs.
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
        // Tool components. TWO markers, because there are two renderers: a known tool gets a
        // `tool-group-root`, and anything else gets the generic `tool-fallback-root` ("Used
        // tool"). Counting only the first read ZERO on a thread that visibly contained tool
        // blocks -- the second wrong selector in a row for this one component, and both times the
        // reading was a confident zero rather than an error.
        tool_groups: qa('[data-slot="tool-group-root"]').length
                   + qa('[data-slot="tool-fallback-root"]').length,
        tool_groups_open: qa('[data-slot="tool-group-content"]').length
                        + qa('[data-slot="tool-fallback-content"]').length,
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
        // ── DOES THE THREAD STILL FOLLOW THE STREAM? ──────────────
        //
        // Three readings, taken with every census, so the answer exists for every window of the
        // film rather than being reconstructed afterwards from timings.
        //
        // This is here because of a specific way a virtualized arm can produce a beautiful and
        // meaningless frame rate. If the thread stops following, the message being streamed
        // drifts out of the viewport; a windowed list then UNMOUNTS it, and the streaming cost
        // collapses to almost nothing. That is not a measurement of virtualization, it is a
        // measurement of not rendering the thing being measured, and it flatters the arm in
        // exactly the direction this campaign keeps having to catch. An fps number from such a
        // run must never be readable without this beside it.
        //
        // `app_at_bottom` is the app's OWN state, not arithmetic: thread.tsx renders the
        // scroll-to-bottom control permanently and hides it with `invisible` exactly when
        // use-intent-aware-autoscroll considers itself at the bottom. `distance_from_bottom` is
        // kept alongside because a virtualizer working from estimated row heights can sit a few
        // pixels off while the app is quite correctly pinned.
        viewport_scroll_top: (D.viewport() || {}).scrollTop || null,
        distance_from_bottom: D.distanceFromBottom(),
        app_at_bottom: D.appSaysAtBottom(),
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

  // ── the follow-the-stream sampler ──────────────────────────────────────────────────────────
  //
  // WHY THIS IS NOT DONE FROM THE DRIVER. The question is "while a reply was streaming, was the
  // thread pinned to the bottom", and the stream runs during the gap windows -- which are
  // measured windows whose whole purpose is to observe the page doing nothing but stream. A
  // `page.evaluate` per sample would put a CDP round trip and a forced style read inside exactly
  // those windows, four times a second, which is the same class of mistake as the census that
  // used to run inside the action windows.
  //
  // So it samples in the page and is READ ONCE PER CELL, outside every window. A 250ms timer,
  // four ticks a second: frames.js documents a 1ms timer as costing nothing at ~150 ticks a
  // second, so this is two orders of magnitude below that. It does no layout it has not already
  // caused -- `classList.contains` is free, and `distanceFromBottom` is only read when a run is
  // in progress.
  //
  // WHAT IT IS FOR. A thread that stops following unmounts the message being streamed, and the
  // streaming cost then collapses because the renderer is no longer rendering the thing under
  // measurement. That produces a superb frame rate and means nothing. `pinned_fraction` is the
  // reading that makes an fps number from a windowed arm readable at all.
  // THE COUNTERS SURVIVE A NAVIGATION, via sessionStorage.
  //
  // They did not, and the symptom was a confident "NOT MEASURED" on the arm that behaved best.
  // The film ends with `thread_reopen`, which falls back to `page.goto` when the New chat control
  // is covered; a full document navigation destroys the JS context and re-runs the init scripts,
  // so a sampler that lived only in memory came back at zero and the whole cell's streaming
  // phase was reported as never sampled. The treatment arm, whose thread_reopen did not run at
  // all, kept its counters and looked like the only arm with data.
  //
  // sessionStorage is per-origin and per-tab and outlives a same-origin navigation, which is
  // exactly the lifetime wanted: one cell, one page, however many documents it passes through.
  // Saved on pagehide rather than on every tick, so the cost is one serialisation per navigation
  // rather than four a second.
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
    // Set once the thread is seen to fall behind while a run is in progress, and never cleared.
    // A thread that drifts away and is later yanked back to the bottom has still failed the
    // contract, and an end-of-cell reading would show it pinned.
    ever_fell_behind: false,
  }, restore() || {});
  window.addEventListener("pagehide", () => {
    try {
      window.sessionStorage.setItem(FOLLOW_KEY, JSON.stringify(F));
    } catch (e) {
      // A full sessionStorage is not worth losing the page over; the reading degrades to
      // "not measured", which is the honest outcome and is already handled.
    }
  });
  const FOLLOW_TICK_MS = 250;
  // How far from the bottom still counts as following. Generous on purpose: a virtualizer
  // working from estimated row heights corrects them as real rows are measured, so it can sit a
  // little short of the exact bottom while behaving perfectly. 64px is under two lines of text,
  // so it cannot hide a thread that has stopped following.
  const FOLLOW_TOLERANCE_PX = 64;
  // TWO PHASES, BECAUSE THE INTENT CONTRACT HAS TWO HALVES.
  //
  // The contract (plans/proud-wiggling-falcon.md): autoscroll follows a stream, AND a user who
  // has scrolled up is never yanked down. Those are opposite requirements about the same
  // scrollTop, so one number cannot score both -- and the first version of this sampler tried,
  // producing 47% to 50% pinned on BOTH arms with an identical 6,615px worst drift. That figure
  // is the film: `scroll_during_generation` drags the viewport thousands of pixels up, twice,
  // while the reply is still arriving, and the app then CORRECTLY declines to drag it back. The
  // sampler was recording the app honouring the second half of the contract and scoring it as a
  // failure of the first.
  //
  // ATTACHED (before the harness scrolls anywhere): the thread must stay pinned as content
  // arrives. This is "does it follow", and it is what the gate scores.
  // DETACHED (after the harness has deliberately scrolled): the thread must NOT come back to the
  // bottom on its own. Any sample that finds it pinned again while still streaming is the app
  // yanking the user down, which is the other half of the contract and is recorded as its own
  // finding rather than mixed into the fraction.
  let detached = false;
  let suspended = 0;
  // WHICH RUN THE USER SCROLLED AWAY FROM, and it is the difference between a yank and a return.
  //
  // `resume()` below clears `detached` only when the gesture itself ended at the bottom. On the
  // real films it never does: `SCROLL_JS` jumps to the bottom and then steps 14 x 420px away from
  // it, so at any rung whose thread is taller than 5,880px the gesture ends thousands of pixels
  // up and `detached` is latched for the rest of the cell. The film then starts TWO MORE RUNS of
  // its own -- `stop_generation` and `send_turn` both submit a turn -- and the app pins to the
  // bottom for them, which session.py already documents as intended behaviour rather than a
  // violation. Every one of those samples was landing in the detached branch, counted as a yank,
  // and excluded from `attached_fraction_of_stream`.
  //
  // Measured, at head, across every 100K payload in outputs/: attached_fraction 0.07 to 0.15 with
  // reattachments 0, on the BASE arm as well as the treatment and on pure null controls, so the
  // `follows_the_stream` gate failed every 100K cell of every run including two copies of the
  // shipped build. It passed only on the 1K smoke film, where the thread is short enough that the
  // gesture's reversal happens to land back at the bottom -- i.e. the verdict was a property of
  // the thread's height, not of the app.
  //
  // So a run the user STARTED is a fresh expression of intent to be at the end, exactly as
  // returning to the bottom is. Re-attachment is granted only when a run that began AFTER the
  // gesture is also OBSERVED at the bottom: an app that declines to pin stays detached and is
  // scored as before, and a pin during the SAME run the user scrolled away from is still a yank.
  let runSeq = 0;
  let wasRunning = false;
  let detachedAtRun = 0;
  setInterval(() => {
    F.samples += 1;
    // Read BEFORE the suspended early-return, or a run that starts and ends inside a deliberate
    // gesture is never seen and the run after it is mistaken for the one the user scrolled away
    // from.
    const running = D.isRunning();
    if (!running) wasRunning = false;
    else if (!wasRunning) { wasRunning = true; runSeq += 1; }
    if (suspended > 0) { F.suspended_samples += 1; return; }
    if (!running) return;
    const app = D.appSaysAtBottom();
    const distance = D.distanceFromBottom();
    // "At the bottom" by the app's own answer, with the geometry standing in only when the build
    // renders no jump control. `app === false` is the app saying no and is never overridden.
    const atBottom =
      app === true || (app === null && distance !== null && distance <= FOLLOW_TOLERANCE_PX);
    if (detached) {
      if (runSeq !== detachedAtRun && atBottom) {
        // A run the user started, found at the end: following again.
        detached = false;
        F.reattachments += 1;
      } else {
        F.detached_samples += 1;
        F.stream_samples += 1;
        // Pinned again, without anybody asking, inside the run they scrolled away from.
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
    // Called by any action that moves the viewport on purpose. Nested-safe, because more than one
    // scope may legitimately be open at once.
    // Called by any action that moves the viewport on purpose. Nested-safe. The FIRST suspend
    // also latches `detached`: from that moment the user has expressed an intent to be somewhere
    // other than the bottom, and everything after it is scored against the second half of the
    // contract instead of the first.
    // `detachedAtRun` is stamped on every suspend rather than only the first, so a second gesture
    // during a later run detaches from THAT run: without it, scrolling away during run 3 would be
    // compared against run 1 and the very next at-bottom sample would re-attach.
    suspend() { suspended += 1; detached = true; detachedAtRun = runSeq; },
    resume() {
      suspended = Math.max(0, suspended - 1);
      // RE-ATTACH IF THE GESTURE LEFT US AT THE END, and this is not a nicety.
      //
      // `detached` used to latch on the first suspend and never clear, so from the harness's
      // first deliberate scroll onwards every remaining sample went to the detached branch and
      // `running_samples` stopped growing. In the shipped film that scroll happens 1.5s into an
      // 18s opening stream and is followed by two more streamed turns, so the follow verdict was
      // computed from the first ~3s and covered 13% of the streaming time -- while reading, and
      // being reported as, "the thread follows the stream". Measured: running_samples 11,
      // detached_samples 72.
      //
      // The contract is about INTENT, and intent is re-expressed by coming back: a user who
      // scrolls up is detached until they return to the end, at which point they are following
      // again. That is the same re-attachment Unsloth's own intent-aware autoscroll implements.
      // Only checked here, on the way out of a deliberate gesture, so the app silently pulling
      // the viewport down on its own is still scored as a yank rather than laundered into a
      // re-attachment.
      if (suspended === 0 && detached) {
        const app = D.appSaysAtBottom();
        const distance = D.distanceFromBottom();
        // EITHER answer is enough, and the geometry is not merely a fallback for a build with no
        // jump control. The control's `invisible` class is updated from a scroll LISTENER, and
        // scroll events are dispatched asynchronously, so a gesture that has just returned the
        // viewport to the end can reach this line while the class still says otherwise. Reading
        // the class alone loses the re-attachment to a race and quietly restores the old
        // never-reattach behaviour on exactly the fast gestures the harness performs.
        // `distanceFromBottom()` is computed from scrollTop and cannot be stale.
        if (app === true || (distance !== null && distance <= FOLLOW_TOLERANCE_PX)) {
          detached = false;
          F.reattachments += 1;
        } else {
          // Still away from the end. Re-stamp against the run in flight NOW, not the one that was
          // in flight when the gesture began: a gesture long enough to span a run boundary would
          // otherwise look like a scroll away from the previous run and be re-attached by the very
          // first sample of the current one.
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
        // The second half of the intent contract, as its own verdict.
        yanked_after_scroll: F.yanked_back_samples > 0,
        // null, not 1.0, when nothing was ever sampled mid-run. A cell whose stream finished
        // before the first tick has not demonstrated that the thread follows; it has
        // demonstrated nothing, and 1.0 would read as a pass.
        pinned_fraction: measured > 0 ? F.running_pinned / measured : null,
        pinned_fraction_reason:
          measured > 0 ? null : "no sample was taken while a reply was streaming",
        // HOW MUCH OF THE STREAM THIS VERDICT ACTUALLY COVERS. `pinned_fraction` is computed
        // over the attached phases only, so without this it can read 1.0 on a cell where the
        // thread was attached for three seconds of an eighteen-second stream. Reported beside it
        // so the one cannot be quoted without the other.
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
