// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// The selector adapter for the REAL Unsloth chat UI: the salvaged actions call the smoke
// fixture's `window.__heavyThread` API and are ported onto this instead. Same method names,
// the app's own selectors, one file to edit when a class changes.
// Salvaged from playwright_heavy_thread.py.
// The chat thread has essentially no test ids (exactly one, `composer-tool-status`), so the
// contract is class hooks (`aui-*`, `unsloth-*`), `data-role`, `data-slot` and accessible
// names read out of the shipped TSX.
// Three selectors carry a trap, handled here rather than per action: the composer exists TWICE
// in compare mode (so everything scopes to the first thread root); the Stop button is REPLACED
// by Queue when the composer has text, since `queueDisabled` follows `composerText`, so
// pressing stop with text measures nothing; and Radix keeps a collapsed Collapsible's content
// mounted for its animation, so "open" is read from `data-state` on the root.

(() => {
  if (window.__sb && window.__sb.dom) return;
  window.__sb = window.__sb || {};

  const q = (sel, root) => (root || document).querySelector(sel);
  const qa = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // Accessible name for the app's TooltipIconButton, which renders the label in a visually
  // hidden span rather than an aria-label. `getByRole(name:)` would do this driver-side, but
  // the actions run inside one page.evaluate.
  const nameOf = (el) => {
    if (!el) return "";
    const aria = el.getAttribute("aria-label");
    if (aria) return aria.trim();
    const sr = el.querySelector(".aui-sr-only, .sr-only");
    if (sr && sr.textContent) return sr.textContent.trim();
    return (el.textContent || "").trim();
  };

  // The two hooks the app publishes a part's progress through: assistant-ui's `data-status` on a
  // text part (markdown-text.tsx) and `aria-busy` on the reasoning content (reasoning.tsx).
  // Named once so the streaming scan and its control cannot drift apart.
  // Every control ComposerRightControls can put in the run-state slot. The research pair is
  // included because it is the same slot, even though nothing here can reach a research run.
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
    // THE DISPATCHED HALF OF A QUEUE RUN: `ComposerRightControls` renders this under
    // `isQueueRunning && !thread.isRunning`, so the thread reports itself NOT running and neither
    // `stopButton()` nor `queueButton()` matches. Read on its own that transient looked like a
    // settled Send arm, giving a differing composer with no run-state difference to explain it:
    // the shape the comparison layer treats as a rendering regression.
    stopQueuedButton() {
      return q('button[aria-label="Stop queued message"]');
    },
    isRunning() {
      return Boolean(D.stopButton() || D.queueButton());
    },
    // WHICH RUN-STATE CONTROL THE COMPOSER IS RENDERING, as a token. `ComposerRightControls`
    // renders exactly one at a time as a function of the run state, and they are DIFFERENT
    // SUBTREES inside `.aui-thread-root`, so `digest_scaffold` carries whichever is up and two
    // arms showing different tokens differ in the scaffold for that reason. `isRunning()` is too
    // coarse to say so, being true for Stop AND Queue.
    runStateControl() {
      for (const name of RUN_STATE_CONTROLS) {
        if (byName("button", name)) return name;
      }
      return "";
    },
    // THE PROMPT QUEUE'S OWN SURFACE: `PromptQueueStack` renders waiting prompts inside the
    // composer root with the accessible name "Prompt queue, <n> of <m>", the only place the queue
    // states its existence in the DOM. Scoped to the thread root so another thread's queue is not
    // read as this one's.
    promptQueue() {
      return q('[aria-label^="Prompt queue,"]', D.threadRoot());
    },
    // A REPLY IS ACTUALLY BEING WRITTEN, which is NOT what `isRunning()` answers. `isRunning()`
    // must accept the Queue button, since with text in the composer a running thread renders no
    // Stop button, and every "may I send now" caller wants that broad reading. It is wrong for a
    // positive control on the streaming probe, because "Queue message" is also rendered under
    // `isQueueRunning && !thread.isRunning` while nothing is generating, and that queued-idle
    // interval needs the opposite treatment from a live stream whose hooks have gone quiet. The
    // queued-idle button always comes with the queue surface, since
    // `getPromptQueueUIItemsForRun` drops only DISPATCHED items. WHAT THIS GIVES UP (pinned in
    // test_studiobench_queued_idle_live.py): with a queue run, a streaming reply and text in the
    // composer this reads false and the control is not armed. Under-claiming, and cheap, since
    // probe blindness is a renamed selector and therefore global.
    // `syncPromptQueueUI` marks dispatched items.
    // A RESEARCH RUN IS A GENERATION THIS PREDICATE CANNOT SEE, deliberately: "Stop research"
    // matches neither `stopButton()` nor `queueButton()`, and a research report renders through
    // `MarkdownPreview` rather than the assistant text part, so `streamingMessages()` finds no
    // `data-status` either. Not patched, because nothing here can start one: `ResearchMessage` is
    // gated on `message.metadata` and the seeder writes `None`. If a research scene is added, move
    // all three in one change and add the report's busy hook to `STATUS_HOOK`.
    // The state is `isResearchActive`.
    generating() {
      if (D.stopButton()) return true;
      if (!D.queueButton()) return false;
      return !D.promptQueue();
    },
    // WHICH MESSAGES ARE STILL BEING WRITTEN, read from the app's own published state:
    // markdown-text.tsx renders `<div data-status={status.type}>`, and the reasoning pane
    // publishes the same fact as `aria-busy` on `[data-slot="reasoning-content"]`, a separate part
    // that can be running while the answer is not. Both are the APP's statements about its own
    // state; reading a timer, a character count or "the last assistant message" is how you
    // attribute a stream to the wrong message on the arm that renders faster.
    streamingMessages() {
      return qa("[data-role]").filter(
        (m) => m.querySelector('[data-status="running"], [aria-busy="true"]') !== null,
      );
    },
    // IS THE STREAMING PROBE BLIND, OR IS THERE NOTHING TO SEE? `streamingMessages()` returning
    // nothing has three causes worth telling apart. THE HOOK IS GONE: a build renamed
    // `data-status`, which is rendered for `complete` parts too, so on a working build every
    // assistant message that rendered a part carries it. THE ROW IS NOT MOUNTED: a windowed arm
    // scrolled away from the tail, which is what windowing is for. THE ROW HAS NO PARTS YET:
    // between the send being accepted and the first part arriving, thread.tsx renders
    // "Generating...", and `send_turn` returns the instant `isRunning()` flips, so a capture lands
    // here twice a film. Scoped to ASSISTANT messages, since only assistant parts render through
    // `MarkdownText`.
    statusHookPresent() {
      return D.assistantMessages().some((m) => m.querySelector(STATUS_HOOK) !== null);
    },
    // Whether the message a reply would be written INTO is publishing parts this probe can read;
    // false means it has none yet, the third case above and not a broken instrument.
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
    // HOW LONG THE THREAD IS, as opposed to how much is mounted. Identical to messageCount() on
    // the shipped build, and different the moment an arm mounts a window, at which point
    // send_turn's "the thread grew", delete's "the count dropped" and thread_reopen's "same
    // messages" ask about the THREAD and are answered about the window, all in the same direction,
    // because the window refills as fast as it empties. aria-setsize is where WAI-ARIA already
    // requires a windowed list to publish this, so it is the accessible name for the quantity
    // rather than a private channel.
    threadTotal() {
      // On the message, or on the row wrapper a virtualizer positions it in. Same walk as
      // runtime/readiness.py: the ordinal belongs on the element that is a member of the set.
      const first = q("[data-role]");
      const owner = first ? first.closest("[aria-setsize]") : q("[aria-setsize]");
      if (owner) {
        const n = Number(owner.getAttribute("aria-setsize"));
        if (Number.isFinite(n) && n >= 0) return n;
      }
      return qa("[data-role]").length;
    },
    // True when the thread publishes a total larger than what it has mounted, i.e. a windowed
    // mount and not merely a short thread.
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

    // The jump-to-bottom control, which thread.tsx renders permanently and hides with `invisible`
    // when the intent-aware autoscroll reports itself at the bottom. Reading the app's own state
    // means the harness and the app cannot disagree about whether the thread is pinned.
    jumpToBottomButton() {
      return q(".aui-thread-scroll-to-bottom");
    },
    appSaysAtBottom() {
      const jump = D.jumpToBottomButton();
      // `null`, NOT `false`, when the control is absent: a build that does not render it has told
      // us nothing, and the two must not be summed.
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
      // data-state on the ROOT, not the presence of the content element: Radix keeps collapsed
      // content mounted for the animation, so a presence check reads every pane as open.
      return qa('[data-slot="reasoning-root"][data-state="open"]').length;
    },
    // STILL MOUNTED IS NOT STILL OPEN. `reasoningOpenCount` flips on the click, but the CHILDREN
    // outlive it on both collapse mechanisms by design: Radix's `Presence` suspends the unmount
    // until `animationend`, and the grid arm renders `present && children` until `transitionend`
    // or its 250 ms backstop. For that window every pane is closed while every span it contributed
    // is still in the document, and a census asked whether it has stopped moving answers yes
    // because it has not started. So a collapse is settled when the content is GONE, which one
    // selector covers on both arms.
    // The grid arm is `UnmeasuredCollapsibleContent`.
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
    // Hover the last assistant message, which is what mounts its action bar: `autohide` unmounts
    // it on every message that is not hovered, so a control read without this is read out of a
    // tree it was never in.
    hoverLastAssistantMessage() {
      const m = D.lastAssistantMessage();
      if (m) {
        m.dispatchEvent(
          new PointerEvent("pointerover", { bubbles: true, pointerType: "mouse" })
        );
      }
      return m;
    },

    // WAIT for one of the action bar's controls, up to `waitMs`, instead of sampling once. The bar
    // is mounted with `hideWhenRunning`, so while the thread generates there is no Copy, Delete or
    // More anywhere: not hidden, absent. Every action needing one is scheduled after a `send_turn`
    // on the NOMINAL drain arithmetic, which assumes the pacer is the binding constraint; at 100K
    // the renderer is, so the reply arrives about 25% later. On the CI run that failed the
    // liveness gate the `message_menu` window opened at 32,000ms, took one more SSE chunk inside
    // itself, and the reply stopped growing 71 characters later inside the same window: a single
    // sample turns that third of a second into `NOT RUN -- no More button`. The wait is bounded,
    // reported, and happens BEFORE any measurement clock starts. Polling per paint and scoped to
    // the last assistant message, so it is O(that message) and stops as soon as the control
    // appears.
    // The drain arithmetic is FOLLOW_UP_CHARS over the field cadence; see test_studiobench_rung_plan.py.
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
        // Re-hovered every pass: the bar unmounts again whenever the message re-renders, which during
        // a stream is on every chunk.
        D.hoverLastAssistantMessage();
        el = D.actionButton(name);
      }
      return {
        el,
        waitedMs: Math.round((performance.now() - started) * 10) / 10,
        // Recorded whether the control was found or not: a miss with `running: true` is the reply not
        // having settled, a miss with `running: false` is a control that is genuinely not there.
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
      // No role="option", no data-model-id: the rows are plain buttons with utility classes, so
      // this is the only available handle and it is recorded as the weak point it is.
      return menu ? qa("button", menu) : [];
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
        // Shiki spans. THE span density check: the field capture ran 90,262 characters against 16,186
        // spans, 5.6 characters per span, and a fixture that does not reproduce that is not measuring
        // the same highlighter load per character.
        highlight_spans: qa("pre span").length,
        // WHERE the spans live, not just how many: a collapsed reasoning pane UNMOUNTS its children,
        // so a thread with the same text can carry wildly different DOM. Without the split, "seeded
        // has 20% fewer spans" has three possible explanations.
        // Tool components, TWO markers because there are two renderers: a known tool gets a
        // `tool-group-root` and anything else a generic `tool-fallback-root`. Counting only the first
        // read ZERO on a thread that visibly contained tool blocks.
        tool_groups: qa('[data-slot="tool-group-root"]').length
                   + qa('[data-slot="tool-fallback-root"]').length,
        tool_groups_open: qa('[data-slot="tool-group-content"]').length
                        + qa('[data-slot="tool-fallback-content"]').length,
        reasoning_spans: qa('[data-slot="reasoning-root"] pre span').length,
        reasoning_code_blocks: qa('[data-slot="reasoning-root"] pre').length,
        content_spans:
          qa("pre span").length - qa('[data-slot="reasoning-root"] pre span').length,
        content_code_blocks: qa("pre").length - qa('[data-slot="reasoning-root"] pre').length,
        // Carried in the census so the peak occupancy and the character count come from the SAME
        // reading; two reads either side of a destructive action disagree.
        assistant_chars: D.assistantChars(),
        viewport_scroll_height: (D.viewport() || {}).scrollHeight || null,
        viewport_client_height: (D.viewport() || {}).clientHeight || null,
        // DOES THE THREAD STILL FOLLOW THE STREAM? Three readings taken with every census, so the
        // answer exists for every window rather than being reconstructed from timings. If the thread
        // stops following, the streamed message drifts out of the viewport, a windowed list UNMOUNTS
        // it, and the streaming cost collapses to almost nothing: a beautiful frame rate that measures
        // not rendering the thing being measured. `app_at_bottom` is the app's OWN state (thread.tsx
        // hides the scroll-to-bottom control with `invisible` exactly when use-intent-aware-autoscroll
        // considers itself at the bottom); `distance_from_bottom` sits alongside because a virtualizer
        // working from estimated row heights can be a few pixels off while correctly pinned.
        viewport_scroll_top: (D.viewport() || {}).scrollTop || null,
        distance_from_bottom: D.distanceFromBottom(),
        app_at_bottom: D.appSaysAtBottom(),
      };
      out.census_cost_ms = Math.round((performance.now() - started) * 100) / 100;
      return out;
    },

    // Characters of assistant text currently in the DOM, for the seeded-vs-streamed equivalence
    // check and chars-per-span.
    assistantChars() {
      let n = 0;
      for (const m of qa('[data-role="assistant"]')) n += (m.textContent || "").length;
      return n;
    },
  };

  window.__sb.dom = D;

  // WHY THIS IS NOT DONE FROM THE DRIVER: the stream runs during the gap windows, whose whole
  // purpose is to observe the page doing nothing but stream, and a `page.evaluate` per sample
  // would put a CDP round trip and a forced style read inside them four times a second. So it
  // samples in the page and is READ ONCE PER CELL, outside every window: a 250ms timer, two
  // orders of magnitude below the 1ms timer frames.js documents as free, doing no layout it has
  // not already caused. `pinned_fraction` is what makes an fps number from a windowed arm
  // readable at all.
  // That timer runs at ~150 ticks a second.
  // THE COUNTERS SURVIVE A NAVIGATION, via sessionStorage. They did not, and the symptom was a
  // confident "NOT MEASURED" on the arm that behaved best: the film ends with `thread_reopen`,
  // whose `page.goto` fallback destroys the JS context and re-runs the init scripts, so an
  // in-memory sampler came back at zero while the treatment arm, whose thread_reopen did not
  // run, kept its counters and looked like the only arm with data. sessionStorage is per-origin
  // and per-tab and outlives a same-origin navigation, exactly the lifetime wanted. Saved on
  // pagehide rather than per tick.
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
    // Set once the thread is seen to fall behind while a run is in progress, and never cleared: a
    // thread that drifts away and is later yanked back has still failed the contract, and an
    // end-of-cell reading would show it pinned.
    ever_fell_behind: false,
  }, restore() || {});
  window.addEventListener("pagehide", () => {
    try {
      window.sessionStorage.setItem(FOLLOW_KEY, JSON.stringify(F));
    } catch (e) {
      // A full sessionStorage is not worth losing the page over; the reading degrades to "not
      // measured", which is already handled.
    }
  });
  const FOLLOW_TICK_MS = 250;
  // How far from the bottom still counts as following. Generous on purpose, since a virtualizer
  // working from estimated heights can sit short of the exact bottom while behaving perfectly;
  // 64px is under two lines, so it cannot hide a thread that has stopped following.
  const FOLLOW_TOLERANCE_PX = 64;
  // TWO PHASES, BECAUSE THE INTENT CONTRACT HAS TWO HALVES (plans/proud-wiggling-falcon.md):
  // autoscroll follows a stream, AND a user who has scrolled up is never yanked down. One number
  // cannot score both: the first version reported 47-50% pinned on both arms with an identical
  // 6,615px worst drift, which is the film, since `scroll_during_generation` drags the viewport
  // thousands of pixels up twice and the app correctly declines to drag it back, so the sampler
  // scored the second half of the contract as a failure of the first. ATTACHED (before the
  // harness scrolls) is "does it follow" and is what the gate scores; DETACHED (after a
  // deliberate scroll) is "it must not come back on its own", recorded as its own finding.
  let detached = false;
  let suspended = 0;
  // WHICH RUN THE USER SCROLLED AWAY FROM, the difference between a yank and a return.
  // `resume()` clears `detached` only when the gesture ended at the bottom, which on the real
  // films it never does: `SCROLL_JS` steps 14 x 420px away, so above 5,880px `detached` latches
  // for the rest of the cell, and the film then starts two more runs whose intended pinning was
  // counted as a yank. Measured at head across every 100K payload: attached_fraction 0.07 to
  // 0.15 with zero reattachments, on the BASE arm and on pure null controls, so
  // `follows_the_stream` failed every 100K cell of every run and passed only on the 1K film. So
  // a run the user STARTED is a fresh expression of intent to be at the end: re-attachment is
  // granted only when a run that began AFTER the gesture is also observed at the bottom.
  let runSeq = 0;
  let wasRunning = false;
  let detachedAtRun = 0;
  setInterval(() => {
    F.samples += 1;
    // Read BEFORE the suspended early-return, or a run that starts and ends inside a deliberate
    // gesture is never seen and the run after it is mistaken for the one scrolled away from.
    const running = D.isRunning();
    if (!running) wasRunning = false;
    else if (!wasRunning) { wasRunning = true; runSeq += 1; }
    if (suspended > 0) { F.suspended_samples += 1; return; }
    if (!running) return;
    const app = D.appSaysAtBottom();
    const distance = D.distanceFromBottom();
    // "At the bottom" by the app's own answer, with geometry standing in only when the build
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
    // The FIRST suspend also latches `detached`: from then the user has expressed an intent to be
    // somewhere other than the bottom, and everything after is scored against the second half of
    // the contract. `detachedAtRun` is stamped on every suspend rather than only the first, so a
    // second gesture during a later run detaches from THAT run.
    suspend() { suspended += 1; detached = true; detachedAtRun = runSeq; },
    resume() {
      suspended = Math.max(0, suspended - 1);
      // RE-ATTACH IF THE GESTURE LEFT US AT THE END, and this is not a nicety. `detached` used to
      // latch on the first suspend and never clear, so from the harness's first deliberate scroll
      // every sample went to the detached branch: in the shipped film that scroll is 1.5s into an 18s
      // opening stream, so the verdict came from the first ~3s and covered 13% of the streaming time
      // while reporting "the thread follows the stream" (running_samples 11, detached_samples 72).
      // The contract is about INTENT, and intent is re-expressed by coming back, exactly as
      // Unsloth's own intent-aware autoscroll implements. Only checked on the way out of a
      // deliberate gesture, so the app pulling the viewport down on its own is still a yank.
      if (suspended === 0 && detached) {
        const app = D.appSaysAtBottom();
        const distance = D.distanceFromBottom();
        // EITHER answer is enough, and the geometry is not merely a fallback: the control's
        // `invisible` class is updated from a scroll LISTENER and scroll events are dispatched
        // asynchronously, so a gesture that has just returned the viewport to the end can reach this
        // line while the class still says otherwise. `distanceFromBottom()` cannot be stale.
        // `distanceFromBottom()` is computed from scrollTop.
        if (app === true || (distance !== null && distance <= FOLLOW_TOLERANCE_PX)) {
          detached = false;
          F.reattachments += 1;
        } else {
          // Still away from the end. Re-stamp against the run in flight NOW, not the one in flight when
          // the gesture began: a gesture spanning a run boundary would otherwise be re-attached by the
          // very first sample of the current run.
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
        // null, not 1.0, when nothing was sampled mid-run: a cell whose stream finished before the
        // first tick has demonstrated nothing, and 1.0 would read as a pass.
        pinned_fraction: measured > 0 ? F.running_pinned / measured : null,
        pinned_fraction_reason:
          measured > 0 ? null : "no sample was taken while a reply was streaming",
        // HOW MUCH OF THE STREAM THIS VERDICT COVERS: `pinned_fraction` is computed over the attached
        // phases only, so without this it can read 1.0 on a cell attached for three seconds of an
        // eighteen-second stream.
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
