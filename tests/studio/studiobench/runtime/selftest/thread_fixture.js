// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// A SYNTHETIC THREAD, with a switch on how it mounts.
//
// This exists so the readiness gate can be shown PASSING on a correctly virtualised thread and
// FAILING on a broken one, in a real browser, against the real `scene/dom.js` adapter and the real
// `runtime/readiness.py` probe. A gate that has only ever been shown passing is not a gate, and
// neither is one that has only ever been shown against a mock of itself.
//
// It reproduces exactly the contract the shipping app publishes and the readiness gate reads --
// `.aui-thread-root`, `.aui-thread-viewport`, `[data-role]`, `.aui-thread-scroll-to-bottom` with
// its `invisible` toggle, the composer textarea -- and nothing else. It is not a replica of Unsloth
// and must never be mistaken for one: it cannot tell you anything about the app's own timing. What
// it can do is put the gate in front of each of the situations it has to tell apart.

(() => {
  const MODES = [
    // Everything mounted, the way the app ships today.
    "full",
    // Mounting is still in progress: the first K messages, growing. This is the state the gate
    // exists to refuse, and the state the old count-based gate DID correctly refuse.
    "mounting",
    // A correct window: the tail of the thread, aria-setsize/aria-posinset published, anchored at
    // the bottom, and the head materialises when you scroll to the top.
    "windowed",
    // A window with no aria-setsize. Nothing outside the app can learn how long this thread is.
    "windowed_no_total",
    // A window over the HEAD of the thread rather than its end: settled, correct total, but the
    // last message is nowhere.
    "windowed_at_top",
    // A window at the end, correct total published, and a store that only holds what is mounted.
    // Indistinguishable from `windowed` while you stand at the bottom of the thread, which is why
    // the completeness probe has to walk to the top.
    "windowed_lost_head",
    // A store that kept the first page AND the last page and lost the middle. The head marker
    // arrives when you scroll to the top, so the marker check alone calls it complete; the
    // ordinals of one mounted window run 1,2,3 then 16,17,18, which is the loss.
    "windowed_lost_middle",
    // The same correct window, but with the ordinals written directly on the [data-role] element
    // instead of on a row wrapper. Both placements must be accepted.
    "windowed_flat",
    // THE THREE MALFORMED ORDINAL CONTRACTS. Each publishes aria-posinset on every mounted row --
    // enough for a gate that only counts attributes -- and each publishes numbers that are not
    // positions in the thread, so nothing outside the app can locate the mounted window.
    //
    // Every row says 0. The attribute is there and it is out of range: aria-posinset is 1-based.
    "windowed_zero_ordinals",
    // Every row says the same number, so six mounted rows claim one position between them.
    "windowed_duplicate_ordinals",
    // A window at the BOTTOM of an 18-message thread numbered 1..6: the index within the window
    // published as though it were the position in the thread. The most likely of the three to be
    // written by accident, and the one that makes a window at the end look like a window at the
    // start.
    "windowed_from_one",
  ];

  function marker(i) {
    // Must match runtime/seeder.turn_marker exactly.
    return "studiobench turn " + i + ": continue with unit " + i;
  }

  window.__fixture = {
    MODES,
    marker,

    build(opts) {
      const mode = opts.mode;
      const turns = opts.turns;          // user/assistant pairs
      const windowSize = opts.windowSize || 6;
      const total = turns * 2;
      document.body.innerHTML = "";

      const root = document.createElement("div");
      root.className = "aui-root aui-thread-root";
      const viewport = document.createElement("div");
      viewport.className = "aui-thread-viewport aui-stream-viewport";
      viewport.style.cssText = "height:400px;overflow-y:auto;position:relative;";
      root.appendChild(viewport);

      const topSpacer = document.createElement("div");
      const list = document.createElement("div");
      const bottomSpacer = document.createElement("div");
      topSpacer.setAttribute("aria-hidden", "true");
      bottomSpacer.setAttribute("aria-hidden", "true");
      viewport.appendChild(topSpacer);
      viewport.appendChild(list);
      viewport.appendChild(bottomSpacer);

      const jump = document.createElement("button");
      jump.className = "aui-thread-scroll-to-bottom";
      root.appendChild(jump);

      const composer = document.createElement("textarea");
      composer.setAttribute("aria-label", "Message input");
      root.appendChild(composer);
      document.body.appendChild(root);

      // Every message the "store" holds. `windowed_lost_head` throws most of them away, which is
      // the data loss the completeness probe is looking for.
      const ROW_PX = 120;
      let store = [];
      for (let i = 0; i < turns; i += 1) {
        store.push({ role: "user", text: marker(i), pos: i * 2 + 1 });
        store.push({ role: "assistant", text: "reply to turn " + i + " ".repeat(200), pos: i * 2 + 2 });
      }
      const declaredTotal = total;
      if (mode === "windowed_lost_head") store = store.slice(-windowSize);
      // THE HEAD AND THE TAIL, AND NOTHING BETWEEN THEM. Half a window at each end, so the store
      // is exactly one window long and every scroll position mounts all of it: the first message
      // is always there for the marker check, and the hole is always there for the ordinals.
      if (mode === "windowed_lost_middle") {
        const keep = Math.max(1, Math.floor(windowSize / 2));
        store = store.slice(0, keep).concat(store.slice(-keep));
      }

      const state = { mode, store, declaredTotal, windowSize, ROW_PX, total };

      // WHAT EACH ROW PUBLISHES AS ITS POSITION. Everything but the three malformed modes
      // publishes the message's real position in the thread, which is what aria-posinset means.
      // The malformed ones publish a number that is not one, each in a different way, so the gate
      // can be shown refusing each of them separately rather than refusing "something about the
      // ordinals".
      function publishedPos(message, indexInWindow) {
        if (mode === "windowed_zero_ordinals") return 0;
        if (mode === "windowed_duplicate_ordinals") return state.declaredTotal;
        if (mode === "windowed_from_one") return indexInWindow + 1;
        return message.pos;
      }

      function render(startIndex, count, publishTotals) {
        list.innerHTML = "";
        const slice = state.store.slice(startIndex, startIndex + count);
        for (let i = 0; i < slice.length; i += 1) {
          const m = slice[i];
          const pos = publishedPos(m, i);
          const el = document.createElement("div");
          el.setAttribute("data-role", m.role);
          el.style.cssText = "height:" + ROW_PX + "px;overflow:hidden;";
          el.textContent = m.text;
          // WHERE THE SHIPPING VIRTUALIZER PUTS THE ORDINALS: on the positioned row wrapper, not
          // on the message. `thread-message-virtualizer.tsx` renders an absolutely positioned div
          // per item and mounts `ThreadPrimitive.MessageByIndex` inside it, so the element that is
          // a member of the set is the wrapper. `wrapped` reproduces that shape exactly; the
          // unwrapped placement is kept too, because the gate must accept both.
          if (publishTotals && mode !== "windowed_flat") {
            const row = document.createElement("div");
            row.setAttribute("aria-setsize", String(state.declaredTotal));
            row.setAttribute("aria-posinset", String(pos));
            row.appendChild(el);
            list.appendChild(row);
            continue;
          }
          if (publishTotals) {
            el.setAttribute("aria-setsize", String(state.declaredTotal));
            el.setAttribute("aria-posinset", String(pos));
          }
          list.appendChild(el);
        }
        // Spacers, so the scroll extent describes the WHOLE thread even though only a window is
        // mounted. A virtualizer that omits these has a scrollbar that lies, which is one of the
        // behavioural invariants in analysis/behaviour.py.
        const above = mode === "full" || mode === "mounting" ? 0 : startIndex * ROW_PX;
        const below =
          mode === "full" || mode === "mounting"
            ? 0
            : Math.max(0, (state.declaredTotal - startIndex - slice.length) * ROW_PX);
        topSpacer.style.height = above + "px";
        bottomSpacer.style.height = below + "px";
      }

      function pin() {
        viewport.scrollTop = viewport.scrollHeight;
        jump.classList.add("invisible");
      }

      state.renderWindowAround = (scrollTop) => {
        // A store SHORTER than the thread it claims cannot be indexed by the thread's own
        // indices: `windowed_lost_middle` holds one window's worth of messages for an
        // eighteen-message thread, so every scroll position mounts the same rows and the ordinals
        // on them are the only thing that says which messages they are.
        if (mode === "windowed_lost_middle") {
          const start = Math.max(
            0,
            Math.min(state.store.length - windowSize, Math.floor(scrollTop / ROW_PX)),
          );
          render(start, windowSize, true);
          return;
        }
        const first = Math.max(
          0,
          Math.min(state.declaredTotal - windowSize, Math.floor(scrollTop / ROW_PX)),
        );
        // `windowed_lost_head` only has the tail in its store, so an index into the full thread
        // has to be translated. It will simply have nothing to show near the top.
        const offset = mode === "windowed_lost_head" ? state.declaredTotal - state.store.length : 0;
        render(Math.max(0, first - offset), windowSize, true);
      };

      if (mode === "full") {
        render(0, state.store.length, false);
        pin();
      } else if (mode === "mounting") {
        // GROWING. One message every 250ms, from the head, and it never finishes inside the
        // window any test here gives it. The count climbs, the element count climbs, the scroll
        // height climbs, and the last message is never reached.
        let n = 1;
        render(0, n, false);
        pin();
        state.timer = setInterval(() => {
          n += 1;
          if (n > state.store.length) { clearInterval(state.timer); return; }
          render(0, n, false);
          pin();
        }, 250);
      } else if (mode === "windowed_at_top") {
        render(0, windowSize, true);
        viewport.scrollTop = 0;
        jump.classList.remove("invisible");
      } else if (mode === "windowed_no_total") {
        render(state.store.length - windowSize, windowSize, false);
        pin();
      } else {
        // windowed, windowed_flat, windowed_lost_head, windowed_lost_middle and the three
        // malformed-ordinal modes: all of them are a window at the END of the thread, and they
        // differ only in what they publish about it.
        render(Math.max(0, state.store.length - windowSize), windowSize, true);
        pin();
        viewport.addEventListener("scroll", () => {
          state.renderWindowAround(viewport.scrollTop);
          jump.classList.toggle(
            "invisible",
            viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop < 8,
          );
        });
      }
      // COPY-FROM-STORE, the shipping fix, reproduced on the same contract.
      //
      // Mirrors `decideThreadCopy` in
      // studio/frontend/src/components/assistant-ui/thread-copy-from-store.ts: intervene only when
      // the selection spans the whole mounted list AND the store holds more than is mounted. The
      // unit tests over there stub `containsNode`; this is the only place the REAL Selection
      // semantics are exercised, which is the part that could actually be wrong.
      if (opts.copyFromStore) {
        viewport.addEventListener("copy", (event) => {
          const selection = window.getSelection();
          const nodes = Array.from(viewport.querySelectorAll("[aria-posinset]"));
          if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
          if (nodes.length === 0 || state.store.length <= nodes.length) return;
          const first = nodes[0];
          const last = nodes[nodes.length - 1];
          if (!selection.containsNode(first, true) || !selection.containsNode(last, true)) return;
          const text = state.store.map((m) => "## " + m.role + "\n\n" + m.text).join("\n\n");
          event.clipboardData.setData("text/plain", text);
          event.preventDefault();
        });
      }

      window.__fixtureState = state;
      return { mode, total, mounted: list.children.length };
    },
  };

  // The two-rAF paint promise the real harness installs from instruments/frames.js. The
  // completeness probe awaits it.
  if (!window.__sbNextPaint) {
    window.__sbNextPaint = () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now()))),
      );
  }
})();
