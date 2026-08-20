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
// its `invisible` toggle, the composer textarea -- and nothing else. It is not a replica of Studio
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
    // The same correct window, but with the ordinals written directly on the [data-role] element
    // instead of on a row wrapper. Both placements must be accepted.
    "windowed_flat",
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

      const state = { mode, store, declaredTotal, windowSize, ROW_PX, total };

      function render(startIndex, count, publishTotals) {
        list.innerHTML = "";
        const slice = state.store.slice(startIndex, startIndex + count);
        for (const m of slice) {
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
            row.setAttribute("aria-posinset", String(m.pos));
            row.appendChild(el);
            list.appendChild(row);
            continue;
          }
          if (publishTotals) {
            el.setAttribute("aria-setsize", String(state.declaredTotal));
            el.setAttribute("aria-posinset", String(m.pos));
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
        // windowed, windowed_flat and windowed_lost_head
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
