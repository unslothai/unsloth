// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// A synthetic thread with switchable mount modes, so the readiness gate can be shown passing and
// failing in a real browser. It mirrors only the app's DOM contract, not its timing.

(() => {
  const MODES = [
    "full",
    "mounting",
    "windowed",
    "windowed_no_total",
    "windowed_at_top",
    // Looks like `windowed` at the bottom; only the completeness probe's walk to the top catches it.
    "windowed_lost_head",
    // Keeps first and last page only; the marker check passes, the ordinals reveal the gap.
    "windowed_lost_middle",
    "windowed_flat",
    // Malformed ordinal contracts. Zero: aria-posinset is 1-based.
    "windowed_zero_ordinals",
    "windowed_duplicate_ordinals",
    // Window index published as the thread position, the likeliest accidental bug.
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
      const turns = opts.turns;
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

      const ROW_PX = 120;
      let store = [];
      for (let i = 0; i < turns; i += 1) {
        store.push({ role: "user", text: marker(i), pos: i * 2 + 1 });
        store.push({ role: "assistant", text: "reply to turn " + i + " ".repeat(200), pos: i * 2 + 2 });
      }
      const declaredTotal = total;
      if (mode === "windowed_lost_head") store = store.slice(-windowSize);
      if (mode === "windowed_lost_middle") {
        const keep = Math.max(1, Math.floor(windowSize / 2));
        store = store.slice(0, keep).concat(store.slice(-keep));
      }

      const state = { mode, store, declaredTotal, windowSize, ROW_PX, total };

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
          // The shipping virtualizer puts ordinals on the row wrapper; the gate must accept both placements.
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
        // Spacers keep the scroll extent equal to the whole thread.
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
        const offset = mode === "windowed_lost_head" ? state.declaredTotal - state.store.length : 0;
        render(Math.max(0, first - offset), windowSize, true);
      };

      if (mode === "full") {
        render(0, state.store.length, false);
        pin();
      } else if (mode === "mounting") {
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
      // Mirrors decideThreadCopy in thread-copy-from-store.ts with real Selection semantics.
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

  if (!window.__sbNextPaint) {
    window.__sbNextPaint = () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now()))),
      );
  }
})();
