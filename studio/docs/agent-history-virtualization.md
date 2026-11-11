# Studio agent-history virtualization — Architecture & implementation

> Implementation notes for reviewers of the opt-in Studio chat-history feature. This document travels with the feature branch and is intended to be linked from the pull request. It describes code responsibilities, not a claim of production approval.

### Summary

Add an **opt-in, bounded DOM renderer for long Studio chat histories** while retaining the existing assistant-ui message/runtime/persistence model. The feature virtualizes *mounted React/DOM rows*, **not the stored conversation**, and is designed to preserve reader position during streaming, scroll gestures and dynamic tool-card layout.

- **Off (default):** use the existing `ProgressiveMessages` renderer.
- **Dynamic:** bound the mounted range using measured message-descendant DOM cost and conservative estimates for unseen rows.
- **Fixed:** use a row-bounded range; the user-facing Fixed mode adapts coverage to the viewport under an estimated DOM budget.
- **Overrides:** a global preference plus a per-chat override with an inherit option; older stored preferences remain readable.

### Architecture / change map (actual module and function boundaries)

| Where | What changed | Logic / architectural role |
| --- | --- | --- |
| [`agent-history-config.ts`](../frontend/src/components/assistant-ui/agent-history-config.ts) | Added pure mode, row-window and DOM-target configuration. | Lightweight shared constants; avoids pulling the renderer and indices into the eager settings/chat entry points. |
| [`agent-history-mode-store.ts`](../frontend/src/features/chat/stores/agent-history-mode-store.ts), [`agent-history-preference-policy.ts`](../frontend/src/features/chat/stores/agent-history-preference-policy.ts) | Added Zustand-persisted global/per-thread preferences and `agentHistoryLevelFor`, `agentHistoryWindowFor` selectors. | **One effective-mode decision** for controls and renderer, per-chat overrides first, legacy preference migration/normalization, browser-local settings only (message storage unchanged). |
| [`thread.tsx`](../frontend/src/components/assistant-ui/thread.tsx) | Added lazy `AgentHistoryScrollMessages` loading and a conditional renderer in `Thread`. | Reuses the existing message render slot, `Thread` viewport and assistant-ui runtime. `Off` remains on `ProgressiveMessages`; enabled threads mount the alternative via `Suspense`. No backend/API change. |
| [`agent-history-scroll-window.ts`](../frontend/src/components/assistant-ui/agent-history-scroll-window.ts) | Added `agentWindowAtIndex`, `agentWindowOnAppend`, `nextProgressiveAgentWindow`, `AgentHeightIndex` and viewport-recovery helpers. | Maintains a **logical transcript range** and sparse measured-height corrections, with ordinary-flow rows and top/bottom spacer elements rather than absolute-positioned messages. Staged seeks admit a small number of previously unmounted rows per frame. |
| [`agent-history-dom-budget.ts`](../frontend/src/components/assistant-ui/agent-history-dom-budget.ts) | Added `AgentDomCostIndex`, `agentBudgetWindow`, `agentPhysicalNeighborWindow` and rebalance/prefetch decisions. | Tracks descendant costs **inside message rows**, conservatively charges unknown rows, controls adjacent admission and avoids idle recentering. A single mandatory very large row can exceed a target: the target is not an unconditional global-DOM guarantee. |
| [`agent-history-scroll-messages.tsx`](../frontend/src/components/assistant-ui/agent-history-scroll-messages.tsx) | Added the opt-in renderer, range/staging lifecycle, measured spacers, row wrappers, ResizeObserver/MutationObserver accounting and scoped assistant-ui providers. | Only the active window is mounted; all messages remain in the runtime. `MessageByIndexProvider` preserves existing message rendering, and the notification gate prunes subscriptions for unmounted rows. Thread switches reset gesture and navigation state. |
| [`use-intent-aware-autoscroll.tsx`](../frontend/src/components/assistant-ui/use-intent-aware-autoscroll.tsx), [`agent-history-navigation.ts`](../frontend/src/components/assistant-ui/agent-history-navigation.ts) | Integrated viewport navigation, anchor compensation, bottom-follow detachment and reader/search intent epochs. | The shared intent-aware viewport controller owns deliberate scroll-position writes; passive layout/scroll-anchor callbacks must not override newer reader/search navigation. Boundary reconciliation is scoped to recent real input. |
| [`agent-history-native-thumb.ts`](../frontend/src/components/assistant-ui/agent-history-native-thumb.ts) | Added pure native-thumb start, live-target and release decisions. | Snapshots native track geometry for the drag, retains a stable virtual extent and resolves the final logical position. Exact top/bottom releases remain explicit; at interior release the browser's physical position wins when valid. The helper does **not** independently write `scrollTop`. |
| [`agent-history-search.ts`](../frontend/src/components/assistant-ui/agent-history-search.ts), [`find-targets.ts`](../frontend/src/features/find-in-page/lib/find-targets.ts), [`find-bar.tsx`](../frontend/src/features/find-in-page/components/find-bar.tsx) | Added logical-transcript text matching, virtual-row reveal and explicit `chat` target scope. | Find navigates text in messages even if their DOM rows are unmounted; epoch-based reveal prevents stale scroll callbacks from winning. The general find bar retains other external targets. |
| [`row-notification-gate.ts`](../frontend/src/components/assistant-ui/row-notification-gate.ts) | Added scoped row-notification cleanup. | Bound subscriptions to the mounted window instead of retaining observers for previously visited message rows. |
| [`chat-header-menu.tsx`](../frontend/src/features/chat/components/chat-header-menu.tsx), [`chat-tab.tsx`](../frontend/src/features/settings/tabs/chat-tab.tsx), settings and locale files | Added chat/global Off–Dynamic–Fixed controls and translations in 14 locales. | UI writes to the same canonical preference store, not a separate renderer-local settings state. |
| [`check-bundle-budget.ts`](../frontend/scripts/check-bundle-budget.ts) | Recalibrated the startup JS ceiling after building the integrated source. | Fork PR #38 build measured 106 eager chunks, 6,644.2 KiB raw / 2,004.3 KiB compressed, exceeding the previous baseline threshold; the full-distribution 75 MiB gate passed. This refresh updates the startup threshold to 6,820,000 B raw / 2,065,000 B compressed with narrow measured headroom, subject to a fresh CI build. The virtualizer remains lazy-loaded; the eager change consists of preferences and UI controls. |
| [`studio-agent-history-ci.yml`](../../.github/workflows/studio-agent-history-ci.yml) and focused TS/Playwright tests | Added regression coverage and wired five standalone browser drivers into regular CI. | PR runs cover UI modes, native thumb, DOM sweep, idle scroll and observational typing cost; a manually dispatched full matrix also includes 50k/200k/400k DOM-churn and retention cases. |

### Data/control flow

```text
Global + per-thread settings
          |
  Zustand preference store
          |
 agentHistoryLevelFor()       (single effective-mode selector)
          |
    Thread / assistant-ui runtime
          |
     Off? ---- yes ---> existing ProgressiveMessages
          |
          no
          v
  lazy AgentHistoryScrollMessages
     |           |             |
 logical rows   DOM costs   reader/search/thumb intent
 & heights      & budget    + autoscroll context
     |           |             |
     +------ range / staged seek
                    |
          mounted assistant-ui message rows
          + top/bottom virtual spacers
                    |
               real viewport
```

### Behavior preserved / intentionally not changed

- Assistant-ui remains the source of the logical message list and per-message rendering; no transcript truncation, rewriting, alternate database or backend endpoint.
- Existing `ProgressiveMessages` remains the **default** and explicit Off path.
- Physical wheel, scrollbar drag, search reveal, append/streaming and thread switches share the existing viewport/reader-intent model rather than competing scroll controllers.
- The feature's expensive window/DOM modules are lazy-loaded only for enabled threads.

## Feature scope and remaining parity work

- The virtual window limits mounted message DOM, not the full logical transcript or persisted assistant-ui message store.
- DOM descendant budgets are best-effort targets. A single necessary, unusually heavy message can exceed the configured target.
- The logical find adapter currently indexes **text parts**. Non-text / tool-HTML search parity and full select-all behavior across unmounted rows still need deliberate acceptance review (see the renderer and search-module source comments).
- The feature is opt-in; the default Off path keeps the existing `ProgressiveMessages` implementation.
- Rendering, scroll corrections and navigation should be tested separately: a compiling feature can still have browser-specific wheel, native-thumb, streaming or search regressions.

## Verification entry points

| Layer | Tests / workflows | Purpose |
| --- | --- | --- |
| Pure UI and policy functions | `studio/frontend/tests/agent-history-*.test.ts` | Window math, native-thumb decisions, search, preference migration, DOM and notification gating. |
| Browser interactions | `tests/studio/playwright_agent_history_scroll.py` | Menu modes, native-scroll behavior, window traversal and search reveal. |
| Large DOM | `tests/studio/playwright_agent_history_dom_sweep.py` | Synthetic heavy tool-like DOM, browser responsiveness and bounded mounted rows. |
| DOM churn / memory | `tests/studio/playwright_agent_history_dom_churn_stress.py`, `playwright_agent_history_idle_scroll_stress.py` | Extreme synthetic DOM, idle/resume wheel navigation and retention. |
| Comparative performance | `tests/studio/playwright_agent_history_perf.py` | Observational Off-vs-enabled typing-cost comparison, not a hard speedup guarantee. |
| CI entry point | `.github/workflows/studio-agent-history-ci.yml` | PR smoke matrix and manually initiated full synthetic stress matrix. |

## Integration and review policy

This design is an alternative rendering strategy within the existing Studio/assistant-ui integration, **not** a new persistence layer, a replacement for the chat runtime or a backend API change. Do not interpret test coverage or baseline measurements as proof of full feature parity until the corresponding end-to-end tests and remaining interaction reviews have passed.
