// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_keyboard_shortcuts.py: a vite entry with no
// backend, driving the real registry, the real store and the real useShortcut
// against a real browser's keyboard. The node suite can reach the pure functions
// but not the listener, so what a chord does to a focused button, to a text
// field, on auto-repeat, or under AltGr is only answerable here.

/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import { useChatNavigationStore } from "@/features/chat/stores/chat-navigation-store";
import {
  COMPOSER_INPUT_SELECTOR,
  useShortcut,
} from "@/features/settings/hooks/use-shortcut";
import * as registry from "@/features/settings/lib/keyboard-shortcuts";
import * as shortcutStore from "@/features/settings/stores/keyboard-shortcuts-store";
/* eslint-enable no-restricted-imports */
import { useState } from "react";
import { createRoot } from "react-dom/client";

/** How long a selection chord keeps a repeat press off the open chat. */
const SELECTION_ACTION_GRACE_MS = 750;

interface Fired {
  action: string;
  detail?: string;
}

declare global {
  interface Window {
    // Optional: the app typechecks this entry, only the harness page installs it.
    __shortcutsSmoke?: {
      registry: typeof registry;
      store: typeof shortcutStore;
      nav: typeof useChatNavigationStore;
      fired: () => Fired[];
      reset: () => void;
      setActiveChat: (value: string | null) => void;
    };
  }
}

const fired: Fired[] = [];
const record = (action: string, detail?: string) => fired.push({ action, detail });

const latchRef = { current: null as { id: string; at: number } | null };
const activeChat = { current: "chat-A" as string | null };

// The shape app-sidebar registers these three with: a selection chord clears the
// selection as it runs, so the latch is what keeps the next press off the open
// chat. Keyed by action, so a different command straight after is not swallowed.
const stampLatch = (id: string, run: () => void) => {
  latchRef.current = { id, at: Date.now() };
  run();
};
const followsSelectionAction = (id: string) => {
  const last = latchRef.current;
  return last?.id === id && Date.now() - last.at < SELECTION_ACTION_GRACE_MS;
};

const withActiveChat = (run: (item: string) => void) => {
  if (!activeChat.current) {
    record("toast.info", "Open a chat first");
    return;
  }
  run(activeChat.current);
};

function Harness() {
  const [selectionCount, setSelectionCount] = useState(0);
  const [toolPending, setToolPending] = useState(true);

  const selectionChord =
    (id: string, onSelection: string, onSuppressed: string, onActive: string) =>
    () => {
      if (selectionCount > 0) {
        stampLatch(id, () => {
          record(onSelection, String(selectionCount));
          setSelectionCount(0);
        });
        return;
      }
      if (followsSelectionAction(id)) {
        record(onSuppressed);
        return;
      }
      withActiveChat((item) => record(onActive, item));
    };

  useShortcut(
    "archiveChat",
    selectionChord("archiveChat", "archiveSelected", "archiveSuppressed", "archiveActive"),
  );
  useShortcut(
    "togglePinChat",
    selectionChord("togglePinChat", "pinSelected", "pinSuppressed", "pinActive"),
  );
  useShortcut("deleteSelectedChats", () => {
    if (selectionCount > 0) {
      record("deleteSelected", String(selectionCount));
      setSelectionCount(0);
    }
  });
  useShortcut("clearAllUnreads", () => {
    const state = useChatNavigationStore.getState();
    const cleared = state.unreadThreadIds.size;
    if (cleared === 0) {
      record("toast.info", "No unread chats");
      return;
    }
    state.clearAllUnreads();
    record(
      "toast.success",
      `Cleared ${cleared} unread ${cleared === 1 ? "chat" : "chats"}`,
    );
  });

  // The bare-key pair, with the options tool-confirmation-controls uses.
  useShortcut("approveToolRequest", () => record("approve"), {
    enabled: toolPending,
    skipInTextFields: true,
  });
  useShortcut("declineToolRequest", () => record("decline"), {
    enabled: toolPending,
    skipInTextFields: true,
    textFieldException: COMPOSER_INPUT_SELECTOR,
  });

  useShortcut("nextChat", () => record("nextChat"), { repeats: true });
  useShortcut("searchChats", () => record("searchChats"), {
    skipInTextFields: true,
  });
  useShortcut("newChat", () => record("newChat"));
  useShortcut("openSettings", () => record("openSettings"));
  useShortcut("toggleSidebar", () => record("toggleSidebar"));
  useShortcut("copySessionId", () => record("copySessionId"));

  return (
    <div>
      <button type="button" id="smoke-button" onClick={() => record("buttonClick")}>
        Deny
      </button>
      <a href="#smoke" id="smoke-link" onClick={() => record("linkClick")}>
        link
      </a>
      <input id="smoke-input" aria-label="plain field" />
      <textarea
        id="smoke-composer"
        aria-label="composer"
        className="aui-composer-input"
      />
      <div id="smoke-editable" contentEditable suppressContentEditableWarning />
      <span id="smoke-selection">{selectionCount}</span>
      <button type="button" id="smoke-select" onClick={() => setSelectionCount(3)}>
        select
      </button>
      <button
        type="button"
        id="smoke-tool-pending"
        onClick={() => setToolPending((value) => !value)}
      >
        {String(toolPending)}
      </button>
      <span id="smoke-ready">ready</span>
    </div>
  );
}

window.__shortcutsSmoke = {
  registry,
  store: shortcutStore,
  nav: useChatNavigationStore,
  fired: () => fired.slice(),
  reset: () => {
    fired.length = 0;
    latchRef.current = null;
    activeChat.current = "chat-A";
  },
  setActiveChat: (value: string | null) => {
    activeChat.current = value;
  },
};

createRoot(document.getElementById("root") as HTMLElement).render(<Harness />);
