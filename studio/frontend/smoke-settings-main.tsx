// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_settings_tabs.py, shaped like smoke-ansi.html: a vite
// entry with no backend and no auth, driving the real SettingsDialog against the real store.

import { TooltipProvider } from "@/components/ui/tooltip";
/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import { SettingsDialog } from "@/features/settings/settings-dialog";
import {
  type SettingsTab,
  useSettingsDialogStore,
} from "@/features/settings/stores/settings-dialog-store";
/* eslint-enable no-restricted-imports */
import { initializeLocale } from "@/i18n";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { Component, StrictMode, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

declare global {
  interface Window {
    // Optional: the app typechecks this entry, and only the harness page installs the handle.
    __settingsSmoke?: {
      open: (tab?: string) => void;
      close: () => void;
      setTab: (tab: string) => void;
      state: () => { open: boolean; activeTab: string };
      errors: () => string[];
    };
  }
}

const seenErrors: string[] = [];
window.addEventListener("error", (e) => {
  seenErrors.push(String(e.message));
});
window.addEventListener("unhandledrejection", (e) => {
  seenErrors.push(String(e.reason));
});

// The app has no boundary above the dialog, so give the harness one and record what reaches
// it; whether anything does is exactly what is under test.
class Boundary extends Component<
  { children: ReactNode },
  { error: string | null }
> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(error: unknown) {
    return { error: String(error) };
  }
  render() {
    if (this.state.error) {
      return <div data-testid="harness-error-boundary">{this.state.error}</div>;
    }
    return this.props.children;
  }
}

const store = useSettingsDialogStore;
window.__settingsSmoke = {
  open: (tab?: string) => {
    store.getState().openDialog(tab as SettingsTab | undefined);
  },
  close: () => {
    store.getState().closeDialog();
  },
  setTab: (tab: string) => {
    store.getState().setActiveTab(tab as SettingsTab);
  },
  state: () => {
    const s = store.getState();
    return { open: s.open, activeTab: s.activeTab };
  },
  errors: () => [...seenErrors],
};

function Harness() {
  return (
    <TooltipProvider>
      <div data-testid="harness-root">
        <Boundary>
          <SettingsDialog />
        </Boundary>
      </div>
    </TooltipProvider>
  );
}

// Several panels use router hooks (Link, useNavigate), so a one-route memory router is in scope.
const harnessRootRoute = createRootRoute({ component: Harness });
const harnessIndexRoute = createRoute({
  getParentRoute: () => harnessRootRoute,
  path: "/",
  component: () => null,
});
const harnessRouter = createRouter({
  routeTree: harnessRootRoute.addChildren([harnessIndexRoute]),
  history: createMemoryHistory({ initialEntries: ["/"] }),
});

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element not found");
const root = createRoot(rootElement);
const strict = !new URLSearchParams(window.location.search).has("nostrict");

function render(): void {
  const tree = <RouterProvider router={harnessRouter} />;
  root.render(strict ? <StrictMode>{tree}</StrictMode> : tree);
}

const localeInitialization = initializeLocale();
if (typeof localeInitialization !== "string") {
  void localeInitialization.then(render);
} else {
  render();
}
