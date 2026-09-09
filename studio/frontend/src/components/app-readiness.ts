// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  createContext,
  createElement,
  useContext,
  useLayoutEffect,
  useState,
  type ReactNode,
} from "react";

function signalReloadSnapshotReady(): void {
  window.dispatchEvent(new Event("unsloth:app-shell-ready"));
}

/** Each mounted desktop subtree owns its callback; late work cannot reveal its successor. */
export function createAppReadinessScope(onReady: () => void) {
  let active = true;
  return {
    activate: () => { active = true; },
    dispose: () => { active = false; },
    signalReady: () => {
      if (!active) return;
      onReady();
      signalReloadSnapshotReady();
    },
  };
}

const AppShellReadyContext = createContext(signalReloadSnapshotReady);
export const AppRevealedContext = createContext(true);

export function useAppShellReadySignal(): () => void {
  return useContext(AppShellReadyContext);
}

/** Suppress the portal itself, without changing its owner's controlled/open store state. */
export function AppPortalGate({ children }: { children: ReactNode }): ReactNode {
  return useContext(AppRevealedContext) ? children : null;
}

export function AppReadinessBoundary({
  onReady,
  revealed,
  children,
}: {
  onReady: (ready: boolean) => void;
  revealed: boolean;
  children: ReactNode;
}): ReactNode {
  const [scope] = useState(() => createAppReadinessScope(() => onReady(true)));
  useLayoutEffect(() => {
    scope.activate();
    return scope.dispose;
  }, [scope]);
  return createElement(
    AppShellReadyContext.Provider,
    { value: scope.signalReady },
    createElement(AppRevealedContext.Provider, { value: revealed }, children),
  );
}
