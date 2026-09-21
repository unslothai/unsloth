// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRouterState } from "@tanstack/react-router";
import {
  Suspense,
  lazy,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { runConfigInbox } from "./inbox";
import {
  receiveRunConfigUrl,
  receiveStartupRunConfigUrl,
  subscribeRunConfigSession,
} from "./receive-link";

const SharedRunConfigLinkEditor = lazy(() =>
  import("./link-editor").then((module) => ({
    default: module.SharedRunConfigLinkEditor,
  })),
);

export function SharedRunConfigLinkHandler() {
  const location = useRouterState({ select: (state) => state.location });
  const pending = useSyncExternalStore(
    runConfigInbox.subscribe,
    runConfigInbox.getSnapshot,
  );
  const [, setAuthRevision] = useState(0);
  const previousUrl = useRef(window.location.href);

  useEffect(() => {
    receiveStartupRunConfigUrl(window.location.href);
    previousUrl.current = window.location.href;
    const onLocation = () => {
      const currentUrl = window.location.href;
      if (currentUrl === previousUrl.current) {
        return;
      }
      previousUrl.current = currentUrl;
      receiveRunConfigUrl(currentUrl);
    };
    const onAuth = () => setAuthRevision((revision) => revision + 1);
    window.addEventListener("hashchange", onLocation);
    window.addEventListener("popstate", onLocation);
    const unsubscribeSession = subscribeRunConfigSession(onAuth);
    return () => {
      window.removeEventListener("hashchange", onLocation);
      window.removeEventListener("popstate", onLocation);
      unsubscribeSession();
    };
  }, []);

  useEffect(() => {
    const currentUrl = new URL(location.href, window.location.origin).href;
    if (previousUrl.current !== currentUrl) {
      previousUrl.current = currentUrl;
      receiveRunConfigUrl(currentUrl);
    }
  }, [location.href]);

  return pending ? (
    <Suspense fallback={null}>
      <SharedRunConfigLinkEditor pending={pending} />
    </Suspense>
  ) : null;
}
