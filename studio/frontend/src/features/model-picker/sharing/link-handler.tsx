// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Suspense,
  lazy,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import { runConfigInbox } from "./inbox";
import {
  receiveStartupRunConfigUrl,
  subscribeRunConfigSession,
} from "./receive-link";

const SharedRunConfigLinkEditor = lazy(() =>
  import("./link-editor").then((module) => ({
    default: module.SharedRunConfigLinkEditor,
  })),
);

export function SharedRunConfigLinkHandler() {
  const pending = useSyncExternalStore(
    runConfigInbox.subscribe,
    runConfigInbox.getSnapshot,
  );
  const [, setAuthRevision] = useState(0);

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (active) {
        receiveStartupRunConfigUrl();
      }
    });
    const onAuth = () => setAuthRevision((revision) => revision + 1);
    const unsubscribe = subscribeRunConfigSession(onAuth);
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  return pending ? (
    <Suspense fallback={null}>
      <SharedRunConfigLinkEditor pending={pending} />
    </Suspense>
  ) : null;
}
