// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LazyImportBoundary,
  LazyImportFailure,
} from "@/components/lazy-import-boundary";
import type { ChatSearch } from "@/features/chat";
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
  import("./runtime").then((module) => ({
    default: module.SharedRunConfigLinkEditor,
  })),
);

export function SharedRunConfigLinkHandler({
  chatSearch,
}: { chatSearch: ChatSearch | null }) {
  const pending = useSyncExternalStore(
    runConfigInbox.subscribe,
    runConfigInbox.getSnapshot,
  );
  const [, setAuthRevision] = useState(0);

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (active) {
        void receiveStartupRunConfigUrl();
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
    <LazyImportBoundary
      key={pending.id}
      fallback={
        <LazyImportFailure
          message="Shared run settings could not load. Reload Studio and reopen the link to try again."
          reloadLabel="Reload Studio"
          dismissLabel="Dismiss"
          onDismiss={() => runConfigInbox.clear(pending.id)}
          testId="shared-run-settings-unavailable"
          className="fixed bottom-4 right-4 z-50 max-w-sm rounded-lg border bg-background p-4"
        />
      }
    >
      <Suspense fallback={null}>
        <SharedRunConfigLinkEditor pending={pending} chatSearch={chatSearch} />
      </Suspense>
    </LazyImportBoundary>
  ) : null;
}
