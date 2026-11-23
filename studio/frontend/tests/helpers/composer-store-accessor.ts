// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Builds the actual ComposerMethods store accessor from the installed
// runtime-clients wrapper, not a hand-written facade. The private context
// provider is required because ComposerClient emits through the store event
// manager; the production React hook supplies the same context.

import "./tap-message-channel.ts";

import type { ComposerRuntime } from "@assistant-ui/core";
import { ComposerClient } from "@assistant-ui/core/store/internal";
import { tapClientResource } from "@assistant-ui/store";
import { createResourceRoot, resource } from "@assistant-ui/tap";
import type { ComposerMethods } from "../../node_modules/@assistant-ui/core/dist/store/scopes/composer.js";
import { withAssistantTapContextProvider } from "../../node_modules/@assistant-ui/store/dist/utils/tap-assistant-context.js";

export { closeTapMessageChannels } from "./tap-message-channel.ts";

export type ComposerAccessorFixture = {
  readonly accessor: ComposerMethods;
  readonly dispose: () => void;
};

export function composerStoreAccessor(
  runtime: ComposerRuntime,
  threadId = "thread-1",
): ComposerAccessorFixture {
  const Root = resource(() => {
    const clientRef = { current: null, parent: null };
    return withAssistantTapContextProvider(
      {
        clientRef,
        emit: () => undefined,
      } as never,
      () =>
        tapClientResource(
          ComposerClient({
            runtime,
            threadIdRef: { current: threadId },
          } as never),
        ),
    );
  });
  const root = createResourceRoot();
  const handle = root.render(Root()) as {
    getValue(): { methods: ComposerMethods };
  };
  return {
    accessor: handle.getValue().methods,
    dispose: () => root.unmount(),
  };
}
