// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import * as events from "../../src/features/auth/session-events.ts";
import { createDeepLinkIntentGate } from "../../src/features/deep-links/deep-link-intent.ts";
import { parseUnslothDeepLink } from "../../src/features/deep-links/parse-deep-link.ts";
import { modelConfigDraftKey } from "../../src/features/model-picker/model-config/model-config-draft.ts";
import { createRunConfigInbox } from "../../src/features/model-picker/sharing/inbox.ts";
import * as linkAddress from "../../src/features/model-picker/sharing/link-address.ts";
import type * as Receiver from "../../src/features/model-picker/sharing/receive-link.ts";
import { loadWithStubs } from "./module-stubs.ts";
import * as links from "./sharing-links.ts";

export const settle = () =>
  new Promise<void>((resolve) => setImmediate(resolve));

export function receiverHarness({
  desktop = false,
  signedIn = () => true,
  loadParser = () => links,
  errors = [],
  cleared = [],
}: {
  desktop?: boolean;
  signedIn?: () => boolean;
  loadParser?: () => typeof links | Promise<typeof links>;
  errors?: string[];
  cleared?: string[];
} = {}) {
  const inbox = createRunConfigInbox();
  let nextId = 0;
  const receiver = loadWithStubs<typeof Receiver>(
    new URL(
      "../../src/features/model-picker/sharing/receive-link.ts",
      import.meta.url,
    ),
    {
      "@/lib/api-base": { isTauri: desktop },
      "@/lib/toast": {
        toast: { error: (message: string) => errors.push(message) },
      },
      "@/features/auth": { ...events, hasAuthToken: signedIn },
      "@/features/deep-links": {
        createDeepLinkIntentGate,
        parseUnslothDeepLink,
      },
      "../model-config/model-config-draft": {
        markModelConfigDraftEdited: () => undefined,
        modelConfigDraftKey,
      },
      "../model-config/model-config-handoff": {
        clearModelConfigHandoff: (id: string) => cleared.push(id),
        createModelConfigHandoffRequestId: () => `request-${++nextId}`,
      },
      "./inbox": { runConfigInbox: inbox },
      "./link-address": linkAddress,
      get "./runtime"() {
        return loadParser();
      },
    },
  );
  return { receiver, inbox, errors, cleared };
}
