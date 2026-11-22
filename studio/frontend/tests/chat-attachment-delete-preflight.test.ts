// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Without WebCrypto (plain http on a LAN address) the lazy node-forge chunk must load before a deletion commits.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Events = {
  prepareChatAttachmentDeletion: (attachmentId: string) => Promise<void>;
};

const HASHED_ID = "content-part-sha256-ab";

function loadEvents(forge: "ok" | "fails", loads: string[]): Events {
  const fakeForge = {
    md: {
      sha256: {
        create: () => ({
          update: () => {},
          digest: () => ({ toHex: () => "00" }),
        }),
      },
    },
  };
  const stubs = {
    get "node-forge"() {
      loads.push("node-forge");
      if (forge === "fails")
        throw new Error("Failed to fetch dynamically imported module");
      return { default: fakeForge };
    },
  };
  return loadWithStubs<Events>(
    new URL(
      "../src/features/chat/utils/chat-attachment-events.ts",
      import.meta.url,
    ),
    stubs,
  );
}

async function withoutSubtle(run: () => Promise<void>): Promise<void> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", {
    value: {},
    configurable: true,
    writable: true,
  });
  try {
    await run();
  } finally {
    if (descriptor) {
      Object.defineProperty(globalThis, "crypto", descriptor);
    } else {
      Reflect.deleteProperty(globalThis, "crypto");
    }
  }
}

test("with WebCrypto, preparing a deletion does not load node-forge", async () => {
  const loads: string[] = [];
  const events = loadEvents("fails", loads);
  await events.prepareChatAttachmentDeletion(HASHED_ID);
  assert.deepEqual(loads, []);
});

test("without WebCrypto, a node-forge chunk that cannot load fails the deletion up front", async () => {
  await withoutSubtle(async () => {
    const loads: string[] = [];
    const events = loadEvents("fails", loads);
    await assert.rejects(events.prepareChatAttachmentDeletion(HASHED_ID));
    assert.deepEqual(loads, ["node-forge"]);
  });
});

test("without WebCrypto, a loadable node-forge lets the deletion go ahead", async () => {
  await withoutSubtle(async () => {
    const loads: string[] = [];
    const events = loadEvents("ok", loads);
    await events.prepareChatAttachmentDeletion(HASHED_ID);
    assert.deepEqual(loads, ["node-forge"]);
  });
});

test("ids the open chat matches without hashing need nothing loaded", async () => {
  await withoutSubtle(async () => {
    const loads: string[] = [];
    const events = loadEvents("fails", loads);
    await events.prepareChatAttachmentDeletion("content-part-3");
    await events.prepareChatAttachmentDeletion("3f2c9b1e-attachment");
    assert.deepEqual(loads, []);
  });
});

test("the uploaded files dialog prepares before it deletes", () => {
  const source = readSrc(
    "features/settings/components/uploaded-files-dialog.tsx",
  );
  const remove = source.slice(source.indexOf("remove: async () => {"));
  const prepared = remove.indexOf(
    "await prepareChatAttachmentDeletion(att.id)",
  );
  const deleted = remove.indexOf("await deleteChatAttachment(");
  assert.ok(prepared >= 0, "remove() should prepare the deletion");
  assert.ok(deleted > prepared, "remove() should prepare before deleting");
});
