// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import {
  disclosureExpired,
  mayAutoApproveTool,
} from "../src/features/chat/api/mcp-image-privacy.ts";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

function harness(image: boolean, gone = false) {
  const posts: unknown[][] = [];
  const grants: unknown[][] = [];
  const hooks: unknown[] = [];
  let cursor = 0;
  const disclosure = image
    ? {
        purpose: "mcp_image_disclosure",
        sizeBytes: 1024,
        serverName: "Server",
        toolName: "inspect",
        destination: "https://example.test/mcp",
        field: "image",
        encoding: "base64",
        expiresAt: Date.now() + 100_000,
        expiresInMs: 100_000,
        receivedAt: Date.now(),
      }
    : undefined;
  class ToolApprovalGoneError extends Error {}
  const store = {
    toolConfirmations: {
      call: {
        sessionId: "session",
        approvalId: "approval",
        autoAllowKey: "session",
        imageDisclosure: disclosure,
      },
    },
    alwaysAllowToolsBySession: new Map([
      ["session", new Set(image ? ["inspect"] : [])],
    ]),
    allowToolAlways: (...args: unknown[]) => grants.push(args),
    clearToolConfirmation: () => {},
  };
  const component = loadWithStubs<{
    ToolConfirmationControls: (
      props: Record<string, unknown>,
    ) => StubElement | null;
  }>(
    new URL(
      "../src/components/assistant-ui/tool-confirmation-controls.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState(initial: unknown) {
          const index = cursor++;
          if (!(index in hooks))
            hooks[index] = typeof initial === "function" ? initial() : initial;
          return [
            hooks[index],
            (value: unknown) => {
              hooks[index] = value;
            },
          ];
        },
        useRef: (value: unknown) => ({ current: value }),
        useEffect: () => {},
        useCallback: (callback: unknown) => callback,
      },
      "@/components/ui/button": { Button: "button" },
      "@/features/chat/api/chat-api": {
        ToolApprovalGoneError,
        resolveToolConfirmation: async (...args: unknown[]) => {
          posts.push(args);
          if (gone) throw new ToolApprovalGoneError();
          return true;
        },
      },
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: (selector: (s: typeof store) => unknown) =>
          selector(store),
      },
      "@/features/settings": { useShortcut: () => {} },
      "@/features/chat": {
        useChatActive: () => true,
        useChatNavigationStore: () => false,
      },
      "@/features/chat/api/mcp-image-privacy": {
        disclosureExpired,
        mayAutoApproveTool,
      },
      "@/features/auth": {
        authFetch: () => {
          throw new Error("unexpected preview fetch");
        },
      },
    },
  );
  function buttons() {
    cursor = 0;
    const tree = component.ToolConfirmationControls({
      toolCallId: "call",
      toolName: "inspect",
      result: undefined,
      status: { type: "running" },
    });
    const found: StubElement[] = [];
    function visit(node: unknown) {
      if (Array.isArray(node)) {
        node.forEach(visit);
        return;
      }
      if (!node || typeof node !== "object" || !("props" in node)) return;
      const el = node as StubElement;
      if (el.type === "button") found.push(el);
      visit(el.props.children);
    }
    visit(tree);
    return found;
  }
  return { buttons, posts, grants, disclosure };
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));
const click = (button: StubElement) => (button.props.onClick as () => void)();

test("a remembered ordinary grant cannot hide or persist an image disclosure decision", async () => {
  const h = harness(true);
  const buttons = h.buttons();
  assert.deepEqual(
    buttons.map((b) => b.props.children),
    ["Share image once", "Deny"],
  );
  click(buttons[0]!);
  await settle();
  assert.deepEqual(h.posts, [
    ["session", "approval", "allow", "mcp_image_disclosure"],
  ]);
  assert.deepEqual(h.grants, []);
});

test("ordinary Always allow still records a grant after a successful response", async () => {
  const h = harness(false);
  click(h.buttons().find((b) => b.props.children === "Always allow")!);
  assert.deepEqual(h.grants, []);
  await settle();
  assert.deepEqual(h.grants, [["session", "inspect"]]);
});

test("a gone image request disables both decisions and cannot post again", async () => {
  const h = harness(true, true);
  click(h.buttons()[0]!);
  await settle();
  const buttons = h.buttons();
  assert.ok(buttons.every((b) => b.props.disabled));
  click(buttons[0]!);
  await settle();
  assert.equal(h.posts.length, 1);
  assert.deepEqual(h.grants, []);
});

test("an expired image request refuses a decision even when invoked directly", async () => {
  const h = harness(true);
  h.disclosure!.expiresInMs = 0;
  const buttons = h.buttons();
  assert.ok(buttons.every((b) => b.props.disabled));
  click(buttons[0]!);
  await settle();
  assert.deepEqual(h.posts, []);
});
