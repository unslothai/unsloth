// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

function permissionUi(owner: boolean, permissionMode: string, loginMode = "multi") {
  const accountSession = loadWithStubs<{
    useFullAccessAllowed: () => boolean;
  }>(new URL("../src/features/auth/account-session.ts", import.meta.url), {
    react: { useSyncExternalStore: (_subscribe: unknown, snapshot: () => boolean) => snapshot() },
    "./login-client": {
      getFullAccessAllowed: () => loginMode !== "multi",
    },
    "./session": {
      getAuthToken: () => `e30.${Buffer.from(JSON.stringify({
        sub: owner ? "unsloth" : "alice", role: owner ? "owner" : "user",
      })).toString("base64url")}.sig`,
    },
  });
  const changes: string[] = [];
  const state = {
    permissionMode,
    setPermissionMode: (value: string) => {
      changes.push(value);
    },
  };
  const component = loadWithStubs<{
    PermissionModeMenuItems: (props: {
      onRequestFullAccess: () => void;
    }) => StubElement;
    FullAccessConfirmDialog: (props: {
      open: boolean;
      onOpenChange: () => void;
    }) => StubElement | null;
  }>(
    new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useEffect: (effect: () => void) => effect(),
        useState: () => [false, () => {}],
      },
      "lucide-react": {
        ChevronDown: "ChevronDown",
        Hand: "Hand",
        ShieldCheck: "ShieldCheck",
      },
      "radix-ui": { DropdownMenu: { Item: "DropdownMenuPrimitive.Item" } },
      "@/features/settings": {
        useSettingsDialogStore: (selector: (state: unknown) => unknown) =>
          selector({ openDialog: () => {} }),
      },
      "@/i18n": { useT: () => (key: string) => key },
      "@/features/auth/account-session": accountSession,
      "@/components/ui/alert-dialog": { AlertDialog: "AlertDialog" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dropdown-menu": { DropdownMenuItem: "DropdownMenuItem" },
      "@/lib/chevron-icons": {},
      "@/lib/sparkles-icon": { SparklesGlyph: "SparklesGlyph" },
      "@/lib/shield-alert-icon": { ShieldAlertGlyph: "ShieldAlertGlyph" },
      "@hugeicons/core-free-icons": {},
      "@/lib/tick-icon": {},
      "@/lib/utils": { cn: () => "" },
      "@hugeicons/react": {},
      "./stores/chat-runtime-store": {
        useChatRuntimeStore: (selector: (state: unknown) => unknown) =>
          selector(state),
      },
    },
  );
  return { component, changes };
}

for (const loginMode of ["single", "multi"]) {
  for (const owner of [true, false]) {
    test(`${loginMode} permission menu allows full access only for the owner (${owner})`, () => {
      const ui = permissionUi(owner, "full", loginMode);
      const menu = ui.component.PermissionModeMenuItems({
        onRequestFullAccess() {},
      });
      const rows = menu.props.children as StubElement[];
      assert.equal(rows.length, owner ? 4 : 3);
      assert.deepEqual(ui.changes, owner ? [] : ["auto"]);
      const dialog = ui.component.FullAccessConfirmDialog({
        open: true,
        onOpenChange() {},
      });
      assert.equal(dialog === null, !owner);
    });
  }
}

test("multi-user policy leaves non-full preferences untouched", () => {
  for (const mode of ["ask", "auto", "off"]) {
    const ui = permissionUi(false, mode);
    ui.component.PermissionModeMenuItems({
      onRequestFullAccess() {
        assert.fail("Full access must be unavailable");
      },
    });
    assert.deepEqual(ui.changes, []);
  }
});
