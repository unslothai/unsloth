// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

function permissionUi(loginMode: string, permissionMode: string) {
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
        CircleAlert: "CircleAlert",
        Hand: "Hand",
        ShieldCheck: "ShieldCheck",
      },
      "@/features/auth/account-session": { useLoginMode: () => loginMode },
      "@/components/ui/alert-dialog": { AlertDialog: "AlertDialog" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dropdown-menu": { DropdownMenuItem: "DropdownMenuItem" },
      "@/lib/chevron-icons": {},
      "@/lib/sparkles-icon": { SparklesGlyph: "SparklesGlyph" },
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

for (const mode of ["single", "multi"]) {
  test(`${mode} permission menu enforces installation full-access policy`, () => {
    const ui = permissionUi(mode, "full");
    const menu = ui.component.PermissionModeMenuItems({
      onRequestFullAccess() {},
    });
    const rows = menu.props.children as StubElement[];
    assert.equal(rows.length, mode === "multi" ? 3 : 4);
    assert.deepEqual(ui.changes, mode === "multi" ? ["auto"] : []);
    const dialog = ui.component.FullAccessConfirmDialog({
      open: true,
      onOpenChange() {},
    });
    assert.equal(dialog === null, mode === "multi");
  });
}

test("multi-user policy leaves non-full preferences untouched", () => {
  for (const mode of ["ask", "auto", "off"]) {
    const ui = permissionUi("multi", mode);
    ui.component.PermissionModeMenuItems({
      onRequestFullAccess() {
        assert.fail("Full access must be unavailable");
      },
    });
    assert.deepEqual(ui.changes, []);
  }
});
