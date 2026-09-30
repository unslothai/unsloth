// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

type Capability = {
  pythonOsIsolated: boolean;
  terminalOsIsolated: boolean;
} | null;

function permissionUi(
  loginMode: string,
  permissionMode: string,
  capability: Capability = null,
) {
  const changes: string[] = [];
  const capabilityStub = {
    loadSandboxCapability: async () => capability,
    loadSettledSandboxCapability: async () => capability,
    sandboxReady: (value: NonNullable<Capability>) =>
      value.pythonOsIsolated && value.terminalOsIsolated,
    capabilityPending: () => false,
  };
  const state = {
    permissionMode,
    setPermissionMode: (value: string) => {
      changes.push(value);
    },
  };
  const component = loadWithStubs<{
    PermissionModeMenuItems: (props: {
      onRequestFullAccess: () => void;
      onRequestSandboxSetup: () => void;
    }) => StubElement;
    PERMISSION_MODE_OPTIONS: readonly { value: string; labelKey: string }[];
    permissionModeOption: (mode: string) => { value: string; labelKey: string };
    pickSandboxedMode: (
      setPermissionMode: (mode: string) => void,
      onRequestSandboxSetup: () => void,
    ) => Promise<void>;
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
      "@/features/auth/account-session": {
        useFullAccessAllowed: () => loginMode !== "multi",
      },
      "@/i18n": { useT: () => (key: string) => key },
      "./api/sandbox-capability": capabilityStub,
      "./sandbox-pick": loadWithStubs(
        new URL("../src/features/chat/sandbox-pick.ts", import.meta.url),
        { "./api/sandbox-capability": capabilityStub },
      ),
      "./sandbox-setup-dialog": {
        SandboxSetupDialog: "SandboxSetupDialog",
        useSandboxSetupDialogStore: () => () => {},
      },
      "@/components/ui/alert-dialog": { AlertDialog: "AlertDialog" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/dropdown-menu": { DropdownMenuItem: "DropdownMenuItem" },
      "@/lib/chevron-icons": {},
      "@/lib/sparkles-icon": { SparklesGlyph: "SparklesGlyph" },
      "@/lib/tick-icon": {},
      "@/lib/utils": { cn: () => "" },
      "@hugeicons/react": {},
      "./stores/chat-runtime-store": {
        useChatRuntimeStore: Object.assign(
          (selector: (state: unknown) => unknown) => selector(state),
          { getState: () => state, subscribe: () => () => {} },
        ),
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
      onRequestSandboxSetup() {},
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
      onRequestSandboxSetup() {},
    });
    assert.deepEqual(ui.changes, []);
  }
});

test("the stored values keep their order and get the new names", () => {
  const ui = permissionUi("single", "auto");
  assert.deepEqual(
    ui.component.PERMISSION_MODE_OPTIONS.map((option) => [
      option.value,
      option.labelKey,
    ]),
    [
      ["ask", "permissionModes.ask.label"],
      ["auto", "permissionModes.auto.label"],
      ["off", "permissionModes.off.label"],
      ["full", "permissionModes.full.label"],
    ],
  );
  assert.equal(ui.component.permissionModeOption("bogus").value, "auto");
});

for (const [name, capability, expected] of [
  ["an older server that cannot say", null, "applied"],
  [
    "a working OS sandbox",
    { pythonOsIsolated: true, terminalOsIsolated: true },
    "applied",
  ],
  [
    "a Terminal without OS isolation",
    { pythonOsIsolated: true, terminalOsIsolated: false },
    "setup",
  ],
  [
    "no OS sandbox",
    { pythonOsIsolated: false, terminalOsIsolated: false },
    "setup",
  ],
] as const) {
  test(`picking Full access in sandbox with ${name} is ${expected}`, async () => {
    const ui = permissionUi("single", "auto", capability);
    const applied: string[] = [];
    let setupRequested = 0;
    await ui.component.pickSandboxedMode(
      (mode) => applied.push(mode),
      () => {
        setupRequested += 1;
      },
    );
    if (expected === "applied") {
      assert.deepEqual(applied, ["off"]);
      assert.equal(setupRequested, 0);
    } else {
      assert.deepEqual(applied, []);
      assert.equal(setupRequested, 1);
    }
  });
}
