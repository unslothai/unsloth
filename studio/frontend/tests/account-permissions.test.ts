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
  backend?: string;
} | null;

function permissionUi(
  owner: boolean,
  permissionMode: string,
  loginMode = "multi",
  capability: Capability = null,
  sandboxLevel = "high",
) {
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
  const levels: string[] = [];
  const popups: boolean[] = [];
  const opened: [string, unknown][] = [];
  const capabilityStub = {
    loadSandboxCapability: async () => capability,
    loadSettledSandboxCapability: async () => capability,
    cachedSandboxCapability: () => null,
    onSandboxCapabilityChange: () => () => {},
    sandboxReady: (value: NonNullable<Capability>) =>
      value.pythonOsIsolated && value.terminalOsIsolated,
    capabilityPending: () => false,
  };
  const state = {
    permissionMode,
    sandboxLevel,
    setSandboxLevel: (value: string) => {
      levels.push(value);
    },
    setPermissionMode: (value: string) => {
      changes.push(value);
    },
  };
  const component = loadWithStubs<{
    PermissionModeMenuItems: (props: {
      onRequestFullAccess: () => void;
    }) => StubElement;
    PermissionMenuLabel: (props: {
      sandboxControls: boolean;
      onOsSandboxMissing?: () => void;
    }) => StubElement;
    PERMISSION_MODE_OPTIONS: readonly { value: string; label: string }[];
    permissionModeOption: (mode: string) => { value: string; label: string };
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
        useLayoutEffect: (effect: () => void) => effect(),
        // The capability hook starts at null; the menu sees the stubbed answer.
        useState: (initial: unknown) => [initial === null ? capability : initial, () => {}],
        useId: () => "id",
        useRef: (current: unknown) => ({ current }),
      },
      "lucide-react": {
        ChevronDown: "ChevronDown",
        Hand: "Hand",
        ShieldCheck: "ShieldCheck",
      },
      "radix-ui": {
        DropdownMenu: {
          Item: "DropdownMenuPrimitive.Item",
          Sub: "DropdownMenuPrimitive.Sub",
          SubTrigger: "DropdownMenuPrimitive.SubTrigger",
        },
      },
      "@/features/settings": {
        useSettingsDialogStore: (selector: (state: unknown) => unknown) =>
          selector({
            openDialog: (tab: string, options?: { scrollTarget?: string }) => {
              opened.push([tab, options?.scrollTarget]);
            },
          }),
      },
      "@/i18n": { useT: () => (key: string) => key },
      "@/features/auth/account-session": accountSession,
      "./api/sandbox-capability": capabilityStub,
      "./sandbox-pick": loadWithStubs(
        new URL("../src/features/chat/sandbox-pick.ts", import.meta.url),
        { "./api/sandbox-capability": capabilityStub },
      ),
      "./sandbox-setup-dialog": {
        SandboxSetupDialog: "SandboxSetupDialog",
        useSandboxSetupDialogStore: (selector: (state: unknown) => unknown) =>
          selector({
            setOpen: (open: boolean) => {
              popups.push(open);
            },
          }),
      },
      "@/components/ui/alert-dialog": { AlertDialog: "AlertDialog" },
      "@/components/ui/button": { Button: "Button" },
      "@/components/ui/switch": { Switch: "Switch" },
      "./sandbox-level": loadWithStubs(
        new URL("../src/features/chat/sandbox-level.ts", import.meta.url),
        {},
      ),
      "@/components/ui/dropdown-menu": {
        DropdownMenuItem: "DropdownMenuItem",
        DropdownMenuSeparator: "DropdownMenuSeparator",
        DropdownMenuSubContent: "DropdownMenuSubContent",
      },
      "@/lib/chevron-icons": {},
      "@/lib/sparkles-icon": { SparklesGlyph: "SparklesGlyph" },
      "@/lib/shield-alert-icon": { ShieldAlertGlyph: "ShieldAlertGlyph" },
      "@hugeicons/core-free-icons": {},
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
  return { component, changes, levels, popups, opened };
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

test("the stored values keep their order and their names", () => {
  const ui = permissionUi(true, "auto");
  assert.deepEqual(
    ui.component.PERMISSION_MODE_OPTIONS.map((option) => [
      option.value,
      option.label,
    ]),
    [
      ["ask", "Ask for approval"],
      ["auto", "Approve for me"],
      ["off", "Run automatically"],
      ["full", "Full access"],
    ],
  );
  assert.equal(ui.component.permissionModeOption("bogus").value, "auto");
});

const NO_OS_SANDBOX = { pythonOsIsolated: false, terminalOsIsolated: false, backend: "bubblewrap" };
const OS_SANDBOX = { pythonOsIsolated: true, terminalOsIsolated: true, backend: "bubblewrap" };
const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

for (const [name, capability] of [
  ["no OS sandbox", NO_OS_SANDBOX],
  ["a working OS sandbox", OS_SANDBOX],
  ["an older server that cannot say", null],
] as const) {
  test(`picking Run automatically with ${name} applies it, with no popup and no row hint`, () => {
    const ui = permissionUi(true, "auto", "single", capability);
    const rows = ui.component.PermissionModeMenuItems({
      onRequestFullAccess() {
        assert.fail("not Full access");
      },
    }).props.children as StubElement[];
    const off = rows[2];
    (off.props.onSelect as () => void)();
    assert.deepEqual(ui.changes, ["off"]);
    assert.deepEqual(ui.popups, []);
    // The row holds its icon and the label/description column only: no "not available" line.
    const [, column] = off.props.children as [StubElement, StubElement];
    assert.equal((column.props.children as unknown[]).length, 2);
  });
}

function sandboxPicker(ui: ReturnType<typeof permissionUi>, onOsSandboxMissing?: () => void) {
  const label = ui.component.PermissionMenuLabel({ sandboxControls: true, onOsSandboxMissing });
  const [heading, control] = label.props.children as [StubElement, StubElement];
  const sub = (control.type as (props: unknown) => StubElement)(control.props);
  const [chip, content] = sub.props.children as [StubElement, StubElement];
  // Heading (with Learn more beside it), then the levels.
  const [pickerHeading, levels] = content.props.children as [StubElement, StubElement[]];
  const [, learnMore] = pickerHeading.props.children as [StubElement, StubElement];
  // Listed weakest first; the stub JSX runtime drops keys, so rows go by position.
  // Disabled leads the list only under Full access.
  const order = levels.length === 3 ? ["off", "low", "high"] : ["low", "high"];
  assert.equal(levels.length, order.length);
  const level = (value: string) => levels[order.indexOf(value)];
  // The ticked row is the level in effect.
  const active = order.find((_, i) => (levels[i].props.children as unknown[])[1] !== null);
  return { heading, chip, level, active, learnMore };
}

test("the menu heading says Permissions and has no switch where Settings shows its own", () => {
  const ui = permissionUi(true, "auto", "single", OS_SANDBOX);
  const label = ui.component.PermissionMenuLabel({ sandboxControls: false });
  const [heading, control] = label.props.children as [StubElement, unknown];
  assert.equal(heading.props.children, "settings.general.permissions.sectionTitle");
  assert.equal(control, null);
});

for (const [name, capability, level, mode, active, disabled] of [
  ["High with a working OS sandbox", OS_SANDBOX, "high", "auto", "high", false],
  ["High saved but no OS sandbox", NO_OS_SANDBOX, "high", "auto", "low", false],
  ["Low", OS_SANDBOX, "low", "auto", "low", false],
  ["Full access", OS_SANDBOX, "high", "full", "off", true],
] as const) {
  test(`the Sandbox picker for ${name}`, () => {
    const ui = permissionUi(true, mode, "single", capability, level);
    const picker = sandboxPicker(ui);
    assert.equal(picker.active, active);
    // The chip still opens, so Full access users can see why the sandbox is off.
    assert.equal(picker.chip.props.disabled, undefined);
    assert.equal(picker.level("low").props.disabled, disabled);
    assert.equal(picker.level("high").props.disabled, disabled);
  });
}

test("under Full access, Disabled is ticked and picking it or a locked level changes nothing", async () => {
  const ui = permissionUi(true, "full", "single", OS_SANDBOX, "high");
  const picker = sandboxPicker(ui);
  assert.equal(picker.level("off").props.disabled, false);
  (picker.level("off").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(ui.levels, []);
  assert.deepEqual(ui.popups, []);
});

test("picking High without an OS sandbox opens the install popup and keeps Low", async () => {
  const ui = permissionUi(true, "auto", "single", NO_OS_SANDBOX, "low");
  (sandboxPicker(ui).level("high").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(ui.popups, [true]);
  assert.deepEqual(ui.levels, []);

  const local: string[] = [];
  const own = sandboxPicker(permissionUi(true, "auto", "single", NO_OS_SANDBOX, "low"), () =>
    local.push("popup"),
  );
  (own.level("high").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(local, ["popup"]);
});

test("picking High with a working OS sandbox applies it; picking Low applies at once", async () => {
  const ui = permissionUi(true, "auto", "single", OS_SANDBOX, "low");
  (sandboxPicker(ui).level("high").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(ui.levels, ["high"]);
  assert.deepEqual(ui.popups, []);

  const high = permissionUi(true, "auto", "single", OS_SANDBOX, "high");
  (sandboxPicker(high).level("low").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(high.levels, ["low"]);
});

test("picking the level already in effect changes nothing", async () => {
  const ui = permissionUi(true, "auto", "single", OS_SANDBOX, "high");
  (sandboxPicker(ui).level("high").props.onSelect as () => void)();
  await tick();
  assert.deepEqual(ui.levels, []);
  assert.deepEqual(ui.popups, []);
});

test("Learn more beside the picker heading opens Settings > Sandbox at Permissions", async () => {
  const ui = permissionUi(true, "auto", "single", OS_SANDBOX);
  const { learnMore } = sandboxPicker(ui);
  (learnMore.props.onSelect as () => void)();
  await tick();
  assert.deepEqual(ui.opened, [["sandbox", "sandbox-permissions"]]);
});
