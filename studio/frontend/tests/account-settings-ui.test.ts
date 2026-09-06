// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";
import { en } from "../src/i18n/locales/en.ts";
import { SETTINGS_SEARCH_INDEX } from "../src/features/settings/settings-search.ts";

const tabUrl = new URL(
  "../src/features/settings/tabs/accounts-tab.tsx",
  import.meta.url,
);
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
function nodes(node: unknown): StubElement[] {
  if (Array.isArray(node)) return node.flatMap(nodes);
  if (!node || typeof node !== "object" || !("props" in node)) return [];
  const element = node as StubElement;
  return [element, ...nodes(element.props.children)];
}
function content(node: unknown): string {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(content).join(" ");
  if (node && typeof node === "object" && "props" in node)
    return content((node as StubElement).props.children);
  return "";
}
function translate(key: string, values: Record<string, string> = {}) {
  const message = key
    .split(".")
    .reduce<unknown>(
      (value, part) => (value as Record<string, unknown>)[part],
      en,
    ) as string;
  return message.replace(
    /\{([^}]+)\}/g,
    (_match, name: string) => values[name] ?? "",
  );
}
function tab(owner = true) {
  const states: unknown[] = [];
  const effects: (() => void)[] = [];
  let cursor = 0;
  const calls: string[] = [];
  const accounts = [
    {
      account_id: "owner",
      username: "unsloth",
      role: "owner",
      is_active: true,
    },
    {
      account_id: "alice-id",
      username: "alice",
      role: "user",
      is_active: true,
    },
  ];
  const setup = {
    account_id: "alice-id",
    username: "alice",
    setup_code: "one-time-secret",
    expires_at: "2026-09-06T13:00:00Z",
  };
  const api = loadWithStubs<{ AccountsTab: () => StubElement | null }>(tabUrl, {
    "react/jsx-runtime": stubJsxRuntime(),
    react: {
      useState: (initial: unknown) => {
        const index = cursor++;
        if (!(index in states)) states[index] = initial;
        return [
          states[index],
          (value: unknown) => {
            states[index] = value;
          },
        ];
      },
      useEffect: (effect: () => void) => {
        effects.push(effect);
      },
    },
    "@/features/auth/account-session": { useIsAccountOwner: () => owner },
    "@/components/ui/button": { Button: "Button" },
    "@/components/ui/input": { Input: "Input" },
    "@/components/ui/label": { Label: "Label" },
    "@/components/ui/alert-dialog": Object.fromEntries(
      [
        "AlertDialog",
        "AlertDialogAction",
        "AlertDialogCancel",
        "AlertDialogContent",
        "AlertDialogDescription",
        "AlertDialogFooter",
        "AlertDialogHeader",
        "AlertDialogTitle",
      ].map((name) => [name, name]),
    ),
    "@/lib/copy-to-clipboard": {
      copyToClipboard: async (value: string) => {
        calls.push(`copy:${value}`);
        return true;
      },
    },
    "@/i18n": { useT: () => translate },
    "../api/accounts": {
      fetchAccounts: async () => {
        calls.push("list");
        return accounts;
      },
      createAccount: async (username: string) => {
        calls.push(`create:${username}`);
        return setup;
      },
      regenerateSetupCode: async (accountId: string) => {
        calls.push(`regenerate:${accountId}`);
        return { ...setup, setup_code: "regenerated-secret" };
      },
      setAccountActive: async (accountId: string, active: boolean) => {
        calls.push(`active:${accountId}:${active}`);
        accounts[1].is_active = active;
      },
      deleteAccount: async (accountId: string) => {
        calls.push(`delete:${accountId}`);
        accounts.splice(1);
      },
    },
  });
  const render = () => {
    cursor = 0;
    const wrapper = api.AccountsTab();
    return wrapper ? (wrapper.type as () => StubElement)() : null;
  };
  return {
    render,
    calls,
    initialize: async () => {
      render();
      for (const effect of effects.splice(0)) effect();
      await tick();
      return render();
    },
  };
}
const click = async (tree: unknown, label: string) => {
  const button = nodes(tree).find(
    (node) => node.type === "Button" && content(node) === label,
  );
  assert.ok(button, label);
  (button.props.onClick as () => void)();
  await tick();
};

test("managed accounts mount no Accounts panel and send no list request", async () => {
  const ui = tab(false);
  assert.equal(await ui.initialize(), null);
  assert.deepEqual(ui.calls, []);
});

test("owner lists accounts without administrative actions on the owner row", async () => {
  const ui = tab();
  const tree = await ui.initialize();
  const ownerRow = nodes(tree).find(
    (node) => node.props["data-testid"] === "account-unsloth",
  );
  assert.ok(ownerRow);
  assert.equal(
    nodes(ownerRow).filter((node) => node.type === "Button").length,
    0,
  );
  assert.match(content(tree), /Installation owner/);
});

test("create shows a copyable expiring setup code once and regeneration replaces it", async () => {
  const ui = tab();
  let tree = await ui.initialize();
  const input = nodes(tree).find(
    (node) => node.props.id === "new-account-username",
  );
  (input?.props.onChange as (event: unknown) => void)({
    target: { value: "alice" },
  });
  tree = ui.render();
  const form = nodes(tree).find((node) => node.type === "form");
  (form?.props.onSubmit as (event: unknown) => void)({ preventDefault() {} });
  await tick();
  tree = ui.render();
  assert.ok(ui.calls.includes("create:alice"));
  assert.match(content(tree), /one-time-secret/);
  assert.match(content(tree), /Expires/);
  assert.match(content(tree), /60 minutes/);
  await click(tree, "Copy setup code");
  assert.ok(ui.calls.includes("copy:one-time-secret"));
  await click(ui.render(), "Done");
  assert.doesNotMatch(content(ui.render()), /one-time-secret/);
  await click(ui.render(), "Regenerate setup code");
  assert.match(content(ui.render()), /regenerated-secret/);
});

test("activation controls follow state and delete requires a named retirement confirmation", async () => {
  const ui = tab();
  let tree = await ui.initialize();
  await click(tree, "Deactivate");
  assert.ok(ui.calls.includes("active:alice-id:false"));
  await click(ui.render(), "Reactivate");
  assert.ok(ui.calls.includes("active:alice-id:true"));
  await click(ui.render(), "Delete account");
  tree = ui.render();
  assert.ok(!ui.calls.some((call) => call.startsWith("delete:")));
  assert.match(content(tree), /Delete alice\?/);
  assert.match(content(tree), /revokes alice's sessions/);
  assert.match(content(tree), /renamed aside, never deleted/);
  const action = nodes(tree).find((node) => node.type === "AlertDialogAction");
  (action?.props.onClick as (event: unknown) => void)({ preventDefault() {} });
  await tick();
  assert.ok(ui.calls.includes("delete:alice-id"));
});

test("Accounts is registered, searchable, and filtered from managed navigation and deferred panels", () => {
  assert.deepEqual(SETTINGS_SEARCH_INDEX.accounts, [
    "settings.accounts.title",
    "settings.accounts.create",
  ]);
  const dialog = readFileSync(
    new URL("../src/features/settings/settings-dialog.tsx", import.meta.url),
    "utf8",
  );
  const store = readFileSync(
    new URL(
      "../src/features/settings/stores/settings-dialog-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(store, /"accounts"/);
  assert.match(dialog, /import\("\.\/tabs\/accounts-tab"\)/);
  assert.match(dialog, /tab\.id !== "accounts" \|\| isOwner/);
  assert.match(dialog, /deferredTab === "accounts" && !isOwner \? "general"/);
  assert.equal((dialog.match(/visibleTabs\.map/g) ?? []).length, 2);
});
