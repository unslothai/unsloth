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
type TabOptions = {
  /** Hold the post-mutation account refresh open, the way a slow backend does. */
  stallRefresh?: boolean;
  /** Replace the fixture rows, for the cases the default two cannot express. */
  accounts?: Record<string, unknown>[];
};
function tab(
  owner = true,
  mutationError: string | null = null,
  options: TabOptions = {},
) {
  const states: unknown[] = [];
  const effects: (() => void)[] = [];
  let cursor = 0;
  const calls: string[] = [];
  let releaseRefresh: (() => void) | null = null;
  const accounts = options.accounts ?? [
    {
      account_id: "owner",
      username: "unsloth",
      role: "owner",
      is_active: true,
      created_at: "2026-09-01T12:00:00Z",
    },
    {
      account_id: "alice-id",
      username: "alice",
      role: "user",
      is_active: true,
      created_at: "2026-09-01T12:00:00Z",
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
      useRef: (initial: unknown) => ({ current: initial }),
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
    "@/features/auth": { useIsAccountOwner: () => owner },
    "@/features/profile": { UserAvatar: "UserAvatar" },
    "@hugeicons/core-free-icons": {},
    "@hugeicons/react": { HugeiconsIcon: "Icon" },
    "@/lib/tick-icon": {},
    "@/lib/utils": {
      cn: (...values: unknown[]) => values.filter(Boolean).join(" "),
    },
    "@/components/ui/spinner": { Spinner: "Spinner" },
    "@/components/ui/dropdown-menu": Object.fromEntries(
      [
        "DropdownMenu",
        "DropdownMenuContent",
        "DropdownMenuItem",
        "DropdownMenuSeparator",
        "DropdownMenuTrigger",
      ].map((name) => [name, name]),
    ),
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
    "@/i18n": { useT: () => translate, useLocale: () => "en" },
    "@/components/ui/dialog": Object.fromEntries(
      [
        "Dialog",
        "DialogContent",
        "DialogHeader",
        "DialogTitle",
        "DialogDescription",
        "DialogFooter",
      ].map((name) => [name, name]),
    ),
    "../api/accounts": {
      fetchAccounts: async () => {
        calls.push("list");
        // The first call is the initial load and always resolves; a stalled run
        // holds every refresh AFTER it, which is the window in which a one-time
        // setup code is on screen while `perform` is still busy.
        if (options.stallRefresh && calls.filter((c) => c === "list").length > 1) {
          await new Promise<void>((resolve) => {
            releaseRefresh = resolve;
          });
        }
        return accounts;
      },
      createAccount: async (username: string) => {
        calls.push(`create:${username}`);
        return setup;
      },
      regenerateSetupCode: async (accountId: string) => {
        calls.push(`regenerate:${accountId}`);
        if (mutationError) throw new Error(mutationError);
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
    releaseRefresh: () => releaseRefresh?.(),
    initialize: async () => {
      render();
      for (const effect of effects.splice(0)) effect();
      await tick();
      return render();
    },
  };
}
const confirm = async (tree: unknown) => {
  const action = nodes(tree).find((node) => node.type === "AlertDialogAction");
  assert.ok(action);
  (action.props.onClick as (event: unknown) => void)({ preventDefault() {} });
  await tick();
};
const click = async (tree: unknown, label: string) => {
  const button = nodes(tree).find(
    (node) =>
      (node.type === "Button" || node.type === "DropdownMenuItem") &&
      content(node).trim() === label,
  );
  assert.ok(button, label);
  ((button.props.onSelect ?? button.props.onClick) as () => void)();
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
  await click(tree, "Create account");
  tree = ui.render();
  assert.equal(
    nodes(tree).find((node) => node.type === "Dialog")?.props.open,
    true,
  );
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
  assert.ok(!ui.calls.some((call) => call.startsWith("regenerate:")));
  await confirm(ui.render());
  assert.ok(ui.calls.includes("regenerate:alice-id"));
  assert.match(content(ui.render()), /regenerated-secret/);
});

test("regenerating a setup code names what it destroys before it runs", async () => {
  const ui = tab();
  await click(await ui.initialize(), "Regenerate setup code");
  const tree = ui.render();
  assert.match(content(tree), /Reset alice's password\?/);
  assert.match(content(tree), /revokes their API keys/);
  const dialog = nodes(tree).find((node) => node.type === "AlertDialog");
  assert.ok(dialog);
  (dialog.props.onOpenChange as (open: boolean) => void)(false);
  await tick();
  assert.ok(!ui.calls.some((call) => call.startsWith("regenerate:")));
  assert.doesNotMatch(content(ui.render()), /regenerated-secret/);
});

test("activation controls follow state and delete requires a named retirement confirmation", async () => {
  const ui = tab();
  let tree = await ui.initialize();
  await click(tree, "Disable");
  assert.ok(ui.calls.includes("active:alice-id:false"));
  await click(ui.render(), "Enable");
  assert.ok(ui.calls.includes("active:alice-id:true"));
  await click(ui.render(), "Delete account");
  tree = ui.render();
  assert.ok(!ui.calls.some((call) => call.startsWith("delete:")));
  assert.match(content(tree), /Delete alice\?/);
  assert.match(content(tree), /revokes alice's sessions/);
  assert.match(content(tree), /renamed aside, never deleted/);
  await confirm(tree);
  assert.ok(ui.calls.includes("delete:alice-id"));
});

test("desktop password control reaches managed accounts and the owner copy names the mode", () => {
  const general = readFileSync(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  // Web keeps the row for everyone; on desktop only the owner is served by Remote access.
  assert.match(general, /\{isTauri && isOwner \? null : \(/);
  const remote = readFileSync(
    new URL(
      "../src/features/settings/components/remote-access-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(remote, /const multi = useLoginMode\(\) === "multi";/);
  assert.doesNotMatch(
    remote,
    /description="Remote browsers sign in as unsloth/,
  );
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
  assert.match(dialog, /settingsTabVisible\(tab\.id, isOwner\)/);
  assert.match(dialog, /resolveSettingsTab\(deferredTab, isOwner\)/);
  assert.equal((dialog.match(/visibleTabs\.map/g) ?? []).length, 2);
});

test("a failed reset keeps the confirmation open and shows the error inside it", async () => {
  const ui = tab(true, "The account could not be reset");
  await click(await ui.initialize(), "Regenerate setup code");
  await confirm(ui.render());
  const tree = ui.render();
  const dialog = nodes(tree).find((node) => node.type === "AlertDialog");
  assert.equal(dialog?.props.open, true);
  const alert = nodes(dialog).find((node) => node.props.role === "alert");
  assert.equal(content(alert), "The account could not be reset");
  const action = nodes(dialog).find(
    (node) => node.type === "AlertDialogAction",
  );
  assert.equal(action?.props.disabled, false);
});

test("accounts show creation dates and username search ignores case and surrounding spaces", async () => {
  const ui = tab();
  const tree = await ui.initialize();
  assert.equal(
    nodes(tree).find((node) => node.type === "time")?.props.dateTime,
    "2026-09-01T12:00:00Z",
  );
  const search = nodes(tree).find(
    (node) => node.props["aria-label"] === "Search accounts",
  );
  assert.ok(search);
  (search.props.onChange as (event: unknown) => void)({
    target: { value: "  ALIce  " },
  });
  const filtered = nodes(ui.render());
  assert.ok(
    filtered.some((node) => node.props["data-testid"] === "account-alice"),
  );
  assert.ok(
    !filtered.some((node) => node.props["data-testid"] === "account-unsloth"),
  );
  (search.props.onChange as (event: unknown) => void)({
    target: { value: "missing" },
  });
  assert.match(content(ui.render()), /No matching accounts/);
  assert.doesNotMatch(content(ui.render()), /No other accounts yet/);
});

test("a one-time setup code can always be dismissed, even while its refresh is still in flight", async () => {
  // `perform` stays busy through the account refresh that follows the mutation,
  // and the code is already on screen by then. Gating dismissal on busy left the
  // plaintext credential with no exit at all on a slow backend: Done disabled,
  // Escape swallowed by the same guard, and the dialog has no close button.
  const ui = tab(true, null, { stallRefresh: true });
  let tree = await ui.initialize();
  await click(tree, "Create account");
  const input = nodes(ui.render()).find(
    (node) => node.props.id === "new-account-username",
  );
  assert.ok(input);
  (input.props.onChange as (event: unknown) => void)({
    target: { value: "bob" },
  });
  const form = nodes(ui.render()).find((node) => node.type === "form");
  assert.ok(form);
  (form.props.onSubmit as (event: unknown) => void)({ preventDefault() {} });
  await tick();
  await tick();

  tree = ui.render();
  assert.match(content(tree), /one-time-secret/, "the code should be showing");
  const done = nodes(tree).find(
    (node) => node.type === "Button" && content(node).trim() === "Done",
  );
  assert.ok(done);
  assert.notEqual(done.props.disabled, true, "Done must stay usable");
  (done.props.onClick as () => void)();
  await tick();
  assert.doesNotMatch(content(ui.render()), /one-time-secret/);

  // Escape and the overlay route through the same handler, so they must work too.
  const ui2 = tab(true, null, { stallRefresh: true });
  let tree2 = await ui2.initialize();
  await click(tree2, "Create account");
  const input2 = nodes(ui2.render()).find(
    (node) => node.props.id === "new-account-username",
  );
  (input2!.props.onChange as (event: unknown) => void)({
    target: { value: "bob" },
  });
  const form2 = nodes(ui2.render()).find((node) => node.type === "form");
  (form2!.props.onSubmit as (event: unknown) => void)({ preventDefault() {} });
  await tick();
  await tick();
  tree2 = ui2.render();
  assert.match(content(tree2), /one-time-secret/);
  const dialog = nodes(tree2).find((node) => node.type === "Dialog");
  (dialog!.props.onOpenChange as (open: boolean) => void)(false);
  await tick();
  assert.doesNotMatch(content(ui2.render()), /one-time-secret/);
  ui.releaseRefresh();
  ui2.releaseRefresh();
});

test("an account the API reports without a creation date shows a dash, not the epoch", async () => {
  // created_at is added as a NULLABLE column on upgrade and the backfill skips
  // rows that already carry an account_id, so the API can answer with null.
  // `new Date(null)` is the epoch rather than an invalid date, so a bare NaN
  // check prints a confident "Jan 1, 1970" for an account nobody created then.
  const ui = tab(true, null, {
    accounts: [
      {
        account_id: "owner",
        username: "unsloth",
        role: "owner",
        is_active: true,
        created_at: null,
      },
      {
        account_id: "alice-id",
        username: "alice",
        role: "user",
        is_active: true,
        created_at: "2026-09-01T12:00:00Z",
      },
    ],
  });
  const tree = await ui.initialize();
  const ownerRow = nodes(tree).find(
    (node) => node.props["data-testid"] === "account-unsloth",
  );
  assert.ok(ownerRow);
  assert.equal(content(nodes(ownerRow).find((node) => node.type === "time")), "—");
  assert.doesNotMatch(content(tree), /1970/);
  const aliceRow = nodes(tree).find(
    (node) => node.props["data-testid"] === "account-alice",
  );
  assert.match(
    content(nodes(aliceRow!).find((node) => node.type === "time")),
    /2026/,
    "a real timestamp must still render",
  );
});
