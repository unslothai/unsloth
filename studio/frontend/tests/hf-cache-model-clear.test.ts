// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  loadWithStubs,
  stubJsxRuntime,
  type StubElement,
} from "./helpers/module-stubs.ts";

type ClearProps = { repoId: string; cachePath: string; disabled?: boolean };
type Delete = (...args: unknown[]) => Promise<void>;

function drive(
  options: { path?: string; disabled?: boolean; remove?: Delete } = {},
) {
  const states: unknown[] = [];
  let cursor = 0;
  let blocked: string[] = [];
  const calls: unknown[][] = [];
  const previews: unknown[][] = [];
  const errors: unknown[][] = [];
  const component = loadWithStubs<{
    HfCacheModelClear: (props: ClearProps) => StubElement;
  }>(
    new URL(
      "../src/features/settings/components/hf-cache-model-clear.tsx",
      import.meta.url,
    ),
    {
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
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/ui/button": { Button: "Button" },
      "@/features/hub/catalog/delete-impact": {
        DeleteImpactSummary: "DeleteImpactSummary",
        useDeleteImpact: (...args: unknown[]) => {
          previews.push(args);
          return { blocked_by: blocked };
        },
      },
      "@/features/hub/catalog/download-card": {
        DeleteConfirmDialog: "DeleteConfirmDialog",
      },
      "@/features/hub/inventory": {
        deleteCachedModel: async (...args: unknown[]) => {
          calls.push(args);
          await options.remove?.(...args);
        },
      },
      "@/features/hub/stores/hf-token-store": {
        useHfTokenStore: (select: (value: { token: string }) => string) =>
          select({ token: "test-hf-token" }),
      },
      "@/i18n": { useT: () => (key: string) => key },
      "@/lib/toast": {
        toast: { error: (...args: unknown[]) => errors.push(args) },
      },
    },
  );
  return {
    calls,
    previews,
    errors,
    block: (models: string[]) => {
      blocked = models;
    },
    render: () => {
      cursor = 0;
      const tree = component.HfCacheModelClear({
        repoId: "Org/Model",
        cachePath: options.path ?? "/cache/hub/models--Org--Model",
        disabled: options.disabled,
      });
      const [button, dialog] = tree.props.children as StubElement[];
      return {
        button,
        dialog,
        open: () => (button.props.onClick as () => void)(),
        confirm: () => (dialog.props.onConfirm as () => void)(),
        close: () =>
          (dialog.props.onOpenChange as (open: boolean) => void)(false),
      };
    },
  };
}

const settle = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};

test("opening and cancelling a confirmation never deletes a model", () => {
  const run = drive();
  assert.equal(run.render().dialog.props.open, false);
  run.render().open();
  assert.equal(run.render().dialog.props.open, true);
  run.render().close();
  assert.equal(run.render().dialog.props.open, false);
  assert.deepEqual(run.calls, []);
});

test("two copies of one repo retain their exact path in preview and deletion", async () => {
  const current = drive({ path: "C:/models/hub/models--Org--Model" });
  const historical = drive({ path: "D:/previous/hub/models--Org--Model" });
  historical.render().open();
  historical.render().confirm();
  await settle();
  assert.deepEqual(historical.previews.at(-1), [
    true,
    "Org/Model",
    null,
    "D:/previous/hub/models--Org--Model",
  ]);
  assert.deepEqual(historical.calls, [
    [
      "Org/Model",
      undefined,
      "test-hf-token",
      "D:/previous/hub/models--Org--Model",
    ],
  ]);
  assert.deepEqual(current.calls, []);
  assert.equal(historical.render().dialog.props.open, false);
});

test("shared-asset impact blocks confirmation even if its handler is called", async () => {
  const run = drive();
  run.block(["Org/Dependent"]);
  run.render().open();
  assert.equal(run.render().dialog.props.blocked, true);
  run.render().confirm();
  await settle();
  assert.deepEqual(run.calls, []);
  assert.equal(run.render().dialog.props.open, true);
});

test("a refresh-disabled clear cannot issue a delete", async () => {
  const run = drive({ disabled: true });
  assert.equal(run.render().button.props.disabled, true);
  run.render().confirm();
  await settle();
  assert.deepEqual(run.calls, []);
});

test("an in-flight delete stays open and cannot be submitted twice", async () => {
  let finish!: () => void;
  const run = drive({
    remove: () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  });
  run.render().open();
  run.render().confirm();
  assert.equal(run.render().dialog.props.deleting, true);
  run.render().confirm();
  run.render().close();
  assert.equal(run.render().dialog.props.open, true);
  assert.equal(run.calls.length, 1);
  finish();
  await settle();
  assert.equal(run.render().dialog.props.open, false);
  assert.equal(run.render().dialog.props.deleting, false);
});

test("a guarded server refusal keeps the selected-copy confirmation open", async () => {
  const run = drive({
    remove: async () => {
      throw new Error("Unload the model before deleting");
    },
  });
  run.render().open();
  run.render().confirm();
  await settle();
  assert.equal(run.render().dialog.props.open, true);
  assert.equal(run.render().dialog.props.deleting, false);
  assert.equal(run.errors.length, 1);
  assert.deepEqual(run.errors[0][1], {
    description: "Unload the model before deleting",
  });
});

test("successful model Clear uses the Hub API's normal inventory refresh and lifecycle events", async () => {
  const requests: { url: string; init?: RequestInit }[] = [];
  const events: string[] = [];
  const hub = loadWithStubs<{ deleteCachedModel: Delete }>(
    new URL("../src/features/hub/inventory/api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return new Response(null, { status: 200 });
        },
      },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": {
        hubTokenHeader: () => ({ "X-HF-Token": "test-hf-token" }),
      },
      "@/lib/format-fastapi-error": {},
      "@/features/hub/lib/local-path": {},
      "@/features/hub/lib/network": {},
      "@/features/hub/lib/token-fingerprint": {},
      "@/features/hub/stores/inventory-events": {
        bumpInventoryVersion: () => events.push("inventory"),
      },
      "@/lib/model-lifecycle-events": {
        notifyModelDeleted: (repo: string) => events.push(`deleted:${repo}`),
      },
      "../download-manager/download-manager-state": {
        discardDeletedModelInventoryHints: () => events.push("hints"),
      },
      "./gguf-variants-cache-events": {
        bumpGgufVariantsCacheVersion: () => events.push("variants"),
      },
    },
  );
  const run = drive({ remove: hub.deleteCachedModel });
  run.render().open();
  run.render().confirm();
  await settle();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/hub/delete-cached");
  assert.equal(requests[0].init?.method, "DELETE");
  assert.deepEqual(JSON.parse(String(requests[0].init?.body)), {
    repo_id: "Org/Model",
    cache_path: "/cache/hub/models--Org--Model",
  });
  assert.deepEqual(events, [
    "hints",
    "inventory",
    "deleted:Org/Model",
    "variants",
  ]);
  assert.equal(run.render().dialog.props.open, false);
});
