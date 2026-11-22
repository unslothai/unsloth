// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";

const HUB_INVENTORY_ID = "cache:safetensors:mlx-community%2FQwen3-0.6B-4bit";

function buildRouter() {
  const rootRoute = createRootRoute({
    beforeLoad: async () => {
      await Promise.resolve();
    },
    component: () => null,
  });
  const page = (path: "/chat" | "/hub" | "/images") =>
    createRoute({
      getParentRoute: () => rootRoute,
      path,
      validateSearch: (search: Record<string, unknown>) => ({
        ...(typeof search.model === "string" ? { model: search.model } : {}),
      }),
      component: () => null,
    });
  return createRouter({
    routeTree: rootRoute.addChildren([
      page("/chat"),
      page("/hub"),
      page("/images"),
    ]),
    history: createMemoryHistory({ initialEntries: ["/chat"] }),
  });
}

function matchFor(router: ReturnType<typeof buildRouter>, routeId: string) {
  return router.state.matches.find((match) => match.routeId === routeId);
}

async function settleOnHub(router: ReturnType<typeof buildRouter>) {
  await router.load();
  await router.navigate({ to: "/hub", search: { model: HUB_INVENTORY_ID } });
  await router.invalidate();
}

test("the root match hands the hub's selection to every persistently mounted page", async () => {
  const router = buildRouter();
  await settleOnHub(router);

  assert.equal(router.state.matches[0]?.routeId, "__root__");
  assert.equal(
    (router.state.matches[0]?.search as { model?: string }).model,
    HUB_INVENTORY_ID,
  );
});

test("location.pathname reaches /images while the hub's search is still committed", async () => {
  const router = buildRouter();
  await settleOnHub(router);

  const navigation = router.navigate({ to: "/images" });
  await Promise.resolve();

  assert.equal(router.state.location.pathname, "/images");
  assert.equal(
    (router.state.matches[0]?.search as { model?: string }).model,
    HUB_INVENTORY_ID,
  );

  await navigation;
});

test("no /images match exists to read a model from until the navigation commits", async () => {
  const router = buildRouter();
  await settleOnHub(router);
  assert.equal(matchFor(router, "/images"), undefined);

  const navigation = router.navigate({ to: "/images" });
  await Promise.resolve();
  assert.equal(matchFor(router, "/images"), undefined);

  await navigation;
  assert.notEqual(matchFor(router, "/images"), undefined);
  assert.deepEqual(matchFor(router, "/images")?.search, {});
});

test("a chat-picker handoff still arrives on the /images match", async () => {
  const router = buildRouter();
  await router.load();
  await router.navigate({
    to: "/images",
    search: { model: "unsloth/Z-Image-Turbo" },
  });
  await router.invalidate();

  assert.deepEqual(matchFor(router, "/images")?.search, {
    model: "unsloth/Z-Image-Turbo",
  });
});
