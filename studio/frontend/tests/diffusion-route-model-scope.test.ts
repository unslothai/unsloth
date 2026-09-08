// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images and Video pages stay mounted off-route and read ?model= to load a pick handed over by the chat picker. /hub names
// its selection with the same param, so these pin both halves of keeping them apart: the trap (location runs ahead of the
// matches, so `active` alone lets the hub's id through) and the fix (no /images match to read one from until it commits).

import assert from "node:assert/strict";
import test from "node:test";

import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";

// What the hub puts in the URL for a downloaded row: an inventory id, not a repo id. Loading it as a diffusion model fails.
const HUB_INVENTORY_ID = "cache:safetensors:mlx-community%2FQwen3-0.6B-4bit";

function buildRouter() {
  const rootRoute = createRootRoute({
    // The real root awaits fetchDeviceType before its chat-only guard.
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

  // A `strict: false` read from a page mounted under the root resolves here.
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

  // `active` is derived from this, so mid-navigation the Images page believes it is the visible one...
  assert.equal(router.state.location.pathname, "/images");
  // ...while the committed matches still describe /hub. Reading the model here is what loaded an inventory id.
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
  // Sidebar navigation carries no search, so the settled route asks for nothing.
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
