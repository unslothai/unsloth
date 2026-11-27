// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

// Only an explicit grant shows the notice: a failed claim never saved the switch.
register("./store-stub-resolver.mjs", import.meta.url);
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { claimHubSourceNotice } = await import(
  "../src/features/settings/api/hub-settings.ts"
);

test("only an explicit grant shows the notice", async (t) => {
  t.after(() => setAuthFetchHandler(null));
  const urls: string[] = [];
  const answer = (respond: () => Response) =>
    setAuthFetchHandler((url, init) => {
      urls.push(`${init?.method} ${url}`);
      return respond();
    });

  answer(() => Response.json({ granted: true }));
  assert.equal(await claimHubSourceNotice(), true);
  assert.deepEqual(urls, ["POST /api/settings/hub/source-notice"]);

  for (const refused of [
    () => Response.json({ granted: false }),
    () => Response.json({ detail: "Not Found" }),
    () => Response.json({ granted: true }, { status: 500 }),
    () => {
      throw new Error("network down");
    },
  ]) {
    answer(refused);
    assert.equal(await claimHubSourceNotice(), false);
  }
});

test("the side-effect mount cannot renumber the existing app shell", () => {
  const root = readFileSync(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  assert.ok(
    root.lastIndexOf("<HubSourceNoticeMount />") >
      root.lastIndexOf("</SidebarProvider>"),
    "a null-rendering root sibling must follow the rendered shell so React useId paths stay stable",
  );
});
