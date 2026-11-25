// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { unloadResident } from "../src/features/api-monitor/unload-resident.ts";

test("declining leaves the resident untouched", async () => {
  const sent: string[] = [];
  const result = await unloadResident({
    readResident: async () => ({ checkpoint: "A", aliases: ["A"] }),
    confirmUnload: async () => false,
    unload: async (id) => {
      sent.push(id);
    },
  });
  assert.deepEqual(sent, []);
  assert.equal(result.declined, true);
  assert.equal(result.stillResident, "A");
});

test("an auto-switched replacement needs its own confirmation", async () => {
  const asked: string[] = [],
    sent: string[] = [];
  let reads = 0;
  const result = await unloadResident({
    readResident: async () => ({
      checkpoint: reads++ === 0 ? "A" : "B",
      aliases: [],
    }),
    confirmUnload: async (id) => {
      asked.push(id);
      return id === "A";
    },
    unload: async (id) => {
      sent.push(id);
    },
  });
  assert.deepEqual(asked, ["A", "B"]);
  assert.deepEqual(sent, ["A"]);
  assert.equal(result.declined, true);
});

test("confirmed unload rechecks that memory is released", async () => {
  let loaded = true;
  const result = await unloadResident({
    readResident: async () =>
      loaded ? { checkpoint: "A", aliases: ["A"] } : null,
    confirmUnload: async () => true,
    unload: async () => {
      loaded = false;
    },
  });
  assert.deepEqual(result, { unloadedAliases: ["A"], stillResident: null });
});


test("failed confirmation or unload propagates without chasing another model", async () => {
  for (const failure of ["confirm", "unload"]) {
    let reads = 0, sends = 0;
    await assert.rejects(unloadResident({
      readResident: async () => { reads++; return {checkpoint:"A", aliases:["A"]}; },
      confirmUnload: async () => { if (failure === "confirm") throw new Error("offline"); return true; },
      unload: async () => { sends++; throw new Error("offline"); },
    }), /offline/);
    assert.equal(reads, 1);
    assert.equal(sends, failure === "confirm" ? 0 : 1);
  }
});

test("an idle backend never asks for confirmation", async () => {
  await unloadResident({readResident: async () => null,
    confirmUnload: async () => { assert.fail("No model to unload"); },
    unload: async () => { assert.fail("No model to unload"); },
  });
});
