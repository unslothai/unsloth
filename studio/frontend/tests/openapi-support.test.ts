import assert from "node:assert/strict";
import test from "node:test";

import {
  type GuardProbe,
  readAdvertisedPatchGuards,
  readGuardProbe,
  resolveGuardProbe,
  schemaDeclaresRepairGuards,
} from "../src/features/chat/utils/openapi-support.ts";

function documentWith(properties: Record<string, unknown>) {
  return {
    components: {
      schemas: {
        ChatThreadPatch: { title: "ChatThreadPatch", type: "object", properties },
      },
    },
  };
}

const OLD_PROPERTIES = {
  title: { type: "string" },
  modelType: { type: "string" },
  archived: { type: "boolean" },
};

const GUARDED = {
  ...OLD_PROPERTIES,
  expectedTitle: { type: "string" },
  expectedOpeningMessageId: { type: "string" },
};

test("a backend that declares the fields enforces the guards", () => {
  assert.equal(schemaDeclaresRepairGuards(documentWith(GUARDED)), true);
});

test("a backend from before the fields does not", () => {
  // Old backends drop unknown fields and write anyway, so the migration stays off.
  assert.equal(schemaDeclaresRepairGuards(documentWith(OLD_PROPERTIES)), false);
});

test("half the guards is not enough", () => {
  assert.equal(
    schemaDeclaresRepairGuards(
      documentWith({ ...OLD_PROPERTIES, expectedTitle: { type: "string" } }),
    ),
    false,
  );
});

test("anything unreadable reads as unsupported", () => {
  for (const document of [
    null,
    undefined,
    "",
    42,
    {},
    { components: null },
    { components: {} },
    { components: { schemas: {} } },
    { components: { schemas: { ChatThreadPatch: {} } } },
    { components: { schemas: { ChatThreadPatch: { properties: null } } } },
  ]) {
    assert.equal(schemaDeclaresRepairGuards(document), false);
  }
});

test("a schema that arrived settles the question either way", () => {
  const supported = readGuardProbe(true, documentWith(GUARDED));
  assert.deepEqual(supported, { supported: true, settled: true });

  assert.deepEqual(readGuardProbe(true, documentWith(OLD_PROPERTIES)), {
    supported: false,
    settled: true,
  });
});

test("an HTTP failure is a moment, not an answer", () => {
  // 401/503 are transient, so caching one would park the migration for the session.
  assert.deepEqual(readGuardProbe(false, null), {
    supported: false,
    settled: false,
  });
});

test("a thread list from a current backend advertises the guards", () => {
  assert.equal(
    readAdvertisedPatchGuards({ threads: [], threadPatchGuards: true }),
    true,
  );
  assert.equal(
    readAdvertisedPatchGuards({
      rows: [],
      hasMore: false,
      total: 0,
      threadPatchGuards: false,
    }),
    false,
  );
});

test("an older backend's list, or anything malformed, advertises nothing", () => {
  for (const data of [
    { threads: [] },
    { rows: [], hasMore: false, total: 0 },
    { threadPatchGuards: "true" },
    { threadPatchGuards: 1 },
    { threadPatchGuards: null },
    null,
    undefined,
    [],
    "threadPatchGuards",
  ]) {
    assert.equal(readAdvertisedPatchGuards(data), null, JSON.stringify(data));
  }
});

function countingProbe(result: GuardProbe) {
  const calls: number[] = [];
  return {
    calls,
    probe: async () => {
      calls.push(1);
      return result;
    },
  };
}

test("an advertised flag settles the guard without fetching the schema", async () => {
  for (const advertised of [true, false]) {
    const { calls, probe } = countingProbe({
      supported: !advertised,
      settled: true,
    });
    assert.deepEqual(await resolveGuardProbe(advertised, probe), {
      supported: advertised,
      settled: true,
    });
    assert.equal(calls.length, 0);
  }
});

test("without a flag (older backend) the schema probe decides as before", async () => {
  for (const result of [
    { supported: true, settled: true },
    { supported: false, settled: true },
    { supported: false, settled: false },
  ]) {
    const { calls, probe } = countingProbe(result);
    assert.deepEqual(await resolveGuardProbe(null, probe), result);
    assert.equal(calls.length, 1);
  }
});

test("the flag and the schema probe agree on what a current backend serves", async () => {
  const schema = readGuardProbe(true, documentWith(GUARDED));
  const flag = await resolveGuardProbe(true, async () => {
    throw new Error("must not probe");
  });
  assert.deepEqual(flag, schema);
});
