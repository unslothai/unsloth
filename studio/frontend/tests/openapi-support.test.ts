import assert from "node:assert/strict";
import test from "node:test";

import {
  readGuardProbe,
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
