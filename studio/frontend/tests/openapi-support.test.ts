import assert from "node:assert/strict";
import test from "node:test";

import {
  readGuardProbe,
  schemaDeclaresExpectedTitle,
} from "../src/features/chat/utils/openapi-support.ts";

/** The shape FastAPI serves for the patch model. */
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

test("a backend that declares the field enforces the guard", () => {
  assert.equal(
    schemaDeclaresExpectedTitle(
      documentWith({ ...OLD_PROPERTIES, expectedTitle: { type: "string" } }),
    ),
    true,
  );
});

test("a backend from before the field does not", () => {
  // It drops the unknown field and applies the write, so the migration has to
  // stay off there.
  assert.equal(schemaDeclaresExpectedTitle(documentWith(OLD_PROPERTIES)), false);
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
    assert.equal(schemaDeclaresExpectedTitle(document), false);
  }
});

test("a schema that arrived settles the question either way", () => {
  const supported = readGuardProbe(
    true,
    documentWith({ ...OLD_PROPERTIES, expectedTitle: { type: "string" } }),
  );
  assert.deepEqual(supported, { supported: true, settled: true });

  // An old backend is a real answer, so it is worth remembering.
  assert.deepEqual(readGuardProbe(true, documentWith(OLD_PROPERTIES)), {
    supported: false,
    settled: true,
  });
});

test("an HTTP failure is a moment, not an answer", () => {
  // 401 while the token warms up, 503 during startup. Remembering these would
  // park the migration for the rest of the session.
  assert.deepEqual(readGuardProbe(false, null), {
    supported: false,
    settled: false,
  });
});
