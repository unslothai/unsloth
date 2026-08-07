import assert from "node:assert/strict";
import test from "node:test";

import { schemaDeclaresExpectedTitle } from "../src/features/chat/utils/openapi-support.ts";

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
