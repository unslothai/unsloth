// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The keyless usage example must only be offered for a base the backend can admit.
 *
 * `keyless_api_access._host_authority_is_direct` refuses any `Host` that names something
 * rather than addressing the socket, so keyless works when the caller spelled a literal
 * or `localhost` and not otherwise. `exposure` cannot answer that question: it is
 * computed backend-side from the RESOLVED address, so a launch bound to a hostname
 * (`unsloth studio -H box.local`) reports `private_lan` while every request to
 * `http://box.local:8888` is refused.
 *
 * Before this guard the panel rendered a copy-paste
 *   curl http://box.local:8888/v1/chat/completions -H "Authorization: Bearer not-needed"
 * that answers 401, and the LAN panel offered the same hostname as a QR code.
 *
 * Asserted against the real source rather than a copy, so the two cannot drift.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const SOURCE = "src/features/settings/components/usage-examples.tsx";

test("keylessBaseEligible refuses a hostname base before consulting exposure", () => {
  const source = readFileSync(SOURCE, "utf8");

  // read the function body only, so a comment elsewhere in the file cannot satisfy this
  const start = source.indexOf("function keylessBaseEligible(");
  assert.ok(start > 0, "keylessBaseEligible not found");
  const end = source.indexOf("\n}", start);
  assert.ok(end > start, "could not delimit keylessBaseEligible");
  const body = source.slice(start, end);

  const guard = body.indexOf("if (!isIpLiteralHost(host)) return false;");
  const disjunct = body.indexOf('exposure === "private_lan"');
  const loopback = body.indexOf("if (isLoopbackHost(host)) return true;");

  assert.ok(guard > 0, "the isIpLiteralHost guard is missing from keylessBaseEligible");
  assert.ok(disjunct > 0, "the private_lan disjunct is missing");
  // the guard must run BEFORE the exposure short circuit, which is the clause that made
  // a hostname eligible in the first place
  assert.ok(
    guard < disjunct,
    "the literal guard must run before exposure can make a hostname eligible",
  );
  // and the loopback fast path stays ahead of both, so localhost keeps working
  assert.ok(loopback > 0 && loopback < guard, "localhost must stay eligible");
});
