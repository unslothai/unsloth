// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import { decodeJwtSubject } from "../src/features/profile/utils/jwt-subject.ts";
import { initialsFromName } from "../src/features/profile/utils/avatar-initials.ts";

test("account identity supplies the sidebar name and avatar unless a display name is set", () => {
  let token: string | null = null;
  const profile = { displayName: "", nickname: "", avatarDataUrl: null };
  const { useEffectiveProfile: renderProfile } = loadWithStubs<{
    useEffectiveProfile: () => {
      displayTitle: string;
      addressName: string;
      sessionSub: string | null;
    };
  }>(
    new URL(
      "../src/features/profile/hooks/use-effective-profile.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": { getAuthToken: () => token, OWNER_USERNAME: "unsloth" },
      "../utils/jwt-subject": { decodeJwtSubject },
      "../stores/user-profile-store": {
        useUserProfileStore: (select: (state: typeof profile) => unknown) =>
          select(profile),
      },
    },
  );

  // A managed account shows the username its owner chose, verbatim. The owner's own
  // id is the reserved literal "unsloth", which is the product name in lower case, so
  // that ONE id displays as "Unsloth" rather than spelling the brand wrongly on every
  // default install. sessionSub keeps the real subject either way.
  for (const [username, shown] of [
    ["unsloth", "Unsloth"],
    ["alice", "alice"],
    ["bob", "bob"],
  ]) {
    token = `test.${Buffer.from(JSON.stringify({ sub: username })).toString("base64url")}.test`;
    const effective = renderProfile();
    assert.equal(effective.displayTitle, shown);
    assert.equal(effective.addressName, shown);
    assert.equal(effective.sessionSub, username);
    assert.equal(
      initialsFromName(effective.displayTitle),
      username[0].toUpperCase(),
    );
  }

  profile.displayName = "  Robert Smith  ";
  profile.nickname = "Rob";
  assert.equal(renderProfile().displayTitle, "Robert Smith");
  assert.equal(renderProfile().addressName, "Rob");

  profile.displayName = "  ";
  assert.equal(renderProfile().displayTitle, "bob");
  for (const invalidToken of [null, "invalid-token"]) {
    token = invalidToken;
    assert.equal(renderProfile().displayTitle, "Unsloth");
  }
});
