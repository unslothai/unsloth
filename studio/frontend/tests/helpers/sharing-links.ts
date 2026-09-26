// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isValidRepoId } from "../../src/features/deep-links/parse-deep-link.ts";
import * as fields from "../../src/features/model-picker/sharing/fields.ts";
import * as linkAddress from "../../src/features/model-picker/sharing/link-address.ts";
import type * as Links from "../../src/features/model-picker/sharing/links.ts";
import { loadWithStubs } from "./module-stubs.ts";

export const { createRunConfigLink, parseRunConfigLink, isShareableModelId } =
  loadWithStubs<typeof Links>(
    new URL(
      "../../src/features/model-picker/sharing/links.ts",
      import.meta.url,
    ),
    {
      "@/features/deep-links": { isValidRepoId },
      "./fields": fields,
      "./link-address": linkAddress,
    },
  );
