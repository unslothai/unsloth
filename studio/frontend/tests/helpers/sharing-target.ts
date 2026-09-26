// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isExternalModelId } from "../../src/features/chat/external-providers.ts";
import { isValidRepoId } from "../../src/features/deep-links/parse-deep-link.ts";
import * as identity from "../../src/features/model-picker/model-config/model-identity.ts";
import type * as Target from "../../src/features/model-picker/sharing/target.ts";
import * as localPath from "../../src/lib/local-path.ts";
import { loadWithStubs } from "./module-stubs.ts";

export const { isRunConfigModelInput, resolveRunConfigTarget } = loadWithStubs<
  typeof Target
>(
  new URL("../../src/features/model-picker/sharing/target.ts", import.meta.url),
  {
    "@/features/chat": { isExternalModelId },
    "@/lib/local-path": localPath,
    "../model-config/model-identity": identity,
    "@/features/deep-links": { isValidRepoId },
  },
);

export { isRunConfigVariantUnresolved } from "../../src/features/model-picker/sharing/variant.ts";
