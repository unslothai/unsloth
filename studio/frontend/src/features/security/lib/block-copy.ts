// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { RemoteCodeBlockCause } from "../types";

export interface BlockCopy {
  title: TranslationKey;
  body: TranslationKey;
  note?: TranslationKey;
}

/** Only a scan that ran may claim findings; every other block says why no check could finish. */
export function blockCopy(
  cause: RemoteCodeBlockCause | null,
  source: string,
  hasProvider: boolean,
): BlockCopy {
  const onModelScope = source === "ModelScope";
  const customCode = hasProvider
    ? "remoteCode.customCodeFromProvider"
    : "remoteCode.customCode";
  switch (cause) {
    case "unsafe_files":
      return {
        title: "remoteCode.unsafeTitle",
        body: hasProvider
          ? "remoteCode.unsafeBodyFromProvider"
          : "remoteCode.unsafeBody",
      };
    case "critical_findings":
      return {
        title: "remoteCode.blockedTitle",
        body: customCode,
        note: "remoteCode.blockedNote",
      };
    case "unscanned_weights":
      return {
        title: "remoteCode.unscannedTitle",
        body: "remoteCode.unscannedBody",
        note: onModelScope
          ? "remoteCode.switchSourceNote"
          : "remoteCode.reconnectNote",
      };
    case "managed_refused":
      return {
        title: "remoteCode.managedTitle",
        body: customCode,
        note: "remoteCode.managedNote",
      };
    case "not_on_source":
      return {
        title: "remoteCode.notFoundTitle",
        body: onModelScope
          ? "remoteCode.notOnModelScopeBody"
          : "remoteCode.notFoundBody",
      };
    default:
      return {
        title: "remoteCode.checksFailedTitle",
        body: "remoteCode.checksFailedBody",
        note: onModelScope ? "remoteCode.switchSourceNote" : undefined,
      };
  }
}
