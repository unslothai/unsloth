// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain .ts so the node tests can load it.
import { type ComponentProps, createElement } from "react";

/** Links and task checkboxes as plain text, for markdown shown inside another control. */
export const INERT_MARKDOWN_COMPONENTS = {
  a: ({ children }: ComponentProps<"a">) =>
    createElement("span", { className: "text-primary underline decoration-primary/40 underline-offset-2" }, children),
  input: ({ type, checked }: ComponentProps<"input">) =>
    type === "checkbox" ? createElement("span", { "aria-hidden": true }, checked ? "☑ " : "☐ ") : null,
};
