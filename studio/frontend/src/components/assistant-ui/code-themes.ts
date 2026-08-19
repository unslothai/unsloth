// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import oneDarkPro from "@shikijs/themes/one-dark-pro";
import oneLight from "@shikijs/themes/one-light";
import type { ThemeRegistrationAny } from "shiki";

// Canonical Atom One Dark / One Light themes from `@shikijs/themes`. Only the
// background is overridden so the code block blends into the app's `--code-block`
// surface; all token colors/scopes are kept intact for consistent multi-language
// highlighting out of the box.
const withTransparentBg = (theme: ThemeRegistrationAny): ThemeRegistrationAny => ({
  ...theme,
  bg: "transparent",
  colors: {
    ...theme.colors,
    "editor.background": "transparent",
  },
});

export const unslothLightTheme: ThemeRegistrationAny = {
  ...withTransparentBg(oneLight),
  name: "unsloth-light",
};

export const unslothDarkTheme: ThemeRegistrationAny = {
  ...withTransparentBg(oneDarkPro),
  name: "unsloth-dark",
};
