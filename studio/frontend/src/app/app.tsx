// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { RouterProvider } from "@tanstack/react-router";
import { router } from "./router";

export function App() {
  return <RouterProvider router={router} />;
}
