// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "next-themes";
import type { ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      {children}
      <Toaster position="top-right" visibleToasts={2} expand={true} />
    </ThemeProvider>
  );
}
