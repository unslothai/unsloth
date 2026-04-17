// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LightRays } from "@/components/ui/light-rays";
import { Card } from "@/components/ui/card";
import { AuthForm } from "./components/auth-form";

export function ChangePasswordPage() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background px-4 py-8 sm:px-6 sm:py-10 md:px-10">
      <LightRays
        count={6}
        color="rgba(34, 197, 94, 0.25)"
        blur={34}
        speed={15}
        length="70vh"
        style={{ opacity: 0.4 }}
      />
      <Card className="relative z-10 w-full max-w-sm px-5 py-6 shadow-border ring-1 ring-border sm:px-6 sm:py-8">
        <AuthForm mode="change-password" />
      </Card>
    </div>
  );
}
