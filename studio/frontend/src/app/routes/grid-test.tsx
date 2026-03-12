// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DashboardGrid, DashboardLayout } from "@/components/layout";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/grid-test",
  beforeLoad: () => requireAuth(),
  component: GridTestPage,
});

function GridTestPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-semibold">Grid Test - 3 Columns</h1>
          <p className="text-muted-foreground">
            max-w-7xl, gap-6, responsive 1→2→3
          </p>
        </div>

        <DashboardGrid cols={3}>
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <CardTitle>Card {i}</CardTitle>
                <CardDescription>~400px at 1280px viewport</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-24 rounded-lg bg-muted" />
              </CardContent>
            </Card>
          ))}
        </DashboardGrid>

        <div>
          <h2 className="text-xl font-semibold">4 Columns</h2>
          <p className="text-muted-foreground">~296px per card at 1280px</p>
        </div>

        <DashboardGrid cols={4}>
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} size="sm">
              <CardHeader>
                <CardTitle>Card {i}</CardTitle>
                <CardDescription>Smaller cards</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-16 rounded-lg bg-muted" />
              </CardContent>
            </Card>
          ))}
        </DashboardGrid>
      </div>
    </DashboardLayout>
  );
}
