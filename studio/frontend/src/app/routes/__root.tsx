import { Navbar } from "@/components/navbar";
import {
  Outlet,
  createRootRoute,
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Suspense } from "react";
import { AppProvider } from "../provider";

export const Route = createRootRoute({
  component: RootLayout,
});

const HIDDEN_NAVBAR_ROUTES = ["/onboarding"];

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);

  return (
    <AppProvider>
      {!hideNavbar && <Navbar />}
      <AnimatePresence initial={false}>
        <motion.div
          key={pathname}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.15 }}
          className="flex-1"
        >
          <Suspense fallback={null}>
            <Outlet />
          </Suspense>
        </motion.div>
      </AnimatePresence>
    </AppProvider>
  );
}
