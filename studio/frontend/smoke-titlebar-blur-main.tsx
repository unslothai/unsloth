// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Production chrome and overlays; deterministic media, no inference or desktop backend.
import { useState, type CSSProperties } from "react";
import { createRoot } from "react-dom/client";
import { WindowTitlebar } from "@/components/tauri/window-titlebar";
import { MediaViewer } from "@/components/media-viewer";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogCancel,
} from "@/components/ui/alert-dialog";
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { GuidedTour } from "@/features/tour/components/guided-tour";
import "./src/index.css";

const art =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540"><rect width="960" height="540" fill="#cce8e5"/><circle cx="725" cy="130" r="65" fill="#f2cb7e"/><path d="M0 360L235 95L510 380L700 200L960 400V540H0Z" fill="#739f8e"/><path d="M0 430L290 265L535 450L770 330L960 450V540H0Z" fill="#345f55"/><path d="M0 480Q250 440 480 485T960 485V540H0Z" fill="#90bbba"/></svg>`,
  );
function Scene() {
  const [kind, setKind] = useState("");
  const [nested, setNested] = useState(false);
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  return (
    <TooltipProvider>
      <div
        className="relative h-dvh overflow-hidden bg-background text-foreground"
        style={
          {
            "--studio-custom-titlebar-height": "40px",
            "--studio-window-chrome-top": "40px",
            "--studio-window-control-inset": "138px",
          } as CSSProperties
        }
      >
        <WindowTitlebar showSidebarSurface />
        <div className="flex h-full pt-10">
          <aside className="w-60 shrink-0 bg-sidebar px-5 py-7">
            <strong className="text-xl">unsloth</strong>
            <div className="mt-8 space-y-5 text-sm">
              <p>New chat</p>
              <p>Model Hub</p>
              <p>Library</p>
              <p>Images</p>
              <p className="rounded-lg bg-accent p-2">Create</p>
            </div>
          </aside>
          <main className="flex-1 p-8">
            <h1 className="mb-2 text-2xl font-medium">Create images</h1>
            <p className="mb-6 text-muted-foreground">
              Deterministic preview fixture · production title bar and media
              viewer
            </p>
            <div className="mb-6 flex gap-3">
              {["media", "dialog", "alert", "sheet", "tour", "scoped"].map((k) => (
                <Button key={k} onClick={() => setKind(k)}>
                  Open {k}
                </Button>
              ))}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button>Open menu</Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem>Menu item</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <img
              data-tour="preview"
              src={art}
              alt="Illustrated mountain lake"
              className="max-h-[60vh] rounded-xl"
            />
          </main>
        </div>
        <div
          ref={setContainer}
          className="absolute inset-x-60 top-48 bottom-8 pointer-events-none"
        />
        <MediaViewer
          open={kind === "media"}
          onOpenChange={(o) => !o && setKind("")}
          title="Illustrated mountain lake"
          meta="Test fixture · 960 × 540"
          noun="image"
          media
          actions={{ onDownload: () => {} }}
        >
          <img
            src={art}
            width={960}
            height={540}
            alt="Illustrated mountain lake preview"
          />
        </MediaViewer>
        <Dialog
          open={kind === "dialog" || kind === "scoped"}
          onOpenChange={(o) => !o && setKind("")}
        >
          <DialogContent
            container={kind === "scoped" ? container : undefined}
            position={kind === "scoped" ? "absolute" : "fixed"}
          >
            <DialogTitle>Example dialog</DialogTitle>
            <DialogDescription>
              Check the title bar while this dialog is open.
            </DialogDescription>
            <Button onClick={() => setNested(true)}>Open nested</Button>
          </DialogContent>
        </Dialog>
        <AlertDialog
          open={kind === "alert" || nested}
          onOpenChange={(o) => {
            if (!o) {
              setNested(false);
              if (kind === "alert") setKind("");
            }
          }}
        >
          <AlertDialogContent>
            <AlertDialogTitle>Confirm action</AlertDialogTitle>
            <AlertDialogDescription>
              Nested modals keep the title bar blurred.
            </AlertDialogDescription>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
          </AlertDialogContent>
        </AlertDialog>
        <Sheet open={kind === "sheet"} onOpenChange={(o) => !o && setKind("")}>
          <SheetContent>
            <SheetTitle>Example sheet</SheetTitle>
            <SheetDescription>
              A viewport backdrop covers the page.
            </SheetDescription>
          </SheetContent>
        </Sheet>
        <GuidedTour
          open={kind === "tour"}
          onOpenChange={(o) => !o && setKind("")}
          steps={[
            {
              id: "preview",
              target: "preview",
              title: "Preview tour",
              body: "The viewport tour backdrop must also cover the titlebar.",
            },
          ]}
          onSkip={() => {}}
          onComplete={() => {}}
        />
      </div>
    </TooltipProvider>
  );
}
// App provider normally publishes these for dialogs portaled to document.body.
document.documentElement.style.setProperty(
  "--studio-custom-titlebar-height",
  "40px",
);
document.documentElement.style.setProperty(
  "--studio-window-chrome-top",
  "40px",
);
createRoot(document.getElementById("root")!).render(<Scene />);
