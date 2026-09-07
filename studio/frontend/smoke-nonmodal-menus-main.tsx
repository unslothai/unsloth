// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_nonmodal_menus.py: the real NonModalDropdownMenu on a real
// overflowing list, so the behaviour measured is the component's own. No backend, no auth.

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import { type ReactElement, StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// Counts document listeners so a leak across opens is measurable. Before React, to see them all.
const listenerCounts: Record<string, number> = {};
const realAdd = document.addEventListener.bind(document);
const realRemove = document.removeEventListener.bind(document);
document.addEventListener = ((type: string, fn: never, opts: never) => {
  listenerCounts[type] = (listenerCounts[type] ?? 0) + 1;
  return realAdd(type, fn, opts);
}) as typeof document.addEventListener;
document.removeEventListener = ((type: string, fn: never, opts: never) => {
  listenerCounts[type] = (listenerCounts[type] ?? 0) - 1;
  return realRemove(type, fn, opts);
}) as typeof document.removeEventListener;

// The guard swallows in the capture phase, so a swallowed click never reaches this one.
let documentClicks = 0;
document.addEventListener("click", () => {
  documentClicks += 1;
});

const ROWS = Array.from({ length: 60 }, (_, index) => index);
// Named once so every list below maps over identities, not over positions.
const MENU_ITEMS = Array.from({ length: 40 }, (_, index) => `item-${index}`);

function RowMenu({ row }: { row: number }): ReactElement {
  return (
    <NonModalDropdownMenu
      side="bottom"
      align="start"
      sideOffset={0}
      className="w-56"
      trigger={(triggerRef) => (
        <button
          ref={triggerRef}
          type="button"
          data-row={String(row)}
          aria-label={`Row options ${row}`}
        >
          options
        </button>
      )}
    >
      <DropdownMenuItem data-testid={`rename-${row}`}>Rename</DropdownMenuItem>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger data-testid={`export-${row}`}>
          Export
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className="max-h-40">
          {MENU_ITEMS.map((id) => (
            <DropdownMenuItem key={id} data-testid={`export-${row}-${id}`}>
              Format {id}
            </DropdownMenuItem>
          ))}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <DropdownMenuSeparator />
      <DropdownMenuItem data-testid={`delete-${row}`}>Delete</DropdownMenuItem>
    </NonModalDropdownMenu>
  );
}

/** The unconverted shape, as a control: whatever this does too is Radix, not the wrapper. */
function ControlRowMenu(): ReactElement {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button type="button" aria-label="Control options">
          control
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="start" className="w-56">
        <DropdownMenuItem data-testid="control-rename">Rename</DropdownMenuItem>
        <DropdownMenuItem data-testid="control-delete">Delete</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** A menu whose own content overflows, so scrolling its viewport is a real scroll event. */
function TallMenu(): ReactElement {
  return (
    <NonModalDropdownMenu
      side="bottom"
      align="start"
      className="max-h-40 w-56"
      trigger={(triggerRef) => (
        <button ref={triggerRef} type="button" aria-label="Tall menu">
          tall
        </button>
      )}
    >
      {MENU_ITEMS.map((id) => (
        <DropdownMenuItem key={id} data-testid={`tall-${id}`}>
          Item {id}
        </DropdownMenuItem>
      ))}
    </NonModalDropdownMenu>
  );
}

function Harness(): ReactElement {
  const [rows, setRows] = useState<number[]>(ROWS);

  useEffect(() => {
    const api = {
      /** Drop a row while its menu is open, to strand the portal if anything can. */
      removeRow(row: number): void {
        setRows((current) => current.filter((value) => value !== row));
      },
      reset(): void {
        setRows(ROWS);
        documentClicks = 0;
      },
      documentClicks: (): number => documentClicks,
      resetClicks(): void {
        documentClicks = 0;
      },
      scrollListeners: (): number => listenerCounts.scroll ?? 0,
      /** Everything a modal Radix menu writes to the document when it opens. */
      documentState: () => ({
        bodyPointerEvents: document.body.style.pointerEvents,
        scrollLocked: document.body.hasAttribute("data-scroll-locked"),
        ariaHidden: document.querySelectorAll("[aria-hidden='true']").length,
        openMenus: document.querySelectorAll("[role='menu']").length,
        activeLabel:
          document.activeElement?.getAttribute("aria-label") ??
          document.activeElement?.tagName ??
          null,
      }),
      supportsPreventScroll: (): boolean => {
        let supported = false;
        const probe = document.createElement("div");
        probe.tabIndex = 0;
        document.body.append(probe);
        probe.focus({
          get preventScroll(): boolean {
            supported = true;
            return true;
          },
        } as FocusOptions);
        probe.remove();
        return supported;
      },
    };
    (window as unknown as { probe: typeof api }).probe = api;
  }, []);

  return (
    <div style={{ display: "flex", gap: 24, padding: 16 }}>
      <div
        id="list"
        style={{
          height: 360,
          width: 260,
          overflowY: "auto",
          border: "1px solid",
        }}
      >
        {rows.map((row) => (
          <div
            key={row}
            data-testid={`row-${row}`}
            style={{
              display: "flex",
              justifyContent: "space-between",
              height: 40,
            }}
          >
            <span>Chat {row}</span>
            <RowMenu row={row} />
          </div>
        ))}
      </div>
      <div
        id="other"
        style={{
          height: 360,
          width: 200,
          overflowY: "auto",
          border: "1px solid",
        }}
      >
        {ROWS.map((row) => (
          <div key={`other-${row}`} style={{ height: 40 }}>
            Other {row}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <TallMenu />
        <input id="outside-input" defaultValue="text" />
        <button id="outside-button" type="button">
          outside
        </button>
        {/* Last, so its modal menu drops below the controls instead of over them. */}
        <ControlRowMenu />
      </div>
      {/* Page-level overflow, so a window scroll is reachable too. */}
      <div style={{ height: 3000, width: 1 }} />
    </div>
  );
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
const strict = !new URLSearchParams(window.location.search).has("nostrict");
createRoot(rootElement).render(
  strict ? (
    <StrictMode>
      <Harness />
    </StrictMode>
  ) : (
    <Harness />
  ),
);
