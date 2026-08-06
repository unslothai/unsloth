


import type { IconSvgElement } from "@hugeicons/react";

// App-wide check mark. Plain tick geometry, sized a touch larger
// than the stock icon so it reads clearly in menus.
export const Tick02Icon: IconSvgElement = [
  [
    "path",
    {
      d: "M4.477 13.299L9.008 17.829L19.123 7.714",
      stroke: "currentColor",
      strokeLinecap: "round",
      strokeLinejoin: "round",
      // Fallback weight, matching the icon set; call sites with a
      // strokeWidth prop override this.
      strokeWidth: "1.5",
      key: "0",
    },
  ],
];
