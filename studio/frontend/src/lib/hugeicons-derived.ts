


// Derived HugeIcons shared between the sidebar and page tabs, so the same visual language appears everywhere.

import { TestTube01Icon } from "@hugeicons/core-free-icons";

// TestTube01Icon's last 2 paths are interior bubbles; slice to the first 3 (outline + cap + liquid line). Original export untouched.
export const TestTubeOutlineIcon = TestTube01Icon.slice(
  0,
  3,
) as typeof TestTube01Icon;
