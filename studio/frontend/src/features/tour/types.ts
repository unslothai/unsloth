import type { ReactNode } from "react";

export type TourStep = {
  id: string;
  target: string; // data-tour="<target>"
  title: string;
  body: ReactNode;
};

export type Rect = { x: number; y: number; w: number; h: number };

export type Placement = "right" | "left" | "top" | "bottom";

