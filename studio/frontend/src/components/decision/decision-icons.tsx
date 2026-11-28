// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Decision icons, drawn from the rounded-stroke set the decision UI was
 * specified with (radio-button, shield-question-mark, license-draft).
 *
 * Inline rather than added to the icon package: these are decision-specific and
 * must follow `currentColor` and the UI's stroke weight everywhere they appear.
 */

import type { ReactElement, SVGProps } from "react";
import { cn } from "@/lib/utils";

type IconProps = SVGProps<SVGSVGElement> & { className?: string };

function Icon({ children, className, ...props }: IconProps & { children: ReactElement }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      // Rounded-stroke drawings reach the 0..24 edge; visible overflow keeps a
      // clipped hairline off the top, and geometric precision keeps circles round.
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      shapeRendering="geometricPrecision"
      aria-hidden="true"
      className={cn("shrink-0 overflow-visible", className)}
      {...props}
    >
      {children}
    </svg>
  );
}

/** Yes / no answer space and the criteria pill: a selected radio button. */
export function DecisionCriteriaIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <circle cx="12" cy="12" r="9" />
        <circle cx="12" cy="12" r="4" />
      </>
    </Icon>
  );
}

/** The question itself: a shield with a question mark. */
export function DecisionQuestionIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M9.5 9C9.5 7.61929 10.6193 6.5 12 6.5C13.3807 6.5 14.5 7.61929 14.5 9C14.5 9.85691 14.0689 10.6131 13.4117 11.0636C12.7283 11.5319 12 12.1716 12 13" />
        <path d="M12.125 16.25H12M12.25 16.25C12.25 16.3881 12.1381 16.5 12 16.5C11.8619 16.5 11.75 16.3881 11.75 16.25C11.75 16.1119 11.8619 16 12 16C12.1381 16 12.25 16.1119 12.25 16.25Z" />
        <path d="M18.7088 3.49534C16.8165 2.55382 14.5009 2 12 2C9.4991 2 7.1835 2.55382 5.29116 3.49534C4.36318 3.95706 3.89919 4.18792 3.4496 4.91378C3 5.63965 3 6.34248 3 7.74814V11.2371C3 16.9205 7.54236 20.0804 10.173 21.4338C10.9067 21.8113 11.2735 22 12 22C12.7265 22 13.0933 21.8113 13.8269 21.4338C16.4576 20.0804 21 16.9205 21 11.2371L21 7.74814C21 6.34249 21 5.63966 20.5504 4.91378C20.1008 4.18791 19.6368 3.95706 18.7088 3.49534Z" />
      </>
    </Icon>
  );
}

/** Editing the instructions: a document with a pencil. */
export function DecisionInstructionsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M19.7502 11V10C19.7502 6.22876 19.7502 4.34315 18.5786 3.17157C17.407 2 15.5214 2 11.7502 2H10.7503C6.97907 2 5.09347 2 3.9219 3.17156C2.75033 4.34312 2.75031 6.22872 2.75028 9.99993L2.75024 14C2.75021 17.7712 2.7502 19.6568 3.92173 20.8284C5.0933 21.9999 6.97898 22 10.7502 22" />
        <path d="M7.25024 7H15.2502M7.25024 12H15.2502" />
        <path d="M13.2502 20.8268V22H14.4236C14.833 22 15.0377 22 15.2217 21.9238C15.4058 21.8475 15.5505 21.7028 15.84 21.4134L20.6636 16.5894C20.9366 16.3164 21.0731 16.1799 21.1461 16.0327C21.285 15.7525 21.285 15.4236 21.1461 15.1434C21.0731 14.9961 20.9366 14.8596 20.6636 14.5866C20.3905 14.3136 20.254 14.1771 20.1067 14.1041C19.8265 13.9653 19.4975 13.9653 19.2173 14.1041C19.0701 14.1771 18.9335 14.3136 18.6605 14.5866L13.8369 19.4106C13.5475 19.7 13.4027 19.8447 13.3265 20.0287C13.2502 20.2128 13.2502 20.4174 13.2502 20.8268Z" />
      </>
    </Icon>
  );
}
