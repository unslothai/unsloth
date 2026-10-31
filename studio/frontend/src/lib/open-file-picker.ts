// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Opens the file chooser. Call from a click handler (needs user activation). */
export function openFilePicker(
  accept: string,
  onFiles: (files: File[]) => void,
): void {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.hidden = true;
  if (accept !== "*") {
    input.accept = accept;
  }

  document.body.appendChild(input);
  input.onchange = (event) => {
    const files = (event.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      onFiles(Array.from(files));
    }
    document.body.removeChild(input);
  };
  input.oncancel = () => {
    if (!input.files || input.files.length === 0) {
      document.body.removeChild(input);
    }
  };
  input.click();
}
