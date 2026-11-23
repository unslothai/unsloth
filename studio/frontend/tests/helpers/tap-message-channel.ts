// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// @assistant-ui/tap creates a module-level MessageChannel for its scheduler.
// A live MessagePort keeps the Node test process alive forever, so record the
// channels and close them after the test file completes.

const Original = globalThis.MessageChannel;
const channels: Array<{ port1: MessagePort; port2: MessagePort }> = [];

if (Original) {
  class TestMessageChannel {
    readonly port1: MessagePort;
    readonly port2: MessagePort;

    constructor() {
      const channel = new Original();
      this.port1 = channel.port1;
      this.port2 = channel.port2;
      channels.push(channel);
    }
  }
  globalThis.MessageChannel =
    TestMessageChannel as unknown as typeof MessageChannel;
}

export function closeTapMessageChannels(): void {
  for (const channel of channels.splice(0)) {
    channel.port1.close();
    channel.port2.close();
  }
}
