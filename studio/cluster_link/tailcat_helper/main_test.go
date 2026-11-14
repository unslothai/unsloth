// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

package main

import (
	"net"
	"testing"

	"github.com/tailscale/tailcat"
)

func TestAClosedClientRefusesNewForwards(t *testing.T) {
	c := &client{tc: &tailcat.Client{}, forwards: map[uint16]net.Listener{}}
	c.close()
	if port, err := c.forward(50070); err == nil {
		t.Fatalf("forward on a closed client opened local port %d", port)
	}
	if len(c.forwards) != 0 {
		t.Fatalf("closed client kept %d listeners", len(c.forwards))
	}
}
