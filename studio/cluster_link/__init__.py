# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Home-only cluster networking: pairing plus llama.cpp RPC over mutual TLS."""

AGENT_PORT_DEFAULT = 50070
PAIR_PORT_DEFAULT = 50071
RPC_PORT_DEFAULT = 50052
NATIVE_RPC_PORT_DEFAULT = 50053
DISCOVERY_GROUP_V4 = "239.255.77.77"
DISCOVERY_GROUP_V6 = "ff02::1"
DISCOVERY_PORT = 50079
PROTOCOL_VERSION = 1

SECURE_RPC_ENV = {"GGML_RPC_NO_RDMA": "1"}
