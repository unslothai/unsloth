# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Home-only cluster networking: pair machines once, then carry llama.cpp RPC between them over
mutually authenticated TLS. ``ggml-rpc-server`` stays bound to loopback on every worker and llama.cpp
itself is unchanged: the master's ``llama-server`` talks to a local relay per worker.
"""

AGENT_PORT_DEFAULT = 50070
PAIR_PORT_DEFAULT = 50071
RPC_PORT_DEFAULT = 50052
DISCOVERY_GROUP_V4 = "239.255.77.77"
DISCOVERY_GROUP_V6 = "ff02::1"
DISCOVERY_PORT = 50079
PROTOCOL_VERSION = 1

# Both ends of a secure link set this: GGML_RDMA_GID skips the RDMA address match, which could
# otherwise negotiate RDMA through a loopback relay and move tensors outside TLS.
SECURE_RPC_ENV = {"GGML_RPC_NO_RDMA": "1"}
