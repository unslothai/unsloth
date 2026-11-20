"""Header Unsloth's own desktop-app gateways add to the requests they forward to Studio.

Inference routes use it to opt those requests (and only those) into gateway-specific prompt handling.
"""

GATEWAY_CLIENT_HEADER = "X-Unsloth-Client"
GATEWAY_CLIENT_VALUE = "claude-desktop"
