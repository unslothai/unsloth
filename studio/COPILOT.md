# GitHub Copilot subscriptions in Studio

Studio can use each user's GitHub Copilot subscription through the official
[Copilot SDK and GitHub OAuth flow](https://docs.github.com/en/copilot/how-tos/copilot-sdk/setup/github-oauth).
Usage belongs to the signed-in user and remains subject to their plan allowance
and organization policies. Personal subscription connections cannot be shared
with other Studio accounts.

## Connect in Studio

1. Open **Settings → Connections → GitHub Copilot** and choose **Connect in browser**.
2. Studio copies the sign-in code and opens GitHub. Paste the code there, approve
   access, and return to Studio. The sign-in card also has **Copy code** and
   **Open GitHub** buttons.
3. Choose **Start chat** to use the enabled Copilot model in a new conversation.
   To change which models are enabled, use **Models** and select **Save connection**.

## Release and self-hosted setup

Studio needs Python 3.11 or newer and the Copilot SDK installed with the Studio
dependencies. To enable sign-in in official builds, register an OAuth app
under the `unslothai` GitHub organization with **Device Flow** enabled. Set
`GITHUB_COPILOT_CLIENT_ID` in `backend/routes/github_copilot_auth.py` to the
app's public Client ID. The regular Studio build includes this default. Device
Flow uses the public ID alone.

For local testing or a self-hosted override, set
`UNSLOTH_GITHUB_COPILOT_CLIENT_ID` in the Studio backend environment to the
public ID of a device-enabled app. This value takes precedence over the
default in the source.

The SDK downloads its pinned Copilot runtime on first use. The backend needs
network access to GitHub Releases and Copilot services. Requests run in a
separate runtime process with temporary state and an explicit user token;
Studio does not use the server owner's Copilot CLI login.

Tokens and pending device credentials are stored in Studio's encrypted,
account-scoped credential vault. The browser receives the sign-in code and flow
ID. Studio refreshes expiring tokens when GitHub supplies a refresh token and
prompts you to reconnect if authorization expires. **Disconnect locally** removes the saved token and cancels pending sign-in.
To revoke the OAuth grant at GitHub too, remove the app under your GitHub
account's authorized OAuth applications.

## Supported behavior

This integration supports text chat, streaming, model discovery and token usage
reporting. The SDK owns the conversation engine; Studio supplies edited chat
history as a transcript in a fresh request, rather than generating earlier
answers again. System instructions are supplied separately.

Images, Studio tools, JSON mode and sampling/output-cap overrides are currently
unsupported. SDK tools, workspace discovery and host permission requests are
disabled. This connection does not expose Copilot as an unrestricted
OpenAI-compatible proxy and does not use `copilot-api` or private Copilot endpoints.

The SDK's software license does not replace the user's Copilot service agreement.
Use the documented integration within the applicable subscription terms.
