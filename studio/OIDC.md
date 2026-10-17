<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0 -->

# OpenID Connect SSO for Unsloth Studio

Unsloth Studio can use a standards-compatible OpenID Connect provider in addition to its existing
local username/password authentication. Keycloak is the reference provider. LDAP and Active
Directory remain behind the identity provider; Studio does not connect to either directly.

## Environment configuration

```dotenv
UNSLOTH_OIDC_ENABLED=true
UNSLOTH_OIDC_ISSUER=https://keycloak.example.com/realms/company
UNSLOTH_OIDC_CLIENT_ID=unsloth
UNSLOTH_OIDC_CLIENT_SECRET=replace-with-the-confidential-client-secret
UNSLOTH_OIDC_SCOPES=openid profile email
UNSLOTH_OIDC_REDIRECT_URI=https://unsloth.example.com/api/auth/oidc/callback
UNSLOTH_OIDC_DISPLAY_NAME=Company SSO
UNSLOTH_OIDC_AUTO_CREATE_USERS=true
UNSLOTH_OIDC_USERNAME_CLAIM=preferred_username
UNSLOTH_OIDC_EMAIL_CLAIM=email
```

`UNSLOTH_OIDC_ISSUER` must exactly match the `issuer` in the provider's discovery document. Do not
append `/.well-known/openid-configuration`; Studio adds that path.

Optional group restriction:

```dotenv
# Comma-separated. When non-empty, at least one value must appear in the ID token's groups claim.
UNSLOTH_OIDC_ALLOWED_GROUPS=

```

An automatically created account always receives the normal `user` role. No OIDC claim grants
installation-owner rights.

## Docker Compose example

```yaml
services:
  unsloth-studio:
    image: your-unsloth-studio-image:latest
    environment:
      UNSLOTH_OIDC_ENABLED: "true"
      UNSLOTH_OIDC_ISSUER: "https://keycloak.example.com/realms/company"
      UNSLOTH_OIDC_CLIENT_ID: "unsloth"
      UNSLOTH_OIDC_CLIENT_SECRET: "${UNSLOTH_OIDC_CLIENT_SECRET}"
      UNSLOTH_OIDC_SCOPES: "openid profile email"
      UNSLOTH_OIDC_REDIRECT_URI: "https://unsloth.example.com/api/auth/oidc/callback"
      UNSLOTH_OIDC_DISPLAY_NAME: "Company SSO"
      UNSLOTH_OIDC_AUTO_CREATE_USERS: "true"
    volumes:
      - unsloth-state:/root/.unsloth

volumes:
  unsloth-state:
```

Keep the client secret in a secret manager or uncommitted `.env` file. Do not bake it into an image.

## Minimal Keycloak configuration

1. Create or select realm `company`.
2. Create client `unsloth` with protocol **OpenID Connect**.
3. Enable **Client authentication** and **Standard flow**.
4. Disable Direct Access Grants unless another application requires them.
5. Set the valid redirect URI to
   `https://unsloth.example.com/api/auth/oidc/callback`.
6. Set Web Origins to `https://unsloth.example.com`.
7. Assign the `openid`, `profile`, and `email` client scopes.
8. Copy the confidential client secret to `UNSLOTH_OIDC_CLIENT_SECRET`.

The ID token must contain the standard `iss`, `sub`, `aud`, `exp`, `iat`, and `nonce` claims. The
default profile mapping also uses `preferred_username`, `email`, and optionally `name`.

For LDAP or Active Directory, configure Keycloak User Federation. Studio only sees the resulting
OIDC identity.

## Account mapping and first login

Studio maps an external identity using `(issuer, subject)`. It never automatically links by email
or username. On first login:

1. Studio validates the authorization state, PKCE exchange, ID-token signature, issuer, audience,
   expiration, authorized party where applicable, nonce, and subject.
2. If `(issuer, subject)` exists, its existing internal account is used.
3. Otherwise, when `UNSLOTH_OIDC_AUTO_CREATE_USERS=true`, Studio creates a normal internal account
   and the mapping in one transaction.
4. A local Studio access token and refresh token are issued through the existing session system.

If the preferred username already exists, Studio creates a deterministic suffixed username. It does
not link to the existing account.

Set `UNSLOTH_OIDC_AUTO_CREATE_USERS=false` to allow only identities already mapped in `auth.db`.

## Local login and fallback

The password form remains available when OIDC is enabled. OIDC provider downtime therefore does not
remove the local fallback. The installation owner should maintain and protect the local owner
credential for recovery.

To disable SSO, set:

```dotenv
UNSLOTH_OIDC_ENABLED=false
```

Restart Studio. Local accounts, passwords, API keys, sessions, and external identity mappings remain
unchanged. The SSO button disappears.

## Upgrade and database migration

Back up the Studio state directory, including `auth/auth.db`, before upgrading. On first database
access, Studio creates the additive `external_identities` table and index with `IF NOT EXISTS`.
Existing account rows are not rewritten or deleted. Downgrading leaves the extra table unused.

Do not delete `auth.db` during an upgrade; it contains account identities, session signing secrets,
API-key material, and the OIDC mappings.

## Security notes

- Use HTTPS for Studio and the provider in production.
- The confidential client secret remains backend-only and is never returned by public configuration.
- Authorization state and nonce are random, short-lived, bounded, and single-use.
- Authorization Code Flow uses PKCE S256 in addition to confidential-client authentication.
- Discovery, token, and JWKS requests have timeouts and do not follow redirects.
- Only asymmetric ID-token algorithms are accepted; unsigned and symmetric provider tokens are
  rejected.
- Discovery and JWKS documents are cached; signing keys are refreshed once when validation fails.
- Session credentials are not placed in callback URLs. The callback uses a short-lived one-time
  handoff code, then reuses Studio's existing access/refresh-token system.
- Studio logout revokes local refresh tokens. It does not yet invoke the provider's
  `end_session_endpoint`, so the provider SSO session may remain active.
- Access tokens and refresh tokens are never intentionally logged. Configure reverse proxies to
  avoid logging sensitive request bodies.
