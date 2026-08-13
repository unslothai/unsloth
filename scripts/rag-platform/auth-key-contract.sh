#!/usr/bin/env bash

set -euo pipefail

container_name="${1:-rag-platform-backend}"
private_key="/ragflow/conf/private.pem"
public_key="/ragflow/conf/public.pem"

if ! docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -qx true; then
  echo "auth key contract failed: runtime container is not running" >&2
  exit 1
fi

if ! docker exec "${container_name}" sh -c \
  "grep -q '^-----BEGIN ENCRYPTED PRIVATE KEY-----$' '${private_key}'"; then
  echo "auth key contract failed: private key is not encrypted PKCS#8" >&2
  exit 1
fi

docker exec "${container_name}" /ragflow/.venv/bin/python -c \
  "from Crypto.PublicKey import RSA; key=RSA.import_key(open('${private_key}','rb').read(),'Welcome'); assert key.has_private()"

private_public_hash="$(
  docker exec "${container_name}" sh -c \
    "openssl pkey -passin pass:Welcome -in '${private_key}' -pubout -outform DER 2>/dev/null | sha256sum | cut -d' ' -f1"
)"
public_hash="$(
  docker exec "${container_name}" sh -c \
    "openssl pkey -pubin -in '${public_key}' -outform DER 2>/dev/null | sha256sum | cut -d' ' -f1"
)"

if [[ -z "${private_public_hash}" || "${private_public_hash}" != "${public_hash}" ]]; then
  echo "auth key contract failed: public/private key pair does not match" >&2
  exit 1
fi

echo "auth key contract passed (encrypted PKCS#8; matching public key)"
