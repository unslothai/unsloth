#!/usr/bin/env bash

set -euo pipefail

key_dir="${RAG_PLATFORM_KEY_DIR:-/rag-platform/keys}"
private_key="${key_dir}/private.pem"
public_key="${key_dir}/public.pem"

install -d -m 0700 "${key_dir}"

if [[ ! -s "${private_key}" ]]; then
  private_tmp="$(mktemp "${key_dir}/.private.pem.XXXXXX")"
  if ! openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \
    -out "${private_tmp}" >/dev/null 2>&1; then
    rm -f -- "${private_tmp}"
    echo "error: failed to generate Rag Platform runtime private key" >&2
    exit 1
  fi
  chmod 0600 "${private_tmp}"
  mv -f -- "${private_tmp}" "${private_key}"
fi

if ! openssl pkey -in "${private_key}" -check -noout >/dev/null 2>&1; then
  echo "error: invalid private key in ${private_key}" >&2
  exit 1
fi

# Derive the public half on every start so a restored or rotated private key
# can never be paired with stale public material.
public_tmp="$(mktemp "${key_dir}/.public.pem.XXXXXX")"
if ! openssl pkey -in "${private_key}" -pubout -out "${public_tmp}" \
  >/dev/null 2>&1; then
  rm -f -- "${public_tmp}"
  echo "error: failed to derive Rag Platform runtime public key" >&2
  exit 1
fi
chmod 0644 "${public_tmp}"
mv -f -- "${public_tmp}" "${public_key}"

ln -sfn "${private_key}" /ragflow/conf/private.pem
ln -sfn "${public_key}" /ragflow/conf/public.pem

exec /ragflow/entrypoint.sh "$@"
