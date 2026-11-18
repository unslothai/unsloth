#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# build-tags.txt is tailcat's release tag list for the pinned commit.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$(mkdir -p "${1:-$here/dist}" && cd "${1:-$here/dist}" && pwd)"
tags="$(tr -d '[:space:]' < "$here/build-tags.txt")"
version="$(cd "$here" && go list -m -f '{{.Version}}' github.com/tailscale/tailcat)"
cd "$here"
for target in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64; do
    os="${target%/*}"
    arch="${target#*/}"
    exe="unsloth-link"
    [ "$os" = windows ] && exe="unsloth-link.exe"
    mkdir -p "$out/$os-$arch"
    CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" go build -trimpath -buildvcs=false -tags "$tags" \
        -ldflags "-s -w -X main.tailcatVersion=$version" -o "$out/$os-$arch/$exe" .
    cp LICENSE.tailcat "$out/$os-$arch/"
done
(cd "$out" && find . -type f -name 'unsloth-link*' | sort | xargs sha256sum > SHA256SUMS)
cat "$out/SHA256SUMS"
