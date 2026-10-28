#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root/web-src"
if ! npm run verify:build --silent; then
  if ! node scripts/build-receipt.mjs --check-install || ! npm ls --all --silent >/dev/null; then
    npm ci --include=dev --no-audit --no-fund
  fi
  npm run build
fi
npm run verify:build --silent
