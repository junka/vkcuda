#!/bin/sh
# Gate for the vc-e2e-check target: fail fast when there is no Vulkan device,
# so we don't build 60 executables before the first run errors out with a
# device-init message. The e2e suite is GPU-dependent.
#
# Exit codes: 0 = device present, proceed; 1 = no device or vulkaninfo unusable.
set -e
: "${VC_VULKANINFO:=vulkaninfo}"
if ! command -v "$VC_VULKANINFO" >/dev/null 2>&1; then
  echo "vc-e2e-check: vulkaninfo not found; cannot verify a Vulkan device is present"
  exit 1
fi
if "$VC_VULKANINFO" --summary 2>/dev/null | grep -q '^GPU'; then
  exit 0
fi
echo "vc-e2e-check: no Vulkan device found (vulkaninfo reports none); disable with -DVC_RUN_E2E_TESTS=OFF on GPU-less machines"
exit 1
