#!/usr/bin/env python3
"""Patch an nvidia-pcc pipeline_cache.bin so a Vulkan SC device will accept it.

nvidia-pcc is a *host-side* tool: it runs on x86-64 (it ships inside the plain
NVIDIA driver .run, e.g. NVIDIA-Linux-x86_64-595.99.02) and never on the target,
so it cannot know the target's identity.  Two fields in the 32-byte cache
header therefore come out wrong and the driver rejects the cache with
VK_ERROR_VALIDATION_FAILED (-1000011001) until they are fixed:

  0x0c  deviceID           pcc writes the chip it compiled for; the SC device
                           requires exactly 0 here.
  0x10  pipelineCacheUUID  pcc writes its own UUID; this must equal the
                           device's VkPhysicalDeviceProperties::pipelineCacheUUID
                           exactly (Thor GB10B/581 SC: e251a40589539d03...).

Header layout (little-endian):
  0x00 uint32 headerSize      0x04 uint32 headerVersion
  0x08 uint32 vendorID        0x0c uint32 deviceID
  0x10 byte[16] pipelineCacheUUID
  0x20 .. 0x38  pcc-private words (headerSize is 56, not the 32 of
                VkPipelineCacheHeaderVersionOne); leave them alone.

Usage:
  patch_cache_header.py --in cache.bin --out cache_devid0.bin \
      [--device-uuid e251a40589539d030000000000000000] [--show]

--device-uuid defaults to the Thor GB10B / SC 581 value; read the real one
from VkPhysicalDeviceProperties::pipelineCacheUUID on the target.
"""
import argparse
import struct
import sys

HEADER_SIZE = 0x20
THOR_GB10B_UUID = "e251a40589539d030000000000000000"


def describe(buf):
    if len(buf) < HEADER_SIZE:
        sys.exit("file is shorter than the %d-byte cache header" % HEADER_SIZE)
    u32 = lambda off: struct.unpack_from("<I", buf, off)[0]
    return {
        "headerSize": u32(0x00),
        "headerVersion": u32(0x04),
        "vendorID": u32(0x08),
        "deviceID": u32(0x0C),
        "pipelineCacheUUID": buf[0x10:0x20].hex(),
        # pcc writes headerSize=56, i.e. 24 bytes past the standard
        # VkPipelineCacheHeaderVersionOne prefix.  Those words are pcc-private
        # and their meaning is unknown; report them raw rather than guessing.
        "vendorExtra0": u32(0x20),
        "vendorExtra1": u32(0x24),
        "vendorExtra2": u32(0x28),
        "vendorExtra3": u32(0x2C),
        "fileSize": len(buf),
    }


def patch(buf, device_uuid_hex):
    if len(device_uuid_hex) != 32:
        sys.exit("--device-uuid must be 32 hex chars")
    b = bytearray(buf)
    struct.pack_into("<I", b, 0x0C, 0)                      # deviceID = 0
    b[0x10:0x20] = bytes.fromhex(device_uuid_hex)           # device's cache UUID
    return bytes(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--device-uuid", default=THOR_GB10B_UUID)
    ap.add_argument("--show", action="store_true",
                    help="print the header and exit without writing")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        buf = f.read()

    if args.show or not args.dst:
        for k, v in describe(buf).items():
            print("%-18s %s" % (k, v))
        return

    out = patch(buf, args.device_uuid)
    with open(args.dst, "wb") as f:
        f.write(out)
    print("%s -> %s (%d bytes)" % (args.src, args.dst, len(out)))
    for k in ("deviceID", "pipelineCacheUUID"):
        before, after = describe(buf)[k], describe(out)[k]
        if before != after:
            print("  %s: %s -> %s" % (k, before, after))


if __name__ == "__main__":
    main()
