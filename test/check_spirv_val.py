#!/usr/bin/env python3
"""spirv-val every SPIR-V module the MLIR backend emits.

Usage: check_spirv_val.py --mlirc <vc> [--spirv-val <path>] <file or dir>...

GPU end-to-end tests need a Vulkan device, which CI does not have, so nothing
there ever looks at the bytes `vc -emit=spirv` writes. This is the closest
device-free check that those bytes are the contract: spirv-val applies the same
structural rules the driver applies at vkCreateShaderModule (capabilities
against the memory model, entry-point interface lists, OpBitcast on logical
pointers, pointer-typed variables under logical addressing), so a lowering that
produces an unloadable module fails here instead of on whoever has a GPU.

Every input .vc is compiled; a file the driver refuses to compile is not a
validation result (those are the frontend / negative IR tests, covered by
vc-check and mlir-check) and is reported as skipped.

Exit status is non-zero when a module fails validation and is not listed in
KNOWN_INVALID, or when a KNOWN_INVALID entry now validates — the exception list
has to say what is actually broken, not what was once.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

# Files whose emitted module is invalid SPIR-V for a reason the lowering still
# has, keyed by repo-relative path, with the reason.
KNOWN_INVALID = {
    # A `T&`-returning __device__ function has to hand the caller a pointer
    # value: the callee's `scf.if` yields a `spirv.ptr`, and MLIR converts that
    # into a Function-storage OpVariable *holding a pointer* (store the chosen
    # address in one branch, load it back after the merge). SPIR-V's logical
    # addressing forbids pointer-valued variables — and pointers as function
    # return values — outright, so the module cannot validate. MoltenVK compiles
    # and runs it anyway. The legal shape is what the GLSL backend does with the
    # same source: inline the call and let each branch carry the store.
    "test/ref_return.vc": "pointer-typed Function variable from a ref-return",
    "test/Frontend/ref-return.vc": "same source shape as test/ref_return.vc",
}


def collect(paths):
    files = []
    for p in paths:
        path = pathlib.Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.vc")))
        else:
            files.append(path)
    return files


def known_invalid_key(path):
    """The KNOWN_INVALID entry for `path`, matched by suffix.

    The cmake target runs from the build directory and CI from the checkout, so
    the same file arrives as `test/ref_return.vc` or `/src/test/ref_return.vc`.
    The exception is about the file, not about how it was spelled to us.
    """
    posix = path.as_posix()
    for key in KNOWN_INVALID:
        if posix == key or posix.endswith("/" + key):
            return key
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mlirc", required=True, help="path to the vc (MLIR) driver")
    ap.add_argument("--spirv-val", default="spirv-val", help="spirv-val binary")
    ap.add_argument("paths", nargs="+", help=".vc files or directories")
    args = ap.parse_args(argv[1:])

    val = args.spirv_val
    if subprocess.run([val, "--version"], capture_output=True).returncode != 0:
        print(f"check_spirv_val: '{val}' is not runnable; install spirv-tools",
              file=sys.stderr)
        return 2

    checked = unexpected = still_invalid = outdated = skipped = 0
    observed = set()
    with tempfile.TemporaryDirectory() as tmp:
        for f in collect([pathlib.Path(p) for p in args.paths]):
            listed = known_invalid_key(f)
            out = pathlib.Path(tmp) / (f.as_posix().replace("/", "_") + ".spv")
            compile_proc = subprocess.run(
                [args.mlirc, "-emit=spirv", str(f), "-o", str(out)],
                capture_output=True, text=True)
            if compile_proc.returncode != 0 or not out.exists():
                skipped += 1
                continue
            checked += 1
            proc = subprocess.run([val, str(out)], capture_output=True, text=True)
            if proc.returncode != 0:
                if listed:
                    observed.add(listed)
                    still_invalid += 1
                else:
                    reason = (proc.stderr.strip().splitlines() or [""])[0]
                    print(f"FAIL: {f}: {reason}", file=sys.stderr)
                    unexpected += 1
            elif listed:
                print(f"FAIL: {f} is listed as known-invalid but now validates; "
                      f"drop it from KNOWN_INVALID", file=sys.stderr)
                outdated += 1

    print(f"{checked} module(s) validated, {still_invalid} accepted as known "
          f"invalid, {skipped} file(s) not compiled to SPIR-V")
    if observed:
        print("known-invalid (fix the lowering; do not add to this list):")
        for key in sorted(observed):
            print(f"  {key}: {KNOWN_INVALID[key]}")
    return 1 if (unexpected or outdated) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
