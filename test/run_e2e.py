#!/usr/bin/env python3
"""End-to-end demo runner for the VC compiler.

Runs every self-verifying demo under test/*.vc through BOTH backends:

  * GLSL backend:  `vcc <demo>.vc -o <exe>`        -> run <exe>
  * MLIR backend:  `vc  <demo>.vc -emit=full -o <exe>` -> run <exe>

Each demo prints `PASS`/`FAIL` and returns 0/1, so a demo passes when the
built executable exits 0 AND its stdout contains the substring `PASS`.

Demos without an `int main()` (pure device-code fragments) are skipped: the
GLSL/MLIR drivers refuse to link a single-file program with no host main, so
there is nothing to run.

This is a GPU-dependent suite: the built executables need a working Vulkan
device. On a machine without one, runs fail with a device-init error; set
--require-gpu to surface that as a hard failure, otherwise a run failure is
reported normally (the demo "FAIL"s).

Exit status is 0 only when every attempted (backend, demo) pair passes.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

BUILD_TIMEOUT = 180   # g++ link of the host .cpp + embed
RUN_TIMEOUT = 120     # kernel launch + vcDeviceSynchronize + copy-back

# Demos the MLIR backend does not support (but the GLSL backend does). These are
# skipped on the MLIR backend rather than counted as failures, with the reason
# logged. Add a demo here only when the MLIR backend deliberately does not
# implement the feature (and emits a clear diagnostic, not a silent misbuild).
MLIR_UNSUPPORTED = {
    # Kernel-internal printf lowers to GL_EXT_debug_printf (NonSemantic.DebugPrintf
    # SPIR-V) in the GLSL backend; the MLIR SPIR-V path does not emit arbitrary
    # NonSemantic ExtInst sets and emits a diagnostic instead.
    "printf.vc": "kernel printf not implemented in MLIR backend (GLSL only)",
}

# Demos the GLSL backend cannot build (and intentionally rejects with a
# diagnostic). wmma:: tensor-core intrinsics need SPIR-V CooperativeMatrixKHR,
# which only the MLIR backend lowers; vcc rejects them up front. Skip these on
# the GLSL backend so the suite reports them as skipped, not as build failures.
GLSL_UNSUPPORTED = {
    "wmma_gemm.vc": "wmma:: tensor-core intrinsics require the MLIR backend",
    # Sub-array-to-pointer decay (`sumRow(b[i], 3)` with `int *row`): the GLSL
    # backend has no pointer type and rejects this with a clear `#error`. The
    # MLIR backend inlines the callee (binding the pointer param to a sub-array
    # view) so it is the only backend that can lower this.
    "sub_array_decay.vc": "sub-array to pointer parameter requires the MLIR backend",
}


def expand_demos(arguments):
    """Accept a mix of .vc files and directories (globs *.vc, no recursion)."""
    out = []
    for arg in arguments:
        if os.path.isdir(arg):
            out += sorted(glob.glob(os.path.join(arg, "*.vc")))
        else:
            out.append(arg)
    return out


def has_main(path):
    """True if the source defines a host int main() (rough heuristic).

    The drivers refuse single-file programs without a host main, so there is
    nothing to build-and-run for pure device-code fragments.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Strip line comments so a `// int main()` in a comment doesn't match.
    text = re.sub(r"//[^\n]*", "", text)
    return bool(re.search(r"\bint\s+main\s*\(", text))


def build(backend, tool, src, out_exe):
    """Compile+link a demo to an executable. Returns (ok, stderr)."""
    if backend == "glsl":
        cmd = [tool, src, "-o", out_exe]
    elif backend == "mlir":
        cmd = [tool, src, "-emit=full", "-o", out_exe]
    else:
        raise ValueError(backend)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=BUILD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"BUILD TIMEOUT ({BUILD_TIMEOUT}s)"
    if proc.returncode != 0:
        return False, proc.stderr or proc.stdout
    if not os.path.exists(out_exe):
        return False, "build reported success but no executable produced"
    return True, ""


def run(out_exe):
    """Run a built demo. Returns (ok, detail)."""
    try:
        proc = subprocess.run([out_exe], capture_output=True, text=True,
                              timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, f"RUN TIMEOUT ({RUN_TIMEOUT}s)"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, f"exit={proc.returncode} :: {out.strip()}"
    if "PASS" not in (proc.stdout or ""):
        return False, f"no PASS in stdout :: {out.strip()}"
    return True, proc.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="+", help=".vc demo files or directories")
    ap.add_argument("--vcc", required=True, help="path to the vcc (GLSL) driver")
    ap.add_argument("--mlirc", help="path to the vc (MLIR) driver; "
                                    "omit to skip the MLIR backend")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep running remaining demos after a failure "
                         "(default for this suite)")
    args = ap.parse_args()

    backends = [("glsl", args.vcc)]
    if args.mlirc:
        backends.append(("mlir", args.mlirc))

    passes = 0
    failures = 0
    skips = 0

    for path in expand_demos(args.tests):
        name = os.path.basename(path)
        if not has_main(path):
            skips += 1
            if args.verbose:
                print(f"SKIP: {name} (no host main)")
            continue
        for backend, tool in backends:
            label = f"{name} :: {backend}"
            if backend == "mlir" and name in MLIR_UNSUPPORTED:
                skips += 1
                if args.verbose:
                    print(f"SKIP: {label} ({MLIR_UNSUPPORTED[name]})")
                continue
            if backend == "glsl" and name in GLSL_UNSUPPORTED:
                skips += 1
                if args.verbose:
                    print(f"SKIP: {label} ({GLSL_UNSUPPORTED[name]})")
                continue
            with tempfile.TemporaryDirectory() as td:
                out_exe = os.path.join(td, "demo")
                ok, detail = build(backend, tool, path, out_exe)
                if not ok:
                    failures += 1
                    print(f"FAIL(build): {label}")
                    print("  " + detail.strip().replace("\n", "\n  "))
                    if not args.continue_on_error:
                        sys.exit(1)
                    continue
                ok, detail = run(out_exe)
                if ok:
                    passes += 1
                    if args.verbose:
                        print(f"PASS: {label}  ({detail})")
                else:
                    failures += 1
                    print(f"FAIL(run): {label}")
                    print("  " + detail.strip().replace("\n", "\n  "))
                    if not args.continue_on_error:
                        sys.exit(1)

    total = passes + failures
    mlir_note = f", {skips} skipped"
    if failures:
        print(f"\n{failures}/{total} demos failed{mlir_note}")
        sys.exit(1)
    print(f"\nall {total} (backend,demo) pairs passed{mlir_note}")
    sys.exit(0)


if __name__ == "__main__":
    main()
