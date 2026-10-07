#!/usr/bin/env python3
"""spirv-val every SPIR-V module both backends emit.

Usage: check_spirv_val.py --mlirc <vc> [--vcc <vcc>] [--spirv-val <path>]
                          <file or dir>...

GPU end-to-end tests need a Vulkan device, which CI does not have, so nothing
there ever looks at the bytes `vc -emit=spirv` and `vcc -emit=spirv` write. This is
the closest device-free check that those bytes are the contract: spirv-val applies
the same structural rules the driver applies at vkCreateShaderModule (capabilities
against the memory model, entry-point interface lists, OpBitcast on logical
pointers, pointer-typed variables under logical addressing), so a lowering that
produces an unloadable module fails here instead of on whoever has a GPU.

Both drivers are asked for every kernel they produce: `vc` serializes all kernels of
a translation unit into one spirv.module, and `vcc` writes one module per kernel
(`<out>.spv`, `<out>.2.spv`, … — SPIR-V cannot concatenate entry points). Every file
the driver leaves next to the requested output name is validated.

An input a driver refuses to compile is not a validation result — those are the
frontend / negative IR tests, covered by vc-check and mlir-check — and is counted as
skipped for that backend. A requested backend that yields no module at all is
reported as a failure, not a silent pass (vcc needs glslc).

Exit status is non-zero when a module fails validation and its file is not listed in
KNOWN_INVALID, or when a KNOWN_INVALID entry validates on every backend that did
produce a module for it — the exception list has to say what is actually broken, not
what was once.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

# Files whose emitted module is invalid SPIR-V for a reason a lowering still has,
# keyed by repo-relative path, with the reason. Applies to both backends: a listed
# file is accepted as failing, and only complained about once nothing fails at all.
# Empty: every module either backend emits validates. Add an entry only for a
# known-broken lowering with its reason, so the list stays a report on what is
# actually broken rather than what once was.
KNOWN_INVALID = {}


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


def modules_from(compiler, src, out):
    """Compile `src` with `compiler` and return the SPIR-V files it wrote.

    None means the driver refused the input (a frontend or negative test), which is
    not a validation result. Otherwise every `<out>*.spv` is returned: the driver
    names the file it was asked for, plus one per extra kernel when the backend is
    `vcc` and the source has more than one kernel.
    """
    proc = subprocess.run([str(compiler), "-emit=spirv", str(src), "-o", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    # vcc names an extra kernel by replacing `.spv` with `.2.spv`, `.3.spv`, ...
    # so everything sharing the stem is in scope, not just the requested file.
    pattern = out.name[: -len(".spv")] + "*.spv"
    written = sorted(out.parent.glob(pattern))
    return [p for p in written if p.read_bytes()[:4] == b"\x03\x02\x23\x07"] or None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mlirc", required=True, help="path to the vc (MLIR) driver")
    ap.add_argument("--vcc", help="path to the vcc (GLSL) driver; omit to skip it")
    ap.add_argument("--spirv-val", default="spirv-val", help="spirv-val binary")
    ap.add_argument("paths", nargs="+", help=".vc files or directories")
    args = ap.parse_args(argv[1:])

    val = args.spirv_val
    if subprocess.run([val, "--version"], capture_output=True).returncode != 0:
        print(f"check_spirv_val: '{val}' is not runnable; install spirv-tools",
              file=sys.stderr)
        return 2

    backends = [("mlir", pathlib.Path(args.mlirc))]
    if args.vcc:
        backends.append(("glsl", pathlib.Path(args.vcc)))

    totals = {name: [0, 0] for name, _ in backends}  # [modules validated, skipped]
    still_invalid = unexpected = outdated = 0
    observed = set()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        for f in collect([pathlib.Path(p) for p in args.paths]):
            flat = f.as_posix().replace("/", "_")
            listed = known_invalid_key(f)
            failures = []
            produced = 0
            for name, driver in backends:
                outs = modules_from(driver, f, tmpdir / f"{flat}.{name}.spv")
                if outs is None:
                    totals[name][1] += 1
                    continue
                totals[name][0] += len(outs)
                produced += len(outs)
                for spv in outs:
                    proc = subprocess.run([val, str(spv)],
                                          capture_output=True, text=True)
                    if proc.returncode == 0:
                        continue
                    reason = (proc.stderr.strip().splitlines() or [""])[0]
                    failures.append(f"{name} {spv.name}: {reason}")
            if failures:
                if listed:
                    observed.add(listed)
                    still_invalid += 1
                else:
                    for msg in failures:
                        print(f"FAIL: {f}: {msg}", file=sys.stderr)
                    unexpected += 1
            elif listed and produced:
                print(f"FAIL: {f} is listed as known-invalid but now validates on "
                      f"every backend that compiled it; drop it from KNOWN_INVALID",
                      file=sys.stderr)
                outdated += 1

    summary = ", ".join(
        f"{totals[name][0]} {name} module(s) validated "
        f"({totals[name][1]} file(s) not compiled)" for name, _ in backends)
    print(f"{summary}; {still_invalid} file(s) accepted as known invalid")
    if observed:
        print("known-invalid (fix the lowering; do not add to this list):")
        for key in sorted(observed):
            print(f"  {key}: {KNOWN_INVALID[key]}")
    silent = [name for name, _ in backends if totals[name][0] == 0]
    if silent:
        print(f"check_spirv_val: backend(s) {', '.join(silent)} produced no module "
              f"at all — the driver is not working (vcc needs glslc), which is not "
              f"a pass", file=sys.stderr)
        return 1
    return 1 if (unexpected or outdated) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
