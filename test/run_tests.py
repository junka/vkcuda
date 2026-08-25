#!/usr/bin/env python3
"""Minimal lit-style test runner for the VC frontend.

Each test file may carry one or more `// RUN:` comment lines; the command
there is executed with:

  * `vcc`      replaced by the path given with --vcc
  * `FileCheck` replaced by the path given with --filecheck
  * `%s`       replaced by the test file path
  * `%t`       replaced by a fresh temp file name

Standard use (frontend regression tests):

  // RUN: vcc -fsyntax-only %s 2>&1 | FileCheck %s
  // CHECK: error: use of undeclared identifier 'x'

Exit status is 0 when every RUN command completes successfully.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

RUN_RE = re.compile(r"//\s*RUN:\s*(.*)")


def run_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [m.group(1) for line in fh if (m := RUN_RE.match(line))]


def expand_tests(arguments):
    """Accept a mix of .vc files and directories (globs *.vc, no recursion)."""
    tests = []
    for arg in arguments:
        if os.path.isdir(arg):
            tests += sorted(glob.glob(os.path.join(arg, "*.vc")))
        else:
            tests.append(arg)
    return tests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tests", nargs="+", help=".vc test files or directories")
    ap.add_argument("--vcc", required=True, help="path to the vcc driver")
    ap.add_argument("--filecheck", required=True,
                    help="path to the FileCheck binary")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep running remaining tests after a failure")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PATH"] = (os.path.dirname(args.filecheck) + os.pathsep +
                   env.get("PATH", ""))

    failed = []
    n_commands = 0
    for path in expand_tests(args.tests):
        for run in run_lines(path):
            n_commands += 1
            cmd = run
            cmd = re.sub(r"\bvcc\b", args.vcc, cmd)
            cmd = re.sub(r"\bFileCheck\b", args.filecheck, cmd)
            fd, tmp = tempfile.mkstemp(suffix=".vc")
            os.close(fd)
            cmd = cmd.replace("%s", path).replace("%t", tmp)

            proc = subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, env=env)
            os.unlink(tmp)

            if proc.returncode == 0:
                if args.verbose:
                    print(f"PASS: {os.path.basename(path)} :: {run}")
                continue

            failed.append(path)
            print(f"FAIL: {os.path.basename(path)} :: {run}")
            print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="")
            if not args.continue_on_error:
                sys.exit(1)

    if failed:
        print(f"{len(failed)}/{len(args.tests)} test files failed")
        sys.exit(1)
    if args.verbose:
        print(f"all {n_commands} RUN command(s) passed")


if __name__ == "__main__":
    main()