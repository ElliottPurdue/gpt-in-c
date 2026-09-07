#!/usr/bin/env python3
"""Checks that threading did not change the arithmetic.

The blocked matmul is threaded with OpenMP, and the argument that this is safe
is that every pragma splits work across loops that partition the *output* of an
accumulation, never across the index being accumulated. Each output element is
therefore still summed by one thread, over the same index, in the same
direction, as it was serially, so the result cannot depend on how many threads
ran.

That is an argument. This is the check.

`make test` will not do it. Its tolerances are 1e-5 absolute and 1e-4 relative,
and a summation reordered over a few hundred float32 values moves a result by
roughly 1e-7 relative. A build with the accumulation order destroyed passes
every test in the suite while the claim it exists to protect is dead. Only an
exact comparison is evidence, so this compares bytes.

Three things are compared, all against the serial build:

  1. the threaded build at one thread, which isolates the pragmas themselves
     from anything about concurrency,
  2. the threaded build at several thread counts, which is the actual claim,
  3. the sampled text, which is the stronger half of it. A loss is one number
     per step and could coincide in the digits printed. Two hundred characters
     drawn from the model's own distribution coincide only if every weight does.

Exit status is 0 when every run matches the serial one byte for byte.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "input.txt"
STEPS = "30"

# 1 first: if the threaded build disagrees with serial at one thread, the
# pragmas changed the code and no amount of thread-count sweeping is meaningful.
# The rest bracket the machine: fewer threads than cores, and more, since
# oversubscription changes the interleaving without changing the partition.
THREAD_COUNTS = ["1", "2", "3", "4", "7", "16", "33"]

# A run that silently fell back to serial would agree with every other run for
# the wrong reason, so the reported count is checked against the requested one.
THREADS_LINE = re.compile(r"^\s*threads\s+(\d+)\s*$", re.MULTILINE)

# Throughput is the one thing that is expected to differ, so it is stripped
# before comparison rather than being allowed to fail the diff. Everything else
# on those lines, the losses and the gradient norms, is compared exactly.
TOKENS_PER_SEC = re.compile(r"\s*\d+ tok/s")
WALL_TIME = re.compile(r"^\s*\d+ steps in [\d.]+ s, \d+ tokens/s\s*$", re.MULTILINE)


def run(binary, threads=None):
    """Run one training run and return its output with timings removed."""
    env = None
    if threads is not None:
        import os

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = threads
        # Without this, a runtime that cannot honour the request is free to
        # give back whatever it likes, and the sweep silently tests one count
        # several times.
        env["OMP_DYNAMIC"] = "FALSE"

    proc = subprocess.run(
        [str(binary), str(CORPUS), STEPS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        sys.exit(
            "%s exited %d\n%s\n%s"
            % (binary.name, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        )

    text = proc.stdout
    reported = None
    match = THREADS_LINE.search(text)
    if match:
        reported = match.group(1)

    # Drop the lines and fields that are allowed to differ.
    text = TOKENS_PER_SEC.sub("", text)
    text = WALL_TIME.sub("  (timing removed)", text)
    # The serial build prints no threads line and the threaded one does, so it
    # cannot take part in a byte comparison between the two.
    text = THREADS_LINE.sub("", text)
    return text, reported


def main():
    serial = ROOT / "build" / "train"
    threaded = ROOT / "build" / "train_omp"
    for path in (serial, threaded):
        if not path.exists() and not path.with_suffix(".exe").exists():
            sys.exit("missing %s: run `make %s` first" % (path.name, path.name))
    if not CORPUS.exists():
        sys.exit("missing %s: run `make data/input.txt` first" % CORPUS)

    if not serial.exists():
        serial = serial.with_suffix(".exe")
    if not threaded.exists():
        threaded = threaded.with_suffix(".exe")

    print("reference: %s (no OpenMP), %s steps" % (serial.name, STEPS))
    reference, _ = run(serial)

    failures = 0
    for count in THREAD_COUNTS:
        output, reported = run(threaded, count)

        if reported is None:
            print("  %3s threads   NO THREAD COUNT REPORTED" % count)
            failures += 1
            continue
        if reported != count:
            # Not necessarily fatal on an oversubscribed request, but it means
            # the run did not test what it claims to, so say so loudly.
            print(
                "  %3s threads   ran with %s instead, not a test of %s"
                % (count, reported, count)
            )
            failures += 1
            continue

        if output == reference:
            print("  %3s threads   identical" % count)
        else:
            print("  %3s threads   DIFFERS from serial" % count)
            failures += 1
            ref_lines = reference.split("\n")
            got_lines = output.split("\n")
            for i, (a, b) in enumerate(zip(ref_lines, got_lines)):
                if a != b:
                    print("      first difference at line %d:" % (i + 1))
                    print("        serial:   %s" % a.strip())
                    print("        threaded: %s" % b.strip())
                    break
            else:
                print(
                    "      same lines, different length: %d vs %d"
                    % (len(ref_lines), len(got_lines))
                )

    total = len(THREAD_COUNTS)
    print(
        "\n  %d/%d thread counts bit-identical to the serial build"
        % (total - failures, total)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
