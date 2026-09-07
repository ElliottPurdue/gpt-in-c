# gcc by default, but an explicitly set CC wins. make's built-in default for CC
# is "cc", which counts as set, so ?= would never take effect here.
ifeq ($(origin CC),default)
CC = gcc
endif

# -msse2 -mfpmath=sse is load-bearing, not tuning. This is a 32-bit toolchain,
# where gcc defaults to the x87 unit and FLT_EVAL_METHOD is 2: every float
# expression is evaluated in 80-bit extended precision and rounded to 32 bits
# only when stored. That makes the library quietly more accurate than the
# float32 it claims to be, hides error that a real single-precision FPU would
# show, and breaks exact comparisons between a value in memory and the same
# value in a register. Targets this code is meant for -- Cortex-M, Xtensa --
# have genuine 32-bit FPUs and no excess precision, so the host build is pinned
# to match them.
CFLAGS  = -std=c99 -O3 -Wall -Wextra -Werror -pedantic -msse2 -mfpmath=sse
LDFLAGS = -lm

SRC   = src/tensor.c src/ops.c src/model.c src/optim.c src/tokenizer.c
TESTS = tests/main.c tests/oracle.c tests/test_ops.c tests/test_model.c tests/test_optim.c tests/test_tokenizer.c

PYTHON ?= python

.PHONY: all test oracle train bench-matmul omp verify-determinism mutate clean

all: test

# The oracle data is generated, not committed: it is derived from ref/*.py and
# would otherwise be a binary blob in the history that nobody can review.
oracle:
	$(PYTHON) ref/reference.py
	$(PYTHON) ref/units.py

test: build/run_tests
	./build/run_tests

build/run_tests: $(SRC) $(TESTS) | build
	$(CC) $(CFLAGS) $(SRC) $(TESTS) -o $@ $(LDFLAGS)

build:
	mkdir -p build

train: build/train data/input.txt
	./build/train data/input.txt 500

build/train: $(SRC) train.c | build
	$(CC) $(CFLAGS) $(SRC) train.c -o $@ $(LDFLAGS)

# The corpus is the project's own source, concatenated. Using the repo itself
# keeps the training data reproducible and the build free of downloads, and a
# character model has plenty to learn from C syntax: matched braces,
# indentation, comment delimiters, identifier conventions.
#
# README.md is deliberately not in it, though it was. That made the corpus
# depend on the document that reports the training results, so correcting a loss
# figure in the README changed the training data, which changed the loss figure.
# A few pages of prose are not worth a fixed point that never settles.
#
# One list, used for both the prerequisites and the recipe. Writing them out
# twice had already gone wrong: tests/*.h and train.c were concatenated but not
# depended on, so editing either left a stale corpus behind and the next run
# trained on the previous revision. $(sort) also pins the order, so the bytes do
# not depend on how the shell happens to expand a glob.
CORPUS = $(sort $(wildcard src/*.c src/*.h tests/*.c tests/*.h ref/*.py) train.c)

# tr strips carriage returns, which is not cosmetic. This tree is checked out on
# Windows with CRLF endings and on the Linux CI runner with LF, so the raw
# concatenation differs by one byte per line: 145,065 bytes and 95 distinct
# characters there against 147,047 and 96 here, a different vocabulary and a
# different loss curve from the same commit. The fingerprint caught it on its
# first CI run. Normalising here rather than through .gitattributes means the
# corpus does not depend on anyone's core.autocrlf.
data/input.txt: $(CORPUS)
	mkdir -p data
	cat $(CORPUS) | tr -d '\r' > $@
	@wc -c < $@ | xargs echo "  corpus bytes:"

# Builds the trainer twice, with the matmul that came first and with the blocked
# one that replaced it, and runs both over the same corpus. Two numbers come out
# of it: the throughput, which should differ, and the loss curve, which should
# not differ at all. The second is the harder claim and the reason the naive
# loops are still in the tree rather than only in the history.
STEPS ?= 30

bench-matmul: data/input.txt | build
	$(CC) $(CFLAGS) -DGPTC_NAIVE_MATMUL $(SRC) train.c -o build/train_naive $(LDFLAGS)
	$(CC) $(CFLAGS) $(SRC) train.c -o build/train_blocked $(LDFLAGS)
	@echo ""
	@echo "=== naive triple loop ==="
	./build/train_naive data/input.txt $(STEPS)
	@echo ""
	@echo "=== blocked ==="
	./build/train_blocked data/input.txt $(STEPS)

# OpenMP builds, as separate targets rather than a variable on the default ones.
#
# A variable would be the obvious design and is the wrong one here: tools/
# mutate.py shells out to `$(MAKE) test` without passing an env, so an exported
# OMP=1 in the caller's environment would win over a makefile default and the
# mutation suite would silently measure a build nobody asked for. Separate
# targets cannot be turned on by accident.
#
# -ffp-contract=off is insurance, not a fix. -std=c99 already disables
# contraction, so a multiply and an add are not fused into one rounding step
# today, and the assembly confirms it. But that is a side effect of the standards
# mode, and a later switch to -std=gnu99 would silently enable FMA and change
# every result. Saying it outright costs nothing and removes the trap.
OMPFLAGS = -fopenmp -ffp-contract=off

build/train_omp: $(SRC) train.c | build
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC) train.c -o $@ $(LDFLAGS)

build/run_tests_omp: $(SRC) $(TESTS) | build
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC) $(TESTS) -o $@ $(LDFLAGS)

omp: build/run_tests_omp
	./build/run_tests_omp

# The claim threading has to earn: same bytes at every thread count.
#
# `make test` cannot establish it. Its tolerances are 1e-5 absolute and 1e-4
# relative, while a reordered sum over a few hundred elements moves a result by
# about 1e-7, so a build with the accumulation order destroyed passes the entire
# suite. Only an exact comparison is evidence, which is what the harness does.
verify-determinism: build/train build/train_omp data/input.txt
	$(PYTHON) tools/verify_determinism.py

# Breaks the library on purpose, one bug at a time, and checks the suite
# notices. A passing suite is evidence about the tests only if they can fail.
mutate:
	$(PYTHON) tools/mutate.py

clean:
	rm -rf build
