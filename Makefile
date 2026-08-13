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

.PHONY: all test oracle train mutate clean

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

# The corpus is the project's own source and prose, concatenated. Using the repo
# itself keeps the training data reproducible and the build free of downloads,
# and a character model has plenty to learn from C syntax: matched braces,
# indentation, comment delimiters, identifier conventions.
data/input.txt: $(wildcard src/*.c src/*.h tests/*.c ref/*.py) README.md
	mkdir -p data
	cat src/*.c src/*.h tests/*.c tests/*.h ref/*.py train.c README.md > $@
	@wc -c < $@ | xargs echo "  corpus bytes:"

# Breaks the library on purpose, one bug at a time, and checks the suite
# notices. A passing suite is evidence about the tests only if they can fail.
mutate:
	$(PYTHON) tools/mutate.py

clean:
	rm -rf build
