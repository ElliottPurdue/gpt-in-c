# gcc by default, but an explicitly set CC wins. make's built-in default for CC
# is "cc", which counts as set, so ?= would never take effect here.
ifeq ($(origin CC),default)
CC = gcc
endif

CFLAGS  = -std=c99 -O2 -Wall -Wextra -Werror -pedantic
LDFLAGS = -lm

SRC   = src/tensor.c src/ops.c
TESTS = tests/main.c tests/oracle.c tests/test_ops.c

PYTHON ?= python

.PHONY: all test oracle clean

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

clean:
	rm -rf build
