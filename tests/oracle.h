/* Loads a tensor dump produced by ref/reference.py or ref/units.py. */

#ifndef ORACLE_H
#define ORACLE_H

#include "../src/tensor.h"

#define ORACLE_MAX_TENSORS 256
#define ORACLE_MAX_NAME 128

typedef struct {
    char name[ORACLE_MAX_NAME];
    tensor value;
} oracle_entry;

typedef struct {
    oracle_entry entries[ORACLE_MAX_TENSORS];
    int count;
} oracle;

int oracle_load(oracle *store, const char *path);

/* Returns NULL when absent. */
const tensor *oracle_find(const oracle *store, const char *name);

/* Exits with status 2 when absent: a missing tensor means the test and the dump
 * have drifted apart, which is a broken test rather than a failed comparison. */
const tensor *oracle_require(const oracle *store, const char *name);

void oracle_free(oracle *store);

#endif /* ORACLE_H */
