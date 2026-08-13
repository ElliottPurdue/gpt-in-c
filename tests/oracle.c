/* Reads the binary dumps written by ref/reference.py and ref/units.py.
 *
 * Lives in tests/ rather than src/ so the library keeps no file I/O and no
 * stdio dependency -- the same objects compile for a freestanding target.
 */

#include "oracle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ORACLE_MAGIC "GPTC"
#define ORACLE_VERSION 1u

static int read_u32(FILE *f, unsigned *out)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, f) != 4) {
        return 0;
    }
    /* Assembled byte by byte rather than fread into an unsigned, so the file
     * stays little-endian regardless of the host. */
    *out = (unsigned)bytes[0] | ((unsigned)bytes[1] << 8) |
           ((unsigned)bytes[2] << 16) | ((unsigned)bytes[3] << 24);
    return 1;
}

int oracle_load(oracle *store, const char *path)
{
    memset(store, 0, sizeof(*store));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "oracle: cannot open %s\n"
                        "  run: python ref/reference.py && python ref/units.py\n",
                path);
        return 0;
    }

    char magic[4];
    unsigned version, count;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, ORACLE_MAGIC, 4) != 0) {
        fprintf(stderr, "oracle: %s is not a GPTC file\n", path);
        fclose(f);
        return 0;
    }
    if (!read_u32(f, &version) || version != ORACLE_VERSION) {
        fprintf(stderr, "oracle: %s has version %u, expected %u\n",
                path, version, ORACLE_VERSION);
        fclose(f);
        return 0;
    }
    if (!read_u32(f, &count) || count > ORACLE_MAX_TENSORS) {
        fprintf(stderr, "oracle: %s holds %u tensors, limit is %d\n",
                path, count, ORACLE_MAX_TENSORS);
        fclose(f);
        return 0;
    }

    for (unsigned i = 0; i < count; ++i) {
        oracle_entry *entry = &store->entries[i];
        unsigned name_length, ndim;

        if (!read_u32(f, &name_length) || name_length >= ORACLE_MAX_NAME) {
            fprintf(stderr, "oracle: bad name length at tensor %u\n", i);
            fclose(f);
            return 0;
        }
        if (fread(entry->name, 1, name_length, f) != name_length) {
            fclose(f);
            return 0;
        }
        entry->name[name_length] = '\0';

        if (!read_u32(f, &ndim) || ndim < 1 || ndim > TENSOR_MAX_DIMS) {
            fprintf(stderr, "oracle: %s has ndim %u\n", entry->name, ndim);
            fclose(f);
            return 0;
        }

        int dims[TENSOR_MAX_DIMS];
        for (unsigned d = 0; d < ndim; ++d) {
            unsigned value;
            if (!read_u32(f, &value)) {
                fclose(f);
                return 0;
            }
            dims[d] = (int)value;
        }

        entry->value = tensor_alloc((int)ndim, dims);
        if (!entry->value.data) {
            fprintf(stderr, "oracle: out of memory for %s\n", entry->name);
            fclose(f);
            return 0;
        }
        if (fread(entry->value.data, sizeof(float), (size_t)entry->value.size, f)
                != (size_t)entry->value.size) {
            fprintf(stderr, "oracle: truncated data for %s\n", entry->name);
            fclose(f);
            return 0;
        }
        store->count++;
    }

    fclose(f);
    return 1;
}

const tensor *oracle_find(const oracle *store, const char *name)
{
    for (int i = 0; i < store->count; ++i) {
        if (strcmp(store->entries[i].name, name) == 0) {
            return &store->entries[i].value;
        }
    }
    return NULL;
}

const tensor *oracle_require(const oracle *store, const char *name)
{
    const tensor *found = oracle_find(store, name);
    if (!found) {
        /* A missing tensor means the dump and the test have drifted apart. That
         * is a broken test rather than a failing one, so it aborts instead of
         * being counted as a numerical disagreement. */
        fprintf(stderr, "oracle: no tensor named '%s' in the dump\n", name);
        exit(2);
    }
    return found;
}

void oracle_free(oracle *store)
{
    for (int i = 0; i < store->count; ++i) {
        tensor_free(&store->entries[i].value);
    }
    store->count = 0;
}
