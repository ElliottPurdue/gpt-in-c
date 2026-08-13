#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int product(int ndim, const int *dims)
{
    int total = 1;
    for (int i = 0; i < ndim; ++i) {
        total *= dims[i];
    }
    return total;
}

tensor tensor_alloc(int ndim, const int *dims)
{
    tensor t;
    memset(&t, 0, sizeof(t));

    if (ndim < 1 || ndim > TENSOR_MAX_DIMS) {
        return t;
    }

    t.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        t.dims[i] = dims[i];
    }
    t.size = product(ndim, dims);
    t.owns_data = 1;
    t.data = (float *)calloc((size_t)t.size, sizeof(float));
    return t;
}

tensor tensor_alloc2(int rows, int cols)
{
    int dims[2] = { rows, cols };
    return tensor_alloc(2, dims);
}

tensor tensor_alloc3(int a, int b, int c)
{
    int dims[3] = { a, b, c };
    return tensor_alloc(3, dims);
}

tensor tensor_alloc4(int a, int b, int c, int d)
{
    int dims[4] = { a, b, c, d };
    return tensor_alloc(4, dims);
}

void tensor_free(tensor *t)
{
    if (t && t->owns_data) {
        free(t->data);
    }
    if (t) {
        t->data = NULL;
        t->size = 0;
        t->ndim = 0;
    }
}

tensor tensor_wrap(float *data, int ndim, const int *dims)
{
    tensor t;
    memset(&t, 0, sizeof(t));

    t.data = data;
    t.ndim = ndim;
    for (int i = 0; i < ndim && i < TENSOR_MAX_DIMS; ++i) {
        t.dims[i] = dims[i];
    }
    t.size = product(ndim, dims);
    t.owns_data = 0;
    return t;
}

void tensor_zero(tensor *t)
{
    if (t && t->data) {
        memset(t->data, 0, (size_t)t->size * sizeof(float));
    }
}

void tensor_copy(tensor *dst, const tensor *src)
{
    if (dst && src && dst->data && src->data && dst->size == src->size) {
        memcpy(dst->data, src->data, (size_t)src->size * sizeof(float));
    }
}

int tensor_same_shape(const tensor *a, const tensor *b)
{
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (int i = 0; i < a->ndim; ++i) {
        if (a->dims[i] != b->dims[i]) {
            return 0;
        }
    }
    return 1;
}

tensor_diff tensor_compare(const tensor *a, const tensor *b)
{
    tensor_diff diff;
    diff.max_abs = 0.0f;
    diff.max_rel = 0.0f;
    diff.index = -1;

    if (!tensor_same_shape(a, b)) {
        diff.max_abs = INFINITY;
        diff.max_rel = INFINITY;
        return diff;
    }

    for (int i = 0; i < a->size; ++i) {
        float x = a->data[i], y = b->data[i];

        if (isnan(x) != isnan(y) || isinf(x) != isinf(y)) {
            diff.max_abs = INFINITY;
            diff.max_rel = INFINITY;
            diff.index = i;
            return diff;
        }
        if (isnan(x) || isinf(x)) {
            continue;   /* both are the same non-finite value */
        }

        float absolute = fabsf(x - y);
        if (absolute > diff.max_abs) {
            diff.max_abs = absolute;
            diff.index = i;
        }

        /* Scaled by the larger magnitude rather than by one side, so the result
         * does not depend on argument order, and floored so that a pair of
         * values near zero reports a small relative error instead of a
         * meaningless large one. */
        float scale = fmaxf(fabsf(x), fabsf(y));
        if (scale > 1e-6f) {
            float relative = absolute / scale;
            if (relative > diff.max_rel) {
                diff.max_rel = relative;
            }
        }
    }
    return diff;
}
