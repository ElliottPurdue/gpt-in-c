/* Flat float32 tensors.
 *
 * Deliberately not a general tensor library. There are no strides, no views and
 * no broadcasting: every tensor owns a contiguous row-major block, and the
 * operations in ops.h index it arithmetically. A transformer needs exactly one
 * memory layout, and supporting more would add the machinery this project
 * exists to avoid.
 *
 * Row-major and contiguous also means the layout matches what PyTorch writes
 * with `.contiguous().numpy().tobytes()`, so oracle data loads with a single
 * fread and comparisons are element-for-element.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIMS 4

typedef struct {
    float *data;
    int dims[TENSOR_MAX_DIMS];
    int ndim;
    int size;           /* product of dims, cached */
    int owns_data;      /* 0 for a borrowed pointer, e.g. a slice of a buffer */
} tensor;

/* Allocates zeroed storage. Returns a tensor with data == NULL on failure;
 * callers in tests check that, and the library itself allocates only at model
 * construction, never inside a forward or backward pass. */
tensor tensor_alloc(int ndim, const int *dims);
tensor tensor_alloc2(int rows, int cols);
tensor tensor_alloc3(int a, int b, int c);
tensor tensor_alloc4(int a, int b, int c, int d);

void tensor_free(tensor *t);

/* Wraps existing memory without copying. The result must not be freed. */
tensor tensor_wrap(float *data, int ndim, const int *dims);

void tensor_zero(tensor *t);
void tensor_copy(tensor *dst, const tensor *src);

int tensor_same_shape(const tensor *a, const tensor *b);

/* Largest absolute and relative difference between two equally shaped tensors.
 * Both matter: absolute error alone flags large values that agree to eight
 * significant figures, and relative error alone explodes on values near zero
 * where the absolute difference is meaningless. The tests require both to be
 * small, which neither measure achieves on its own. */
typedef struct {
    float max_abs;
    float max_rel;
    int index;          /* where max_abs occurred, for reporting */
} tensor_diff;

tensor_diff tensor_compare(const tensor *a, const tensor *b);

#endif /* TENSOR_H */
