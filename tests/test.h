/* A test harness in a header, matching the one in attitude-estimation: no
 * framework, no build system integration, and nothing to install. */

#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <math.h>

extern int tests_run;
extern int checks_failed;

#define RUN(fn) do {                                                          \
    tests_run++;                                                              \
    fn();                                                                     \
} while (0)

#define CHECK(cond) do {                                                      \
    if (!(cond)) {                                                            \
        checks_failed++;                                                      \
        printf("  FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                         \
} while (0)

#define CHECK_NEAR(got, want, tol) do {                                       \
    double _g = (double)(got), _w = (double)(want);                           \
    if (!(fabs(_g - _w) <= (tol))) {                                          \
        checks_failed++;                                                      \
        printf("  FAILED %s:%d  %s = %.9g, expected %.9g (tol %.3g)\n",       \
               __FILE__, __LINE__, #got, _g, _w, (double)(tol));              \
    }                                                                         \
} while (0)

#define CHECK_BELOW(got, limit) do {                                          \
    double _g = (double)(got);                                                \
    if (!(_g < (double)(limit))) {                                            \
        checks_failed++;                                                      \
        printf("  FAILED %s:%d  %s = %.9g, expected < %.9g\n",                \
               __FILE__, __LINE__, #got, _g, (double)(limit));                \
    }                                                                         \
} while (0)

/* Compares a computed tensor against the oracle's, reporting both error
 * measures on failure so it is obvious whether the disagreement is a scaling
 * problem, a layout problem, or float noise. */
#define CHECK_MATCHES(computed, expected, abs_tol, rel_tol) do {              \
    tensor_diff _d = tensor_compare((computed), (expected));                  \
    if (!(_d.max_abs <= (abs_tol) || _d.max_rel <= (rel_tol))) {              \
        checks_failed++;                                                      \
        printf("  FAILED %s:%d  %s vs %s: max_abs %.3g (tol %.3g), "          \
               "max_rel %.3g (tol %.3g) at index %d\n",                       \
               __FILE__, __LINE__, #computed, #expected,                      \
               (double)_d.max_abs, (double)(abs_tol),                         \
               (double)_d.max_rel, (double)(rel_tol), _d.index);              \
    }                                                                         \
} while (0)

#endif /* TEST_H */
