/* Tiny shared harness for Stanley test files. Single-header. */
#ifndef STANLEY_TEST_CHECK_H
#define STANLEY_TEST_CHECK_H

#include <stdio.h>

static int _st_pass = 0, _st_fail = 0;

#define CHECK(cond, name) do {                                                 \
    if (cond) { _st_pass++; printf("  [PASS] %s\n", name); }                   \
    else      { _st_fail++; printf("  [FAIL] %s  (%s:%d)\n", name, __FILE__, __LINE__); } \
} while (0)

#define CHECK_REPORT(suite_name) do {                                          \
    printf("\n=== %s: %d passed, %d failed ===\n",                             \
           suite_name, _st_pass, _st_fail);                                    \
    return _st_fail == 0 ? 0 : 1;                                              \
} while (0)

#endif
