#include <stdio.h>

int tests_run = 0;
int checks_failed = 0;

void register_op_tests(void);
void register_model_tests(void);
void register_optim_tests(void);

int main(void)
{
    printf("gpt-in-c test suite\n\n");

    register_op_tests();
    register_model_tests();
    register_optim_tests();

    printf("\n%d tests, %d failed checks\n", tests_run, checks_failed);
    return checks_failed == 0 ? 0 : 1;
}
