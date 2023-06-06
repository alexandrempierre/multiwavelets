'''módulo step_3'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
#
from simple import step_0, step_3


MAX_EXP = 4
INDICES = tuple(
    (n, k)
    for n in np.power(
        2, np.arange(2, MAX_EXP + 1), dtype=int
    )
    for k in np.power(
        2, np.arange(0, MAX_EXP, 1), dtype=int
    )
    if n > k
)


class Test_S1_S2:
    # pylint: disable=invalid-name, missing-function-docstring
    '''_'''
    def test_shift_scale_matrices(self):
        n, k = 8, 2
        xs = np.linspace(0, 1, n)
        mus = step_0.mu_list(xs, k)
        sigmas = step_0.sigma_list(xs, k)
        matrices = next(step_3.shift_scale_matrices(n, k, mus, sigmas), None)
        assert matrices is not None
        assert np.allclose(a, b)
