'''módulo step_3'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
#
from .. import step_0, step_3


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
        S1, S2 = matrices
        assert np.all(
            S1 == np.array([
                [1, -14/9,  784/81, -21952/729],
                [0,   7/3, -392/27,    5488/81],
                [0,     0,    49/9,   -1372/27],
                [0,     0,       0,     343/27],
            ])
        )
        assert np.all(
            S2 == np.array([
                [1, 4/7, 16/49, 64/323],
                [0,   1,   8/7,  48/49],
                [0,   0,     1,   12/7],
                [0,   0,     0,      1],
            ])
        )
