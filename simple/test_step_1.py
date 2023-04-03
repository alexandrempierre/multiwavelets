'''módulo test_step_0
'''

# testar se callable e list são próximos o suficiente um do outro e do prod

__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
from numpy import linalg
import step_0
import step_1


MAX_EXP = 8
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
N, K = 4, 1


class Test_M:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste automatizado para verificar se os diferentes métodos de calcular
a matriz de momentos transladada e dilatada chegam a resultados suficientemente
próximos
'''
    def test_M_callable_equals_M(self):
        for n, k in INDICES:
            xs = np.linspace(0, 1, n, endpoint=True)
            Ms_callable = step_1.shifted_scaled_moments_matrices_callable(
                xs, k
            )
            Ms = step_1.shifted_scaled_moments_matrices(xs, k)
            assert np.all(
                linalg.norm(M - Ms_callable(i), np.inf, axis=1) < 1e-16
                for i, M in enumerate(Ms)
            )
            assert sum(
                np.abs(M - Ms_callable(i)).sum() for i, M in enumerate(Ms)
            ) == 0
            assert all(
                np.all(M == Ms_callable(i)) for i, M in enumerate(Ms)
            )

    def test_M_callable_equals_M_list(self):
        for n, k in INDICES:
            xs = np.linspace(0, 1, n, endpoint=True)
            Ms_callable = step_1.shifted_scaled_moments_matrices_callable(
                xs, k
            )
            mus = step_0.mu_list(xs, k)
            sigmas = step_0.sigma_list(xs, k)
            ss = step_0.s_array(n, k)
            Ms_list = step_1.shifted_scaled_moments_matrices_list(
                xs, k, mus, sigmas, ss
            )
            assert np.all(
                linalg.norm(M - Ms_callable(i), np.inf, axis=1) < 1e-16
                for i, M in enumerate(Ms_list)
            )
            assert sum(
                np.abs(M - Ms_callable(i)).sum()
                for i, M in enumerate(Ms_list)
            ) == 0
            assert all(
                np.all(M == Ms_callable(i)) for i, M in enumerate(Ms_list)
            )

    def test_M_callable_nears_M_prod(self):
        for n, k in INDICES:
            xs = np.linspace(0, 1, n, endpoint=True)
            Ms_callable = step_1.shifted_scaled_moments_matrices_callable(
                xs, k
            )
            mus = step_0.mu_list(xs, k)
            sigmas = step_0.sigma_list(xs, k)
            Ms_prod = step_1.shifted_scaled_moments_matrices_prod(
                xs, k, mus, sigmas
            )
            assert np.all(
                linalg.norm(M - Ms_callable(i), np.inf, axis=1) < 1e-16
                for i, M in enumerate(Ms_prod)
            )
            # assert all(
            #     [np.all(M == Ms_callable(i)) for i, M in enumerate(Ms_prod)]
            # )
