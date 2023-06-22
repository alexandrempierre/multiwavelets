'''módulo step_1
'''

# testar se callable e list são próximos o suficiente um do outro e do prod

__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
from numpy import linalg
from scipy.special import comb
# from simple import step_0, step_1
from .. import step_0, step_1
# import simple


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

    def test_M_values(self):
        n, k = 8, 2
        xs = np.linspace(0, 1, n, endpoint=True)
        mus = step_0.mu_list(xs, k)
        sigmas = step_0.sigma_list(xs, k)
        ss = step_0.s_array(n, k)
        Ms = step_1.shifted_scaled_moments_matrices_list(xs, k, mus, sigmas, ss)
        Ms_value = [
            np.array([
                [1, -1,   1,   -1   ],
                [1, -1/3, 1/9, -1/27],
                [1,  1/3, 1/9,  1/27],
                [1,  1,   1,    1   ]
            ]),
            np.array([
                [1, -1,   1,   -1   ],
                [1, -1/3, 1/9, -1/27],
                [1,  1/3, 1/9,  1/27],
                [1,  1,   1,    1   ]
            ]),
        ]
        assert np.allclose(Ms_value[0], Ms[0])
        assert np.allclose(Ms_value[1], Ms[1])


class Test_S:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste automatizado para verificar se os diferentes métodos de calcular
a matriz de momentos transladada e dilatada chegam a resultados suficientemente
próximos
'''
    def test_S_value(self):
        k, mu, sigma = 2, 3, 7
        S_value = np.array([
            [
                comb(c - 1, r - 1, exact=True) * (-mu)**(c - r) /sigma**(c - 1)
                if r <= c else 0
                for c in range(1, 2*k + 1)
            ]
            for r in range(1, 2*k + 1)
        ])
        S = step_1.shift_scale_matrix(k, mu, sigma)
        assert np.allclose(S_value, S, )
        assert np.all(S_value == S)
