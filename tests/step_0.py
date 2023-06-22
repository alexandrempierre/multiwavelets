# ruff: noqa: E741
'''módulo step_0
'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


import numpy as np
from .. import step_0
from .step_0_values import (
    Test_s_values, Test_mu_values, Test_sigma_values
)


MAX_EXP = 16
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


class Test_s:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste automatizado para verificar se os diferentes métodos de calcular
a s chegam a resultados suficientemente próximos
'''
    def test_one_based_s_callable_equals_zero_based_s_callable(self):
        for n, k in INDICES:
            s_k_0 = step_0.s_callable(k, 0)
            s_callable_0 = np.array([
                s_k_0(i) for i in range(n // (2*k))
            ])
            s_k_1 = step_0.s_callable(k, 1)
            s_callable_1 = np.array([
                s_k_1(i) for i in range(1, n // (2*k) + 1)
            ])
            assert np.all(s_callable_0 == s_callable_1)

    def test_one_based_s_array_equals_zero_based_s_array(self):
        for n, k in INDICES:
            s_array_0 = step_0.s_array(n, k, 0)
            s_array_1 = step_0.s_array(n, k, 1)
            assert np.all(s_array_0 == s_array_1)

    def test_s_callable_equals_s_array(self):
        for n, k in INDICES:
            s_k_0 = step_0.s_callable(k, 0)
            s_array_0 = step_0.s_array(n, k, 0)
            s_callable_0 = np.array([
                s_k_0(i) for i in range(n // (2*k))
            ])
            assert np.all(s_array_0 == s_callable_0)


class Test_mu:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste automatizado para verificar se os diferentes métodos de calcular
a mu chegam a resultados suficientemente próximos
'''
    def test_one_based_mu_callable_equals_zero_based_mu_callable(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            l = int(np.log2(n // k))
            # index starts with 1
            mu_j_i_1 = step_0.mu_callable(x, k, 1)
            mu_callable_1 = [
                [
                    mu_j_i_1(j, i)
                    for i in range(1, n // (k * 2**j) + 1)
                ]
                for j in range(1, l + 1)
            ]
            # index starts with 0
            mu_j_i_0 = step_0.mu_callable(x, k, 0)
            mu_callable_0 = [
                [
                    mu_j_i_0(j, i)
                    for i in range(n // (k * 2**(j + 1)))
                ]
                for j in range(l)
            ]
            # assert that the results produced are equal
            assert len(mu_callable_1) == len(mu_callable_0)
            for (values_mu_1, values_mu_0) in zip(
                mu_callable_1, mu_callable_0
            ):
                assert len(values_mu_1) == len(values_mu_0)
                for (value_mu_1, value_mu_0) in zip(
                    values_mu_1, values_mu_0
                ):
                    assert value_mu_1 == value_mu_0

    def test_one_based_mu_array_equals_zero_based_mu_array(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            mu_list_1 = step_0.mu_list(x, k, 1)
            mu_list_0 = step_0.mu_list(x, k, 0)
            assert len(mu_list_1) == len(mu_list_0)
            for (values_mu_1, values_mu_0) in zip(
                mu_list_1, mu_list_0
            ):
                assert len(values_mu_1) == len(values_mu_0)
                for (value_mu_1, value_mu_0) in zip(
                    values_mu_1, values_mu_0
                ):
                    assert value_mu_1 == value_mu_0

    def test_mu_callable_equals_mu_array(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            l = int(np.log2(n // k))
            mu_list_0 = step_0.mu_list(x, k, 0)
            mu_j_i_0 = step_0.mu_callable(x, k, 0)
            mu_callable_0 = [
                [
                    mu_j_i_0(j, i)
                    for i in range(n // (k * 2**(j + 1)))
                ]
                for j in range(l)
            ]
            # assert that the results produced are equal
            assert len(mu_list_0) == len(mu_callable_0)
            for (values_mu_list, values_mu_callable) in zip(
                mu_list_0, mu_callable_0
            ):
                assert len(values_mu_list) == len(values_mu_callable)
                for (value_mu_list, value_mu_callable) in zip(
                    values_mu_list, values_mu_callable
                ):
                    assert value_mu_list == value_mu_callable


class Test_sigma:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste automatizado para verificar se os diferentes métodos de calcular
a sigma chegam a resultados suficientemente próximos
'''
    def test_one_based_sigma_callable_equals_zero_based_sigma_callable(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            l = int(np.log2(n // k))
            # index starts with 1
            sigma_j_i_1 = step_0.sigma_callable(x, k, 1)
            sigma_callable_1 = [
                [
                    sigma_j_i_1(j, i)
                    for i in range(1, n // (k * 2**j) + 1)
                ]
                for j in range(1, l + 1)
            ]
            # index starts with 0
            sigma_j_i_0 = step_0.sigma_callable(x, k, 0)
            sigma_callable_0 = [
                [
                    sigma_j_i_0(j, i)
                    for i in range(n // (k * 2**(j + 1)))
                ]
                for j in range(l)
            ]
            # assert that the results produced are equal
            assert len(sigma_callable_1) == len(sigma_callable_0)
            for (values_sigma_1, values_sigma_0) in zip(
                sigma_callable_1, sigma_callable_0
            ):
                assert len(values_sigma_1) == len(values_sigma_0)
                for (value_sigma_1, value_sigma_0) in zip(
                    values_sigma_1, values_sigma_0
                ):
                    assert value_sigma_1 == value_sigma_0

    def test_one_based_sigma_array_equals_zero_based_sigma_array(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            sigma_list_1 = step_0.sigma_list(x, k, 1)
            sigma_list_0 = step_0.sigma_list(x, k, 0)
            assert len(sigma_list_1) == len(sigma_list_0)
            for (values_sigma_1, values_sigma_0) in zip(
                sigma_list_1, sigma_list_0
            ):
                assert len(values_sigma_1) == len(values_sigma_0)
                for (value_sigma_1, value_sigma_0) in zip(
                    values_sigma_1, values_sigma_0
                ):
                    assert value_sigma_1 == value_sigma_0

    def test_sigma_callable_equals_sigma_array(self):
        for n, k in INDICES:
            x = np.linspace(0, 1, n)
            l = int(np.log2(n // k))
            sigma_list_0 = step_0.sigma_list(x, k, 0)
            sigma_j_i_0 = step_0.sigma_callable(x, k, 0)
            sigma_callable_0 = [
                [
                    sigma_j_i_0(j, i)
                    for i in range(n // (k * 2**(j + 1)))
                ]
                for j in range(l)
            ]
            # assert that the results produced are equal
            assert len(sigma_list_0) == len(sigma_callable_0)
            for (values_sigma_list, values_sigma_callable) in zip(
                sigma_list_0, sigma_callable_0
            ):
                assert len(values_sigma_list) == len(values_sigma_callable)
                for (value_sigma_list, value_sigma_callable) in zip(
                    values_sigma_list, values_sigma_callable
                ):
                    assert value_sigma_list == value_sigma_callable


Test_s_values  # pylint: disable=pointless-statement

Test_mu_values  # pylint: disable=pointless-statement

Test_sigma_values  # pylint: disable=pointless-statement
