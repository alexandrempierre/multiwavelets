methods_s = ""
methods_mu = ""
methods_sigma = ""
for exp_n=2:16
  n = 2^exp_n
  x = LinRange(0, 1, n)
  for exp_k=0:(exp_n - 1)
    k = 2^exp_k
    l = convert(Int, log2(n / k))
    s = [(i - 1) * 2k for i=1:div(n, 2k)]
    mu = [
      [
        (x[1 + (i - 1)k * 2^j] + x[i*k*2^j]) / 2.0
        for i=1:div(n, k * 2^j)
      ] for j=1:l
    ]
    sigma = [
      [
        (x[i*k*2^j] - x[1 + (i - 1)k * 2^j]) / 2.0
        for i=1:div(n, k * 2^j)
      ] for j=1:l
    ]
    global methods_s *= """
    def test_s_values_with_n_$(n)_and_k_$(k)(self):
        n, k = $n, $k
        s = $s
        s_k = step_0.s_callable($(k))
        assert np.all(np.abs(s_k(i) - s[i]) < float_info.epsilon for i in range(n // (2 * k)))

"""
    global methods_mu *= """
    def test_mu_values_with_n_$(n)_and_k_$(k)(self):
        n, k = $n, $k
        x = np.linspace(0, 1, n)
        mu = $mu
        mu_list = step_0.mu_list(x, k, 0)
        assert len(mu) == len(mu_list)
        for mu_row, mu_list_row in zip(mu, mu_list):
            assert len(mu_row) == len(mu_list_row)
            assert np.all(np.abs(mu_value - mu_list_value) < float_info.epsilon for mu_value, mu_list_value in zip(mu_row, mu_list_row))

"""
    global methods_sigma *= """
    def test_sigma_values_with_n_$(n)_and_k_$(k)(self):
        n, k = $n, $k
        x = np.linspace(0, 1, n)
        sigma = $sigma
        sigma_list = step_0.sigma_list(x, k, 0)
        assert len(sigma) == len(sigma_list)
        for sigma_row, sigma_list_row in zip(sigma, sigma_list):
            assert len(sigma_row) == len(sigma_list_row)
            assert np.all(np.abs(sigma_value - sigma_list_value) < float_info.epsilon for sigma_value, sigma_list_value in zip(sigma_row, sigma_list_row))

"""
  end
end
open("test_step_0_values.py", "w") do io
  write(
    io,
    """__all__ = ['Test_s_values', 'Test_mu_values', 'Test_sigma_values']


import numpy as np
from sys import float_info
import step_0


class Test_s_values:
$methods_s
class Test_mu_values:
$methods_mu
class Test_sigma_values:
$methods_sigma
"""
  )
end;