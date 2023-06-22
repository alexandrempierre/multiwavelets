'''módulo step_4
'''


__author__ = 'Alexandre Pierre'
__email__ = 'alexandrempierre [at] gmail [dot] com'


from .. import step_4


class Test_ZeroOrPowerOf2:
    # pylint: disable=invalid-name, missing-function-docstring
    '''teste com alguns valores para verificar se a função dá os resultados
esperados
'''
    def test_zero_or_power_of_2(self):
        powers_of_2 = {0} | {2**exponent for exponent in range(1_000)}
        numbers = set(range(100_000))
        for pow2, notpow2 in zip(powers_of_2, numbers - powers_of_2):
            assert step_4.is_zero_or_power_of_2(pow2)
            assert not step_4.is_zero_or_power_of_2(notpow2)


class TestExtraction:
    # pylint: disable=invalid-name, missing-function-docstring
    '''testes para conferir se o processo de extrar as submatrizes está correto
'''
    def test_(self):
        pass
