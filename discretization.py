"""discretization"""

__all__ = ["gauleg", "trapezoidal"]
__author__ = "Alexandre Pierre"
__email__ = "alexandrempierre [at] gmail [dot] com"


import math

from collections.abc import Callable

from scipy.special import eval_legendre


def gauleg(
    x1: float, x2: float, n: int, eps: float = 1e-14
) -> tuple[list[float], list[float]]:
    xs = [0] * n
    ws = [0] * n
    m = (n + 1) // 2
    xm = (x2 + x1) / 2
    xl = (x2 - x1) / 2
    for i in range(m):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        while True:
            p1 = 1
            p2 = 0
            for j in range(n):
                p3, p2 = p2, p1
                p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1)
            pp = n * (z * p1 - p2) / (z * z - 1)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= eps:
                break
        xs[i] = xm - xl * z
        xs[n - 1 - i] = xm + xl * z
        ws[i] = 2 * xl / ((1 - z * z) * pp * pp)
        ws[n - 1 - i] = ws[i]
    return xs, ws


def trapezoidal(x1: float, x2: float, n: int) -> tuple[list[float], list[float]]:
    xs = [x1 + (x2 - x1) * step / (n - 1) for step in range(n)]
    ws = (
        [(xs[1] - xs[0]) / 2]
        + [(x_next - x_prev) / 2 for x_prev, x_next in zip(xs[:-2], xs[2:])]
        + [(xs[-1] - xs[-2]) / 2]
    )
    return xs, ws


def condition_number(
    xs: list[float], ws: list[float], kernel: Callable[[float, float], float]
) -> float:
    pass


if __name__ == "__main__":
    roots, weights = gauleg(-1, 1, 9)
    # print(len(weights))
    print(*eval_legendre(9, roots), sep="\n")
