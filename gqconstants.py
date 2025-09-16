
from decimal import Decimal, getcontext
from typing import List
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15,10)
class HighPrecisionGaussInt:
    """
    High precision Gaussian quadrature integration using decimal arithmetic
    Equivalent to __float128 precision (~34 decimal digits)
    """

    def __init__(self, npoints: int, precision: int = 40):
        # Set precision higher than __float128 (~34 digits) for intermediate calculations
        getcontext().prec = precision

        self.lroots: List[Decimal] = []
        self.weight: List[Decimal] = []
        self.lcoef: List[List[Decimal]] = []
        self.precision = precision

        # High precision constants
        self.PI = Decimal(str(math.pi)).quantize(Decimal(10) ** -(precision-5))
        # More precise pi calculation
        self.PI = self._calculate_pi_high_precision()

        self.init(npoints)

    def _calculate_pi_high_precision(self) -> Decimal:
        """Calculate pi to high precision using Machin's formula"""
        # pi/4 = 4*arctan(1/5) - arctan(1/239)
        getcontext().prec = self.precision + 10  # Extra precision for intermediate calc

        def arctan_series(x: Decimal, terms: int = None) -> Decimal:
            if terms is None:
                terms = self.precision + 5

            x_squared = x * x
            result = x
            term = x

            for n in range(1, terms):
                term *= -x_squared
                result += term / (2 * n + 1)

            return result

        one_fifth = Decimal(1) / Decimal(5)
        one_239 = Decimal(1) / Decimal(239)

        pi_quarter = 4 * arctan_series(one_fifth) - arctan_series(one_239)
        pi = 4 * pi_quarter

        getcontext().prec = self.precision  # Reset precision
        return pi

    def _cos_high_precision(self, x: Decimal) -> Decimal:
        """Calculate cosine using Taylor series"""
        # Reduce x to [0, 2*pi)
        two_pi = 2 * self.PI
        x = x % two_pi

        # Use Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
        result = Decimal(1)
        term = Decimal(1)
        x_squared = x * x

        for n in range(1, self.precision + 5):
            term *= -x_squared / (Decimal(2*n-1) * Decimal(2*n))
            result += term

            # Stop if term becomes negligible
            if abs(term) < Decimal(10) ** -(self.precision + 2):
                break

        return result

    def _abs_decimal(self, x: Decimal) -> Decimal:
        """Absolute value for Decimal"""
        return x if x >= 0 else -x

    def lege_eval(self, n: int, x: Decimal) -> Decimal:
        """Evaluate Legendre polynomial at x using Horner's method"""
        s = self.lcoef[n][n]
        for i in range(n, 0, -1):
            s = s * x + self.lcoef[n][i - 1]
        return s

    def lege_diff(self, n: int, x: Decimal) -> Decimal:
        """Evaluate derivative of Legendre polynomial at x"""
        n_dec = Decimal(n)
        return n_dec * (x * self.lege_eval(n, x) - self.lege_eval(n - 1, x)) / (x * x - Decimal(1))

    def init(self, npoints: int):
        """
        Calculates abscissas and weights to high precision
        for n-point quadrature rule
        """
        # Initialize arrays
        self.lroots = [Decimal(0)] * npoints
        self.weight = [Decimal(0)] * npoints
        self.lcoef = [[Decimal(0) for _ in range(npoints + 1)] for _ in range(npoints + 1)]

        # Initialize Legendre polynomial coefficients
        self.lcoef[0][0] = Decimal(1)
        self.lcoef[1][1] = Decimal(1)

        # Generate Legendre polynomial coefficients using recurrence relation
        for n in range(2, npoints + 1):
            n_dec = Decimal(n)
            n_minus_1 = Decimal(n - 1)

            self.lcoef[n][0] = -n_minus_1 * self.lcoef[n - 2][0] / n_dec

            for i in range(1, n + 1):
                two_n_minus_1 = Decimal(2 * n - 1)
                self.lcoef[n][i] = ((two_n_minus_1 * self.lcoef[n - 1][i - 1] -
                                   n_minus_1 * self.lcoef[n - 2][i]) / n_dec)

        # Find roots using Newton-Raphson method
        eps = Decimal(10) ** -(self.precision - 5)  # High precision tolerance

        for i in range(1, npoints + 1):
            # Initial guess using asymptotic formula
            i_dec = Decimal(i)
            npoints_dec = Decimal(npoints)

            # x = cos(π * (i - 0.25) / (npoints + 0.5))
            angle = self.PI * (i_dec - Decimal('0.25')) / (npoints_dec + Decimal('0.5'))
            x = self._cos_high_precision(angle)

            # Newton-Raphson iteration
            max_iterations = 100
            for iteration in range(max_iterations):
                x1 = x

                # Newton-Raphson step: x_new = x - f(x)/f'(x)
                f_val = self.lege_eval(npoints, x)
                f_prime = self.lege_diff(npoints, x)

                if f_prime == 0:
                    break

                x = x - f_val / f_prime

                # Check convergence
                if self._abs_decimal(x - x1) <= eps:
                    break

            # Store root
            self.lroots[i - 1] = x

            # Calculate weight
            x1 = self.lege_diff(npoints, x)
            self.weight[i - 1] = Decimal(2) / ((Decimal(1) - x * x) * x1 * x1)

    def PrintWA(self):
        # Print results with high precision
        print(f"==== {len(self.weight)} ====")
        for i in range(len(self.weight)):
            print(f"Weight: {self.weight[i]}")
            print(f"Root:   {self.lroots[i]}")
            print()

    def integ(self, f, a: float, b: float) -> Decimal:
        """
        Integrate function f from a to b using Gaussian quadrature
        """
        a_dec = Decimal(str(a))
        b_dec = Decimal(str(b))

        c1 = (b_dec - a_dec) / Decimal(2)
        c2 = (b_dec + a_dec) / Decimal(2)
        sum_val = Decimal(0)

        for i in range(len(self.weight)):
            # Convert to float for function evaluation, then back to Decimal
            x_eval = float(c1 * self.lroots[i] + c2)
            f_val = Decimal(str(f(x_eval)))
            sum_val += self.weight[i] * f_val

        return c1 * sum_val

def integ_trap(f, a: float, b: float, n) -> Decimal:
    """
    Integrate function f from a to b using trapezoidal rule
    """

    ### complete code here ###
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    I = ((h/2)*(y[0] + 2*np.sum(y[1:-1]) +y[-1]))

    return I  # integral of function f
def integ_simp(f, a: float, b: float, n) -> Decimal:
    """
    Integrate function f from a to b using Simpson's rule
    """
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    I = ((h/3)*(y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) +y[-1]))
    return I
    ### complete code here ###
def compute_error(numerical, exact):
  return abs(float(numerical) - float(exact)) / abs(float(exact))
  # Example usage and testing
if __name__ == "__main__":
    import sys
    # Create high precision integrator
    #print(f"Creating {order}-point Gaussian quadrature with high precision...")
    #gauss_hp = HighPrecisionGaussInt(order, precision=40)
    #gauss_hp.PrintWA()
    exact = 1 - np.exp(-1)

    N = [2,4,8,16,32,64,128]

    error_gauss_list = []
    error_trap_list = []
    error_simp_list = []
    n_log_list = []
    for n in N:
        gauss_hp = HighPrecisionGaussInt(n, precision=15)
        numerical_gauss = gauss_hp.integ(lambda x: np.exp(-x), 0, 1)
        error_gauss = compute_error(numerical_gauss, exact)
        numerical_trap = integ_trap(lambda x: np.exp(-x), 0, 1, n)
        error_trap = compute_error(numerical_trap, exact)
        numerical_simp = integ_simp(lambda x: np.exp(-x), 0, 1, n)
        error_simp = compute_error(numerical_simp, exact)


        if error_simp != 0:
          error_simp_list.append((error_simp))
        elif error_simp == 0.0:
          error_simp_list.append(0)
        if error_trap != 0:
          error_trap_list.append((error_trap))
        elif error_trap == 0.0:
          error_trap_list.append(0)
        if error_gauss != 0:
          error_gauss_list.append((error_gauss))
        elif error_gauss == 0.0:
          error_gauss_list.append(0)
        n_log_list.append(np.log10(n))
        print(f"N: {n} - Error (gauss): {error_gauss} - Error (simpson's): {error_simp} - Error (trapezoidal): {error_trap}")


    plt.style.use("dark_background")
    plt.title(r"$\log_{10}(N)$ vs $\log_{10}(|error|)$ for Various Numerical Integration Methods Acting on $\int_{0}^{1}e^{-x}dx$")
    plt.loglog(N, error_gauss_list, label="Gaussian Quadrature", color = "magenta")
    plt.loglog(N, error_simp_list, label="Simpson's Rule", color = "cyan")
    plt.loglog(N, error_trap_list, label="Trapezoidal Rule", color = "orange")
    plt.xlabel("log10(N)")
    plt.ylabel("log10(Error)")
    plt.legend()
    plt.show()
    plt.savefig("Errors.png")
