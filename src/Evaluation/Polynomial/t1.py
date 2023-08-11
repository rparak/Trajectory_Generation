"""
PolynomialGenerator class definition
"""
from math import comb, factorial
import numpy as np


class PolynomialGenerator():
    """
    PolynomialGenerator generates polynomial trajectories from constraints
    """

    def __init__(self):
        pass

    def generate_coefficients(self, q_init, q_final, t_f, t_0=0.0):
        """
        Calculates polynomial trajectory coefficients from constraints

        Args:
            q_init (list): list of constraints: [q_init, dq_init, ddq_init]
            q_final (list): list of constraints: [q_final, dq_final, ddq_final]
            t_f (float): Finish time
            t_0 (float, optional): Start time

        Returns:
            np.array: Resultant polynomial coefficients
        """
        A_0 = self._get_constraint_submatrix(t_0)
        A_f = self._get_constraint_submatrix(t_f)

        A = np.vstack((A_0, A_f))
        q = np.hstack((np.array(q_init), np.array(q_final)))

        return np.linalg.inv(A).dot(q)

    def _get_constraint_submatrix(self, t):
        """
        Generates time constraint submatrix for time t

        Args:
            t (float): Time constraint

        Returns:
            np.ndarray: Time constraint submatrix
        """
        A = np.array([[1, t, t**2, t**3, t**4, t**5],
                      [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
                      [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3]])
        return A

    def polynomial_from_coefs(self, coefs, from_t, to_t, n):
        """
        Generates discrete polynomial of len(coefs) - 1 degree of size n

        Args:
            coefs (array-like): Coefficients of the polynomial to generate
            from_t (float): Start point
            to_t (float): End point
            n (int): Number of points

        Returns:
            np.ndarray: Array of n points from the polynomial
        """
        ts = np.linspace(from_t, to_t, n)
        poly = 0

        for i, coef in enumerate(coefs):
            poly += coef * ts**i

        return poly
    
pg = PolynomialGenerator()

coeff = pg.generate_coefficients([0.0,0.0,0.0], [1.57,0.0,0.0], 1.0, 0.0)
q = pg.polynomial_from_coefs(coeff, 0.0, 1.0, 10)

import matplotlib.pyplot as plt
plt.plot(q)
plt.show()