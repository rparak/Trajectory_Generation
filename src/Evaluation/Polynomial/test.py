# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Trajectory/Core
import Lib.Trajectory.Core as Trajectory
# Custom Script:
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics


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
    
"""
Description:
    Initialization of constants.
"""
# Save the data to a file.
CONST_SAVE_DATA = False

def main():
    """
    Description:
        ...
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Trajectory_Generation')[0] + 'Trajectory_Generation'

    thetas_0 = Mathematics.Degree_To_Radian(np.array([ 0.0], dtype=np.float32))
    thetas_1 = Mathematics.Degree_To_Radian(np.array([90.0], dtype=np.float32))

    Polynomial_Cls = Trajectory.Polynomial_Profile_Cls(thetas_0, thetas_1, 100)
    Polynomial_Cls.Generate()

    pg = PolynomialGenerator()
    coeff = pg.generate_coefficients([0.0,0.0,0.0], [1.57,0.0,0.0], 1.0, 0.0)
    q = pg.polynomial_from_coefs(coeff, 0.0, 1.0, 100)


    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of ....
    for i, s_theta_i in enumerate(Polynomial_Cls.s.T):
        ax.plot(Polynomial_Cls.t, s_theta_i, '.--', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, label=r'$\theta_{%d}$' % (i + 1))

    ax.plot(Polynomial_Cls.t, q)

    # Set parameters of the graph (plot).
    ax.set_title(f'Trajectory Polynomial Profile: Position', fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(Polynomial_Cls.t) - 0.1, np.max(Polynomial_Cls.t) + 0.1, 0.1))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(np.append(thetas_0, thetas_1)) - 0.1, np.max(np.append(thetas_0, thetas_1)) + 0.1, 0.314))
    #   Label
    ax.set_xlabel(r'Normalized Time ', fontsize=15, labelpad=10)
    ax.set_ylabel(r'$\theta_{i}$ in radians', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    if CONST_SAVE_DATA == True:
        pass
    else:
        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())