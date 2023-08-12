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
import Lib.Trajectory.Profile
# Custom Script:
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics
    
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

    thetas_0 = Mathematics.Degree_To_Radian(np.array([ 0.0, 0.0], dtype=np.float32))
    thetas_1 = Mathematics.Degree_To_Radian(np.array([90.0, -90.0], dtype=np.float32))

    Polynomial_Cls = Lib.Trajectory.Profile.Polynomial_Cls(100)
    (thetas_s, thetas_dot, thetas_ddot) = Polynomial_Cls.Generate(thetas_0, thetas_1)

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of ....
    for i, thetas_s_i in enumerate(thetas_s.T):
        ax.plot(Polynomial_Cls.t, thetas_s_i, '.--', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, label=r'$\theta_{%d}$' % (i + 1))

    # Set parameters of the graph (plot).
    ax.set_title(f'Trajectory Polynomial Profile: Position s(t)', fontsize=25, pad=25.0)
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