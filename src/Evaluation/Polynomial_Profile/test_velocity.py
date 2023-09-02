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
#   ../Lib/Trajectory/Utilities
import Lib.Trajectory.Utilities
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
        A program to generate multi-axis velocity trajectories of fifth degree polynomials.

        Further information can be found in the programme below.
            ../Lib/Trajectory/Profile.py
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Trajectory_Generation')[0] + 'Trajectory_Generation'

    # Initialization of multi-axis constraints for trajectory generation.
    Ax_Constraints_0 = [np.array([Mathematics.Degree_To_Radian(10.0), 0.0, 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-10.0), 0.0, 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-45.0), 0.0, 0.0], dtype=np.float32)]
    Ax_Constraints_f = [np.array([Mathematics.Degree_To_Radian(90.0), 0.0, 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-90.0), 0.0, 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(45.0), 0.0, 0.0], dtype=np.float32)]

    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Lib.Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.01)
    
    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of multi-axis trajectories.
    for i, (ax_0_i, ax_f_i) in enumerate(zip(Ax_Constraints_0, Ax_Constraints_f)):
        # Generation of velocity trajectories from input parameters.
        (_, s_dot, _) = Polynomial_Cls.Generate(ax_0_i[0], ax_f_i[0], ax_0_i[1], ax_f_i[1], ax_0_i[2], ax_f_i[2], 
                                                0.0, 1.0)

        ax.plot(Polynomial_Cls.t, s_dot, '.-', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, label=r'$\dot{s}_{%d}(t)$' % (i + 1))

    # Set parameters of the graph (plot).
    ax.set_title(r'Trajectory Polynomial Profile', fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(Polynomial_Cls.t) - 0.1, np.max(Polynomial_Cls.t) + 0.1, 0.1))
    #   Label
    ax.set_xlabel(r't', fontsize=15, labelpad=10)
    ax.set_ylabel(r'$\dot{s}(t)$', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.75, linestyle = ':')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    if CONST_SAVE_DATA == True:
        # Set the full scree mode.
        plt.get_current_fig_manager().full_screen_toggle()

        # Save the results.
        plt.savefig(f'{project_folder}/images/Polynomial_Profile/velocity.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()


if __name__ == "__main__":
    sys.exit(main())