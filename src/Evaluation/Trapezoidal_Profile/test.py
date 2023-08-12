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

    # Initialization of multi-axis parameters for trajectory generation.
    s_0 = [Mathematics.Degree_To_Radian(10.0)]
    s_f = [Mathematics.Degree_To_Radian(90.0)]

    # Initialization of the class to generate trajectory.
    Trapezoidal_Cls = Lib.Trajectory.Profile.Trapezoidal_Cls(N=100)
    
    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of multi-axis trajectories.
    for i, (s_0_i, s_f_i) in enumerate(zip(s_0, s_f)):
        # Generation of position trajectories from input parameters.
        (s, _, _) = Trapezoidal_Cls.Generate(s_0_i, s_f_i)

        ax.plot(s, '.--', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, label=r'$s_{%d}(t)$' % (i + 1))

    # Set parameters of the graph (plot).
    ax.set_title(r'Trajectory Trapezoidal Profile: Position', fontsize=25, pad=25.0)
    #   Set the x ticks.
    #ax.set_xticks(np.arange(np.min(Trapezoidal_Cls.t) - 0.1, np.max(Trapezoidal_Cls.t) + 0.1, 0.1))
    #   Label
    ax.set_xlabel(r't', fontsize=15, labelpad=10)
    ax.set_ylabel(r's(t)', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
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
        plt.savefig(f'{project_folder}/images/Trapezoidal_Profile/position.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()

    
if __name__ == "__main__":
    sys.exit(main())