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
import Lib.Trajectory.Core
# Custom Script:
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics
    
"""
Description:
    Initialization of constants.
"""
# Save the data to a file.
CONST_SAVE_DATA = False
# The method of the multi-segment trajectory generation. 
#   Note:
#       method = 'Trapezoidal', 'Polynomial'
CONST_METHOD = 'Trapezoidal'

def main():
    """
    Description:
       A program to generate ....

        Further information can be found in the programme below.
            ../Lib/Trajectory/Core.py
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Trajectory_Generation')[0] + 'Trajectory_Generation'

    # Initialization of multi-segment constraints for trajectory generation.
    #  1\ Input control points (waypoints) to be used for trajectory generation.
    P = np.array([Mathematics.Degree_To_Radian(0.0), Mathematics.Degree_To_Radian(90.0), 
                  Mathematics.Degree_To_Radian(55.0), Mathematics.Degree_To_Radian(-15.0)], dtype=np.float32)
    #  2\ Trajectory duration between control points.
    delta_T = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    #  3\ Duration of the blend phase.
    t_blend = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Initialization of the class to generate multi-segment trajectory.
    MST_Cls = Lib.Trajectory.Core.Multi_Segment_Cls(CONST_METHOD, delta_time=0.1)
    
    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # Generation of position multi-segment trajectories from input parameters.
    (s, _, _, T, L) = MST_Cls.Generate(P, delta_T, t_blend)

    # Visualization of relevant structures.
    ax.plot(T, P, 'o--', color='#d0d0d0', linewidth=1.0, markersize = 8.0, 
            markeredgewidth = 4.0, markerfacecolor = '#ffffff', label='Control Points')
    ax.plot(MST_Cls.t, s, '.-', color='#ffbf80', linewidth=1.0, markersize = 3.0, 
            markeredgewidth = 1.5, label='Trajectory (L = %1.2f)' % L)
    
    # Set parameters of the graph (plot).
    ax.set_title(r'Multi-segment Linear Trajectory with %s Blends' % CONST_METHOD, fontsize=25, pad=25.0)
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(MST_Cls.t), np.max(MST_Cls.t), 0.5))
    #   Label
    ax.set_xlabel(r't', fontsize=15, labelpad=10)
    ax.set_ylabel(r's(t)', fontsize=15, labelpad=10) 
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
        plt.savefig(f'{project_folder}/images/Polynomial_Profile/position.png', format='png', dpi=300)
    else:
        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())