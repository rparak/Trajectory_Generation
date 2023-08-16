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

def main():
    """
    Description:
       A program to generate multi-axis position trajectories of fifth degree polynomials.

        Further information can be found in the programme below.
            ../Lib/Trajectory/Profile.py
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Trajectory_Generation')[0] + 'Trajectory_Generation'

    # Initialization of the class to generate trajectory.
    MP_Trajectory_Cls = Lib.Trajectory.Core.Multi_Point_Cls(0.1, 10.0, 2.0)

    # ...
    (t, s) = MP_Trajectory_Cls.Generate(np.array([10.0, 60.0, 80.0, 10.0]))
    
if __name__ == "__main__":
    sys.exit(main())