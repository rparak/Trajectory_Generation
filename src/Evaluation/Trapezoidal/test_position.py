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

    s_0 = [{'Position': Mathematics.Degree_To_Radian(-10.0), 'Velocity': 0.0, 'Acceleration': 0.0},
           {'Position': Mathematics.Degree_To_Radian(10.0), 'Velocity': 0.0, 'Acceleration': 0.0},
           {'Position': Mathematics.Degree_To_Radian(-45.0), 'Velocity': 0.0, 'Acceleration': 0.0}]
    s_f = [{'Position': Mathematics.Degree_To_Radian(90.0), 'Velocity': 0.0, 'Acceleration': 0.0},
           {'Position': Mathematics.Degree_To_Radian(-90.0), 'Velocity': 0.0, 'Acceleration': 0.0},
           {'Position': Mathematics.Degree_To_Radian(45.0), 'Velocity': 0.0, 'Acceleration': 0.0}]

    Trapezoidal_Cls = Lib.Trajectory.Profile.Trapezoidal_Cls(100)

    for i, (s_0_i, s_f_i) in enumerate(zip(s_0, s_f)):
        (s, s_dot, s_ddot) = Trapezoidal_Cls.Generate(s_0_i, s_f_i)
    
if __name__ == "__main__":
    sys.exit(main())