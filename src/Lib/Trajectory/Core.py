"""
## =========================================================================== ## 
MIT License
Copyright (c) 2023 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: Core.py
## =========================================================================== ## 
"""

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Trajectory/Utilities
import Lib.Trajectory.Utilities

"""
Description:
    Some useful references about the field of trajectory generation:
        1. Modern Robotics: Mechanics, Planning, and Control, Kevin M. Lynch and Frank C. Park
        2. Trajectory Planning for Automatic Machines and Robots by Luigi Biagiotti, Claudio Melchiorri
        3. Introduction To Robotics, Mechanics and Control, John J. Craig
""" 

# Notes:
# https://github.com/novice1011/trajectory-planning
# https://repository.gatech.edu/server/api/core/bitstreams/34337c0d-c60a-4235-82da-7c87d275ff13/content
# https://www.slideshare.net/66551122/lecture-1-trajectory-generation
# books
# https://www.researchgate.net/publication/360441729_Multi-segment_trajectory_tracking_of_the_redundant_space_robot_for_smooth_motion_planning_based_on_interpolation_of_linear_polynomials_with_parabolic_blend
# https://www.researchgate.net/publication/325810192_Creating_Through_Points_in_Linear_Function_with_Parabolic_Blends_Path_by_Optimization_Method?enrichId=rgreq-65d1dcee4edc6a72c802c7a89774de87-XXX&enrichSource=Y292ZXJQYWdlOzMyNTgxMDE5MjtBUzo2Mzg0NzEyNTMyOTEwMDhAMTUyOTIzNDgxNzU3OA%3D%3D&el=1_x_3&_esc=publicationCoverPdf
# Turning Paths Into Trajectories Using Parabolic Blends
# https://github.com/SakshayMahna/Robotics-Mechanics/tree/main/Part-14-CubicInterpolation/tools
# http://timeconqueror.blogspot.com/2012/10/trajectory-generation-solution.html


# Trajectory with non-null initial and final velocities ...
# Finished this week.

class Multi_Point_Cls(object):
    """
    Description:
        ...

    Initialization of the Class:
        Args:
            (1) method [string]: ...
            (2) delta_time [float]: The difference (spacing) between the time values.

        Example:
            Initialization:
                # Assignment of the variables.
                method = 'Trepezoidal'
                delta_time = 0.01

                # Initialization of the class.
                Cls = Multi_Point_Cls(method, delta_time)

            Features:
                # Properties of the class.
                Cls.t
                ...
                Cls.N

                # Functions of the class.
                Cls.Generate()
    """

    def __init__(self, method: str, delta_time: float) -> None:
        try:
            method in ['Trapezoidal', 'Polynomial']

            self.__method = method

            # The difference (spacing) between the time values.
            self.__delta_time = delta_time

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
        
    @property
    def t(self) -> tp.List[float]:
        """
        Description:
            Get evenly spaced time values at the following interval, see below:
                t_0 <= t <= t_f,

                where t_0 is ... and t_f is .. .

        Returns:
            (1) parameter [Vector<float> 1xn]: Time values.
                                                Note:
                                                    Where n is the number of time values.
        """
                
        return self.__t
    
    @property
    def N(self) -> int:
        """
        Description:
           Get the number of time points of the trajectory.
        
        Returns:
            (1) parameter [int]: Number of time points.
        """
                
        return self.__t.size
    
    def Generate(self, P: tp.List[tp.List[float]], t: tp.List[float], t_blend: tp.List[float]) -> None:
        """
        Description:
            ...

        Args:
            (1) P [Vector<float> mx1]: Input control points (waypoints) to be used for trajectory generation.
                                        Note:
                                        Where m is the number of points.
            (2) ... []: ..
            (3) ... []: ..

        Returns:
            (1) parameter []: ...
        """

        return None
