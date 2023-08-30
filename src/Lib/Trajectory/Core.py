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
    def __init__(self, method: str, delta_time: float) -> None:
        try:
            method in ['Trapezoidal', 'Polynomial']

            self.__method = method

            # The difference (spacing) between the time values.
            self.__delta_time = delta_time

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
        
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
