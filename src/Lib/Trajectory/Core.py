# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Trajectory/Core
import Lib.Trajectory.Profile

"""
Description:
    Some useful references about the field of trajectory generation:
        1. Modern Robotics: Mechanics, Planning, and Control, Kevin M. Lynch and Frank C. Park
        2. Trajectory Planning for Automatic Machines and Robots by Luigi Biagiotti, Claudio Melchiorri
        3. Robotics, Vision and Control, Fundamental Algorithms in Python, Peter Corke
""" 

# Notes:
# https://github.com/novice1011/trajectory-planning
# https://repository.gatech.edu/server/api/core/bitstreams/34337c0d-c60a-4235-82da-7c87d275ff13/content
# https://www.slideshare.net/66551122/lecture-1-trajectory-generation
# 
class Multi_Point_Cls(object):
    def __init__(self, delta_time: float, t_blend: float, v_max: float) -> None:
        self.__dt = delta_time
        self.__t_blend = t_blend
        self.__v_max = v_max
        
    def Generate(self, P: tp.List[tp.List[float]]):
        """
        Description:
            ...

        Args:
            (1) P [Vector<float> mx1]: Input control points to be used for trajectory generation.
                                        Note:
                                        Where m is the number of points.
        """
        
        # Initialization of the class to generate trajectory.
        Polynomial_Cls = Lib.Trajectory.Profile.Polynomial_Cls(N=100)

        # ...
        s_previous = P[0]; s_dot_previous = np.array([0.0], dtype=np.float32)
        s = np.empty((0,1), dtype=np.float32)
        for i, P_i in enumerate(P[1::]):
            pass

        return (np.linspace(Lib.Trajectory.Profile.CONST_T_0, Lib.Trajectory.Profile.CONST_T_1, s.size), s)