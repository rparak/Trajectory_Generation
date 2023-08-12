# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Time x ∈ [0: The starting value of the sequence, 
#           1: The end value of the sequence] {0.0 <= x <= 1.0}
CONST_T_0 = 0.0
CONST_T_1 = 1.0

class Polynomial_Cls(object):
    """
    Description:
        ...
    """
            
    def __init__(self, N: int) -> None:
        # The value of the time must be within the interval: 
        #   0.0 <= t <= 1.0
        self.__t = np.linspace(CONST_T_0, CONST_T_1, N)

        self.__A  = self.__Quintic_Polynomial()

    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Time.
                                                Note:
                                                    Where n is the number of points.
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
                
        return self.__t.shape[0]
    
    def __Quintic_Polynomial(self):
        """
        Descrtiption:
            Quintic Polynomial (5th degree)

            # http://oramosp.epizy.com/teaching/18/robotics/lectures/Topic7_Trajectory_Generation.pdf?i=1
            # http://www.mnrlab.com/uploads/7/3/8/3/73833313/trajectory.pdf
            # https://www.scribd.com/presentation/434786742/8-QUINTIC-and-LFSB-Trajectory-planning-20-Sep-2018Reference-Material-I-Quintic-Polynomial-Trajectory-pptx#
            # https://www.youtube.com/watch?v=HqQBL6xcj4w
            
            s(t) = 
        """
        
        t_0 = self.__t[0]; t_f = self.__t[-1]

        return np.array([[1.0, t_0,    t_0**2,       t_0**3,        t_0**4,        t_0**5],
                         [0.0, 1.0, 2.0 * t_0, 3.0 * t_0**2,  4.0 * t_0**3,  5.0 * t_0**4],
                         [0.0, 0.0,       2.0,    6.0 * t_0, 12.0 * t_0**2, 20.0 * t_0**3],
                         [1.0, t_f,    t_f**2,       t_f**3,        t_f**4,        t_f**5],
                         [0.0, 1.0, 2.0 * t_f, 3.0 * t_f**2,  4.0 * t_f**3,  5.0 * t_f**4],
                         [0.0, 0.0,       2.0,    6.0 * t_f, 12.0 * t_f**2, 20.0 * t_f**3]], dtype=np.float32)

    def Generate(self, s_0, s_f):
        """
        Description:
            ...
        """

        # Initialization of the output varliables.
        s = np.zeros(self.N, dtype=np.float32)
        s_dot = s.copy(); s_ddot = s.copy()

        # Find the coefficients of the polynomial of degree 5 (quintic).
        #   Note:
        #       The coefficients are of the form x^0 + x^1 + .. + x^n, 
        #       where n is equal to 5.
        C = np.linalg.inv(self.__A).dot(np.append(list(s_0.values()), 
                                                  list(s_f.values())))
        
        # Analytic expression (position):
        #   s(t) = a_{0} + a_{1}*t + a_{2}*t^2 + a_{3}*t^3 + a_{4}*t^4 + a_{5}*t^5
        for i, C_i in enumerate(C):
            s[:] += (self.__t ** i) * C_i
        
        # Analytic expression (velocity):
        #   s_dot(t) = a_{1} + 2*a_{2}*t + 3*a_{3}*t^2 + 4*a_{4}*t^3 + 5*a_{5}*t^4
        for i, (C_i_dot, var_i) in enumerate(zip(C[1:], np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)), 
                                             start=1):
            s_dot[:] += (self.__t ** (i - 1)) * C_i_dot * var_i

        # Analytic expression (acceleration):
        #   s_ddot(t) = 2*a_{2}*t + 6*a_{3}*t^2 + 12*a_{4}*t^3 + 20*a_{5}*t^4
        for i, (C_i_ddot, var_i) in enumerate(zip(C[2:], np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)), 
                                              start=2):
            s_ddot[:] += (self.__t ** (i - 2)) * C_i_ddot * var_i

        return (s, s_dot, s_ddot)
    
class Trapezoidal_Cls(object):
    def __init__(self, N: int) -> None:
        # The value of the time must be within the interval: 
        #   0.0 <= t <= 1.0
        self.__t = np.linspace(CONST_T_0, CONST_T_1, N)
    
    @property
    def t(self) -> tp.List[float]:
        """
        Description:
           Get the time as an interval of values from 0 to 1.
        
        Returns:
            (1) parameter [Vector<float> 1xn]: Time.
                                                Note:
                                                    Where n is the number of points.
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
                
        return self.__t.shape[0]
    
    def Generate(self):
        return True