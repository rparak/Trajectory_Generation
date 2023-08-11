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

class Polynomial_Profile_Cls(object):
    def __init__(self, s_0: tp.Union[float, tp.List[float]], s_1: tp.Union[float, tp.List[float]], N: int) -> None:
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
    
    @property
    def s(self):
        return None
    
    @property
    def s_dot(self):
        return None
    
    @property
    def s_ddot(self):
        return None
    
    def Generate(self):
        return True
    
class Trapezoidal_Profile_Cls(object):
    def __init__(self) -> None:
        pass

    @property
    def s(self):
        return None
    
    @property
    def s_dot(self):
        return None
    
    @property
    def s_ddot(self):
        return None
    
    def Generate(self):
        return True