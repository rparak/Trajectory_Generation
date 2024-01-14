# An Open-Source Trajectory Generation Library Useful for Robotics Applications

## Requirements

**Programming Language**

```bash
Python
```

**Import Libraries**
```bash
More information can be found in the individual scripts (.py).
```

**Supported on the following operating systems**
```bash
Windows, Linux, macOS
```

## Project Description
An open-source library for generating trajectories using two different methods (trapezoidal, polynomial). The library provides access to various classes for working with multi-axis (Trapezoidal_Profile_Cls, Polynomial_Profile_Cls) trajectories as well as multi-segment (Multi_Segment_Cls) trajectories.

The trajectory profile, which contains position, velocity, and acceleration, is generated from input constraints explained in the individual classes.

The repository also contains a transformation library with the necessary project-related functions. See link below.

[/rparak/Transformation](https://github.com/rparak/Transformation)

The library can be used within the Robot Operating System (ROS), Blender, PyBullet, Nvidia Isaac, or any program that allows Python as a programming language.

## Multi-Axis Trajectory of a Trapezoidal Profile

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Script:
#   ../Trajectory/Utilities
import Trajectory.Utilities
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

def main():
    """
    Description:
        A program to generate multi-axis trapezoidal trajectories.

        Further information can be found in the programme below.
            ../Trajectory/Profile.py
    """

    # Initialization of multi-axis constraints for trajectory generation.
    Ax_Constraints_0 = [np.array([Mathematics.Degree_To_Radian(10.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-10.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-45.0), 0.0], dtype=np.float32)]
    Ax_Constraints_f = [np.array([Mathematics.Degree_To_Radian(90.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-90.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(45.0), 0.0], dtype=np.float32)]

    # Initialization of the class to generate trajectory.
    Trapezoidal_Cls = Trajectory.Utilities.Trapezoidal_Profile_Cls(delta_time=0.01)

    # Obtain multi-axis trajectories.
    for i, (ax_0_i, ax_f_i) in enumerate(zip(Ax_Constraints_0, Ax_Constraints_f)):
        # Generation of trajectories from input parameters.
        (s, s_dot, s_ddot) = Trapezoidal_Cls.Generate(ax_0_i[0], ax_f_i[0], ax_0_i[1], ax_f_i[1], 
                                                      0.0, 1.0)

if __name__ == '__main__':
    sys.exit(main())
```

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/position.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/velocity.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/acceleration.png width="600" height="350">
</p>

## Multi-Axis Trajectory of a Polynomial Profile

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Script:
#   ../Trajectory/Utilities
import Trajectory.Utilities
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

def main():
    """
    Description:
        A program to generate multi-axis polynomial trajectories.

        Further information can be found in the programme below.
            ../Trajectory/Profile.py
    """

    # Initialization of multi-axis constraints for trajectory generation.
    Ax_Constraints_0 = [np.array([Mathematics.Degree_To_Radian(10.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-10.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-45.0), 0.0], dtype=np.float32)]
    Ax_Constraints_f = [np.array([Mathematics.Degree_To_Radian(90.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(-90.0), 0.0], dtype=np.float32),
                        np.array([Mathematics.Degree_To_Radian(45.0), 0.0], dtype=np.float32)]

    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.01)

    # Obtain multi-axis trajectories.
    for i, (ax_0_i, ax_f_i) in enumerate(zip(Ax_Constraints_0, Ax_Constraints_f)):
        # Generation trajectories from input parameters.
        (s, s_dot, s_ddot) = Polynomial_Cls.Generate(ax_0_i[0], ax_f_i[0], ax_0_i[1], ax_f_i[1],
                                                     0.0, 1.0)

if __name__ == '__main__':
    sys.exit(main())
```

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Polynomial_Profile
$ ../Evaluation/Polynomial_Profile/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/position.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Polynomial_Profile
$ ../Evaluation/Polynomial_Profile/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/velocity.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Polynomial_Profile
$ ../Evaluation/Polynomial_Profile/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/acceleration.png width="600" height="350">
</p>

# Multi-Segment Linear Trajectory with Trapezoidal Blends

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Script:
#   ../Trajectory/Core
import Trajectory.Core
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

def main():
    """
    Description:
        A program to generate a multi-segment trajectory using the selected method.

        Possible methods of generating a multi-segment trajectory are as follows:
            1\ Trapezoidal (parabolic)
            2\ Polynomial (quintic)

        Further information can be found in the programme below.
            ../Trajectory/Core.py
    """

    # Initialization of multi-segment constraints for trajectory generation.
    #  1\ Input control points (waypoints) to be used for trajectory generation.
    P = np.array([Mathematics.Degree_To_Radian(0.0), Mathematics.Degree_To_Radian(90.0), 
                  Mathematics.Degree_To_Radian(55.0), Mathematics.Degree_To_Radian(-15.0)], dtype=np.float32)
    #  2\ Trajectory duration between control points.
    delta_T = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    #  3\ Duration of the blend phase.
    t_blend = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Initialization of the class to generate multi-segment trajectory.
    MST_Cls = Trajectory.Core.Multi_Segment_Cls('Trapezoidal', delta_time=0.1)

    # Generation multi-segment trajectories from input parameters.
    (s, s_dot, s_ddot, T, L) = MST_Cls.Generate(P, delta_T, t_blend)

if __name__ == '__main__':
    sys.exit(main())
```

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/position_Trapezoidal.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/velocity_Trapezoidal.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/acceleration_Trapezoidal.png width="600" height="350">
</p>

# Multi-Segment Linear Trajectory with Polynomial Blends

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Script:
#   ../Trajectory/Core
import Trajectory.Core
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

def main():
    """
    Description:
        A program to generate a multi-segment trajectory using the selected method.

        Possible methods of generating a multi-segment trajectory are as follows:
            1\ Trapezoidal (parabolic)
            2\ Polynomial (quintic)

        Further information can be found in the programme below.
            ../Trajectory/Core.py
    """

    # Initialization of multi-segment constraints for trajectory generation.
    #  1\ Input control points (waypoints) to be used for trajectory generation.
    P = np.array([Mathematics.Degree_To_Radian(0.0), Mathematics.Degree_To_Radian(90.0), 
                  Mathematics.Degree_To_Radian(55.0), Mathematics.Degree_To_Radian(-15.0)], dtype=np.float32)
    #  2\ Trajectory duration between control points.
    delta_T = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    #  3\ Duration of the blend phase.
    t_blend = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Initialization of the class to generate multi-segment trajectory.
    MST_Cls = Trajectory.Core.Multi_Segment_Cls('Polynomial', delta_time=0.1)

    # Generation multi-segment trajectories from input parameters.
    (s, s_dot, s_ddot, T, L) = MST_Cls.Generate(P, delta_T, t_blend)

if __name__ == '__main__':
    sys.exit(main())
```

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/position_Polynomial.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/velocity_Polynomial.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/acceleration_Polynomial.png width="600" height="350">
</p>

## Contact Info
Roman.Parak@outlook.com

## Citation (BibTex)
```bash
@misc{RomanParak_TrajectoryGeneration,
  author = {Roman Parak},
  title = {An open-source trajectory generation library useful for robotics applications},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://https://github.com/rparak/Trajectory_Generation}}
}
