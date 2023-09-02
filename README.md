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

A simple program that describes how to work with the library can be found below. The whole program is located in the individual evaluation folder.

```py 
# System (Default)
import sys
# Numpy (Array computing) [pip3 install numpy]
import numpy as np

def main():
    pass

if __name__ == '__main__':
    sys.exit(main())
```

The repository also contains a transformation library with the necessary project-related functions. See link below.

[/rparak/Transformation](https://github.com/rparak/Transformation)

The library can be used within the Robot Operating System (ROS), Blender, PyBullet, Nvidia Isaac, or any program that allows Python as a programming language.

## Multi-Axis Trajectory of a Trapezoidal Profile

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/position.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/velocity.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trapezoidal_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trapezoidal_Profile/acceleration.png width="600" height="350">
</p>

## Multi-Axis Trajectory of a Polynomial Profile

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Polynomial_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/position.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Polynomial_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/velocity.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Polynomial_Profile
$ ../Evaluation/Trapezoidal_Profile/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Polynomial_Profile/acceleration.png width="600" height="350">
</p>

# Multi-Segment Linear Trajectory with Trapezoidal Blends

It is necessary to change one of the class input parameters to "**Trapezoidal**".

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/position_Trapezoidal.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/velocity_Trapezoidal.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/acceleration_Trapezoidal.png width="600" height="350">
</p>

# Multi-Segment Linear Trajectory with Polynomial Blends

It is necessary to change one of the class input parameters to "**Polynomial**".

**Position**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_position.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/position_Polynomial.png width="600" height="350">
</p>

**Velocity**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_velocity.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/velocity_Polynomial.png width="600" height="350">
</p>

**Acceleration**

```bash
$ /> cd Documents/GitHub/Trajectory_Generation/src/Evaluation/Trajectory
$ ../Evaluation/Trajectory/> python3 test_acceleration.py
```

<p align="center">
    <img src=https://github.com/rparak/Trajectory_Generation/blob/main/images/Trajectory/acceleration_Polynomial.png width="600" height="350">
</p>

## Contact Info
Roman.Parak@outlook.com

## Citation (BibTex)
```bash
@misc{RomanParak_DataConverter,
  author = {Roman Parak},
  title = {An open-source trajectory generation library useful for robotics applications},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://https://github.com/rparak/Trajectory_Generation}}
}
