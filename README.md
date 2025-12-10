# GRiD Demo: "Hello World" for GRiD Python

This package contains an example to demonstrate the functionality of the CUDA-based GRiD (**G**PU-Accelerated **Ri**gid Body **D**ynamics) library. The repository can be used to verify that GRiD Python is set up and running correctly on your machine.

GRiD is a library of functions that computes physics quantities needed for trajectory planning algorithms by generating CUDA kernels for your specific robot.

**Example GRiD Usage: Trajectory Planning**
1. A target pose is defined using joint angles or end-effector positions and converted into a series of waypoints from the current pose.
2. A motor control algorithm is used to calculate the torques needed to move the robot to the next waypoint.
3. GRiD is invoked and `forward_dynamics` is called to convert the calculated torques into accelerations
4. An integrator is used to convert the accelerations into updated velocities and positions.
5. The robot moves to the next waypoint, and steps 2-4 is repeated until the final target position is reached.

An example of this use case can be found in the implementation of KKT systems solvers in [MPCGPU](https://github.com/A2R-Lab/MPCGPU/tree/0efde8c63c38465bba630ba569c4f8a30c1b009c).

## Installation
This package contains several layers of submodules! Make sure to run `git submodule update --init --recursive` after cloning.

### Building GRiD Python

    pip install GRiD/bindings

### Demo
To test that you have installed the GRiD bindings correct, run:

    python hello_grid.py

## Usage
To use the GRiD Python in your project, include `from GRiD.bindings import gridCuda` at the top of your file.

## API Reference
[Full API Reference and Documentation](#api-reference)

### Classes
- `GRiDDataFloat(q, qd, u)`: Single-precision (float) implementation of RBD functions
- Note that the double implementation caused errors

### Functions
- `load_joint_info(q, qd, u)`: Update the input parameters for RBD calculations
- `get_end_effector_positions()`: Calculates end-effector poses
- `get_end_effector_position_gradients()`: Calculates end-effector pose gradients
- `inverse_dynamics()`: Calculates the RNEA torque vector
- `minv()`: Calculates the inverse of the mass matrix
- `forward_dynamics()`: Calculates joint accelerations
- `inverse_dynamics_gradient()`: Calculates Jacobian of inverse dynamics w.r.t *q* and *qd*
- `forward_dynamics_gradient()`: Calculates Jacobian of forward dynamics w.r.t *q* and *qd*

### Variables
- `NUM_JOINTS`: Number of joints defined in the URDF
- `NUM_EES`: Number of end-effectors based on the URDF specification


## Requirements

    C++11 compatible compiler
    CUDA Toolkit >= 11.1 (compatible with compute capability 8.6)
    CMake >= 3.10
    Python >= 3.6
    pybind11