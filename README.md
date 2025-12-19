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

## Installation and Usage
This package contains several layers of submodules! Make sure to run `git submodule update --init --recursive` after cloning.

**Note:** GRiD's dependencies are not be automatically installed. Make sure to follow the submodule's README to install any additional dependencies for GRiD.

### Building GRiD Python

    pip install GRiD/bindings

### Usage
To use the GRiD Python in your project, include `from GRiD.bindings import gridCuda` at the top of your file.

## Demo
To test the installation of the GRiD bindings, run:

    python hello_grid.py

### Demo Explanation
`hello_grid.py` is a minimal but complete "hello world" program using GRiD's Python Bindings. The goal of the demo is to verify that both forward dynamics and inverse dynamics are working correctly and to illustrate how GRiD fits into a typical robot control loop.

The demo simulates the motion of the front-left HFE (hip flexion/extension) joint of the [HyQ quadruped robot](https://iit-dlslab.github.io/papers/khan15phd.pdf) (see pages 83-85). Specifically, it:
1. Defines a target joint angle for the HFE joint.
2. Uses a simple PD controller to compute required joint accelerations.
3. Uses inverse dynamics to compute the motor torque required to achieve that acceleration, accounting for gravity and other bias.
4. Uses forward dynamics to simulate how the robot responds to that torque.
5. Integrates joint accelerations over time to update joint velocities and positions.
6. Plots joint motion and other control values for visualization.

These steps are repeated in a loop to move the joint to the desired angle and hold it there for a specified number of simulation time step.

### Control structure

The control loop implemented in the demo follows this structure:
1. define position error and velocity error
2. PD controller
3. desired acceleration
4. inverse dynamics
5. torque
6. forward dynamics
7. joint acceleration
8. Euler integration

This approach is commonly referred to as *computed-torque control*.

### Generated plots

When the demo is run, several plots are saved as PNG files in the plots directory.

#### 1. Joint angle over time (`lf_hfe_angle.png`)

This plot shows the actual HFE joint angle as the controller moves it toward the target.
- The horizontal axis is time (seconds)
- The vertical axis is joint angle (radians)
- A dashed line indicates the target angle

This plot helps check if the joint reach the desired position smoothly and accurately.

#### 2. Torque over time (`lf_hfe_torque.png`)

This plot shows the motor torque applied at the HFE joint.
- The horizontal axis is time (seconds)
- The vertical axis is torque (Nm)

This plot illustrates:
- How much effort the motor applies to move the joint
- The large initial torque needed to accelerate the leg
- The steady torque required to hold the joint against gravity once it reaches the target

#### 3. Desired vs. actual acceleration (lf_hfe_accel_tracking.png)

This plot compares:
- The desired/ideal joint acceleration computed by the PD controller
- The actual joint acceleration produced by GRiD's calculated dynamics

This plot demonstrates:
- How closely the robot follows the commanded acceleration
- The effect of inertia, gravity, and coupling on the system response
- Good overlap between the curves indicates that inverse and forward dynamics are consistent

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