"""
This is a basic demo using GRiD Python to simulate the trajectory of a HyQ Qudraped's
joint and end effector positions under a sinusoidal torque input.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import contextlib
from GRiD.bindings import gridCuda

# Temporary solution to suppress GRiD's printf outputs
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def simulate_simple_trajectory(dt=0.001, T=1.0):
    print("Running GRiD Hello World Trajectory Demo...\n")

    nj = gridCuda.NUM_JOINTS   # number of joints in the robot model
    steps = int(T / dt)        # number of simulation timesteps

    # ----------------------------------------------------
    # 1. INITIAL CONDITIONS
    # ----------------------------------------------------
    q  = np.zeros(nj, dtype=np.float32)   # joint positions    q(t)
    qd = np.zeros(nj, dtype=np.float32)   # joint velocities   q̇(t)
    u  = np.zeros(nj, dtype=np.float32)   # joint torques      τ(t)

    history = []   # to record q₀(t) over time
    ee_history = []  # to record end-effector positions over time

    # ----------------------------------------------------
    # 2. SIMULATION LOOP
    # ----------------------------------------------------
    for i in range(steps):
        t = i * dt

        # ------------------------------------------------
        # 2a. CONTROL INPUT (u): torque applied
        #     Apply a sinusoidal torque on joint 0 (LF_HAA)
        # ------------------------------------------------
        u[0] = 2.0 * np.sin(2 * np.pi * t)

        # ------------------------------------------------
        # 2b. PHYSICS COMPUTATION (q̈ = f(q, q̇, u))
        #     Create a GRiD data object for this timestep
        # ------------------------------------------------
        grid = gridCuda.GRidDataFloat(q, qd, u)
        with suppress_stdout_stderr():
            qdd = grid.forward_dynamics() # GRiD → q̈(t)

        with suppress_stdout_stderr():
            ee = grid.get_end_effector_positions()   # shape (6, Nees)
        xyz = ee[:3]   # position of left foot end effector

        # ------------------------------------------------
        # 2c. INTEGRATION (Euler)
        #     Integrate q̇ and q using q̈ from GRiD
        #     This is the state-transition function
        # ------------------------------------------------
        qd += qdd * dt                    # velocity update
        q  += qd  * dt                    # position update

        # ------------------------------------------------
        # 2d. LOGGING
        # ------------------------------------------------
        history.append(q[1])              # record joint 1 (LF_HFE) position
        ee_history.append(xyz.copy())     # record LF end-effector position

    #print("\nFinal joint position:", q[0])
    return np.array(history), np.array(ee_history)


if __name__ == "__main__":
    traj, ee_traj = simulate_simple_trajectory()

    # Time axis
    dt = 0.001
    steps = len(traj)
    T = dt * steps
    t = np.linspace(0, T, steps)

    # Plot trajectory of the joint
    plt.figure(figsize=(8,4))
    plt.plot(t, traj, linewidth=2)
    plt.title("Joint 1 Position Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.grid(True)
    plt.savefig("trajectory1.png")
    print("Saved plot to trajectory1.png")

    plt.figure(figsize=(8,4))
    plt.plot(t, ee_traj[:,0], label="X")
    plt.plot(t, ee_traj[:,1], label="Y")
    plt.plot(t, ee_traj[:,2], label="Z")
    plt.title("End Effector Position Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.legend()
    plt.savefig("end_effector_xyz.png")
    print("Saved plot to end_effector_xyz.png")
