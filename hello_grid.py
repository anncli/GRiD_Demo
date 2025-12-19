"""
This is a basic demo using GRiD Python to simulate moving the front left
HFE (hip flexion/extension) joint of a HyQ quadruped robot using inverse
and forward dynamics with a basic PD controller.
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


# DEMO CONFIGS
dt = 0.01    # time steps
T  = 1.0     # total time

Kp = 150.0   # PD contoller proportional gain
Kd = 30.0    # PD controller derivative gain

LF_HFE = 1   # URDF Joint ID

def computed_torque_demo():
    nj = gridCuda.NUM_JOINTS
    steps = int(T / dt)

    q  = np.zeros(nj, dtype=np.float32)
    qd = np.zeros(nj, dtype=np.float32)

    q_des  = np.zeros(nj, dtype=np.float32)
    qd_des = np.zeros(nj, dtype=np.float32)

    q_des[LF_HFE] = 0.4  # desired target position for LF_HFE joint

    # For logging
    q_hist = []
    tau_hist = []
    qdd_des_hist = []
    qdd_hist = []

    print("Running demo...")
    print(f"Target HFE angle: {q_des[LF_HFE]:.3f} rad")

    for _ in range(steps):
        # 1. Find required acceleration to reach target
        qdd_des = (
            Kp * (q_des - q) +
            Kd * (qd_des - qd)
        ).astype(np.float32)

        # 2. Find torque needed for acceleration considering mass, gravity, and inertia
        # 2a. Bias forces (C + g)
        grid = gridCuda.GRidDataFloat(q, qd, np.zeros(nj, dtype=np.float32))
        with suppress_stdout_stderr():
            bias = grid.inverse_dynamics()

        # 2b. Mass matrix inverse to solve for required torques
        with suppress_stdout_stderr():
            Minv = grid.minv()

        # 2c. Compute total torque
        tau = bias + np.linalg.solve(Minv, qdd_des)

        # 3. Forward dynamics (simulating physics) using computed torque
        grid = gridCuda.GRidDataFloat(q, qd, tau)
        with suppress_stdout_stderr():
            qdd = grid.forward_dynamics()

        # 4. Update joint states with Euler integration
        qd += qdd * dt
        q  += qd  * dt

        # 5. Log LF_HFE joint values for plotting
        q_hist.append(q[LF_HFE])
        tau_hist.append(tau[LF_HFE])
        qdd_des_hist.append(qdd_des[LF_HFE])
        qdd_hist.append(qdd[LF_HFE])

    print(f"Final LF_HFE angle: {q[LF_HFE]:.3f} rad")
    print(f"Final error: {(q_des[LF_HFE] - q[LF_HFE]):.3f} rad")
    return np.array(q_hist), np.array(tau_hist), np.array(qdd_des_hist), np.array(qdd_hist), q_des[LF_HFE]


if __name__ == "__main__":
    q_hist, tau_hist, qdd_des_hist, qdd_hist, q_des = computed_torque_demo()
    t = np.linspace(0, T, len(q_hist))

    PLOTS_DIRECTORY = "plots"
    if not os.path.exists(PLOTS_DIRECTORY):
        os.makedirs(PLOTS_DIRECTORY)

    # Joint angle over time
    plt.figure()
    plt.plot(t, q_hist, label="actual")
    plt.axhline(q_des, linestyle="--", label="target")
    plt.title("LF_HFE Joint Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("rad")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOTS_DIRECTORY}/lf_hfe_angle.png", dpi=150)
    plt.close()

    # Torque over time
    plt.figure()
    plt.plot(t, tau_hist)
    plt.title("LF_HFE Torque")
    plt.xlabel("Time (s)")
    plt.ylabel("Nm")
    plt.grid()
    plt.savefig(f"{PLOTS_DIRECTORY}/lf_hfe_torque.png", dpi=150)
    plt.close()

    # Desired vs. actual acceleration over time
    plt.figure()
    plt.plot(t, qdd_des_hist, label="desired")
    plt.plot(t, qdd_hist, linestyle="--", label="actual")
    plt.title("LF_HFE Acceleration Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("rad/s²")
    plt.legend()
    plt.grid()
    plt.savefig(f"{PLOTS_DIRECTORY}/lf_hfe_accel_tracking.png", dpi=150)
    plt.close()

    print(f"Saved plots to {PLOTS_DIRECTORY} directory:")
    print(" - lf_hfe_angle.png")
    print(" - lf_hfe_torque.png")
    print(" - lf_hfe_accel_tracking.png")
