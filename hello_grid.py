import numpy as np
from GRiD.bindings import gridCuda

def simulate_simple_trajectory(dt=0.001, T=1.0):
    print("Running GRiD Hello World Trajectory Demo...\n")

    nj = gridCuda.NUM_JOINTS
    steps = int(T / dt)

    # Initial state
    q = np.zeros(nj, dtype=np.float32)
    qd = np.zeros(nj, dtype=np.float32)
    u = np.zeros(nj, dtype=np.float32)

    # Track trajectory of joint 0
    history = []

    for i in range(steps):
        t = i * dt

        # simple sinusoidal torque on joint 0
        u[0] = 2.0 * np.sin(2 * np.pi * t)

        # Create GRiD object for this state
        grid = gridCuda.GRidDataFloat(q, qd, u)

        # Compute joint accelerations using GRiD GPU dynamics
        qdd = grid.forward_dynamics()

        # Euler integration
        qd += qdd * dt
        q += qd * dt

        history.append(q[0])

        if i % (steps // 10) == 0:
            print(f"t = {t:.3f}  |  q[0] = {q[0]:.5f}")

    print("\nFinal joint position:", q[0])
    return np.array(history)


if __name__ == "__main__":
    traj = simulate_simple_trajectory()
