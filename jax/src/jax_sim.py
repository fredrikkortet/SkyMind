"""
Simple Flight Dynamics Simulation using JAX.

State vector: [x, y, z, vx, vy, vz]  (shape: (6,))
Dynamics: free flight (no forces), i.e. constant velocity.
"""

import argparse
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import jax.random
from functools import partial


# -- 1. Dynamics --------------------------------------------------------------

def flight_dynamics(state: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Time derivative of state under free-flight (no forces).

    Args:
        state: (6,) array  [x, y, z, vx, vy, vz]
        t:     scalar time (unused here, kept for generality)

    Returns:
        dstate/dt: (6,) array  [vx, vy, vz, 0, 0, 0]
    """
    velocity = state[3:]                    # extract vx, vy, vz
    acceleration = jnp.zeros(3)            # no forces -> zero acceleration
    return jnp.concatenate([velocity, acceleration])


# -- 2. RK4 Integrator --------------------------------------------------------

@partial(jax.jit, static_argnums=(0,))
def rk4_step(f, state: jnp.ndarray, t: float, dt: float) -> jnp.ndarray:
    """
    Single RK4 integration step.

    Args:
        f:     dynamics function  f(state, t) -> dstate/dt
        state: current state (6,)
        t:     current time
        dt:    timestep

    Returns:
        next_state: (6,)
    """
    k1 = f(state,              t)
    k2 = f(state + dt/2 * k1,  t + dt/2)
    k3 = f(state + dt/2 * k2,  t + dt/2)
    k4 = f(state + dt   * k3,  t + dt)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# -- 3. Simulation Loop (via jax.lax.scan) ------------------------------------

@partial(jax.jit, static_argnums=(3))
def simulate(initial_state: jnp.ndarray, t0: float, dt: float, n_steps: int):
    """
    Run the simulation for n_steps using jax.lax.scan.

    Args:
        initial_state: (6,) array
        t0:            initial time
        dt:            fixed timestep
        n_steps:       number of integration steps

    Returns:
        trajectory: (n_steps + 1, 6) array  (includes the initial state)
    """
    def step(carry, _):
        state, t = carry
        next_state = rk4_step(flight_dynamics, state, t, dt)
        return (next_state, t + dt), next_state

    _, trajectory_tail = jax.lax.scan(step, (initial_state, t0), None, length=n_steps)

    # Prepend the initial state so trajectory[0] == initial_state
    trajectory = jnp.concatenate([initial_state[None, :], trajectory_tail], axis=0)
    return trajectory


# Batched version: map simulate over N initial states -> (N, n_steps+1, 6)
simulate_batch = jax.vmap(simulate, in_axes=(0, None, None, None))


# -- 4. ACMI Export ------------------------------------------------------------

def export_acmi(trajectories, dt, t0, filename):
    """
    Export one or more trajectories to an ACMI 2.1 text file for TacView.

    Args:
        trajectories: (N, n_steps+1, 6) array — N vehicles
        dt:           timestep
        t0:           initial time
        filename:     output path

    Converts Cartesian (x=east, y=north, z=up) to lon/lat/alt using a
    simple equator approximation (1 deg ~ 111 320 m).
    """
    ref_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metres_per_deg = 111_320
    n_vehicles = trajectories.shape[0]
    n_frames = trajectories.shape[1]

    with open(filename, "w") as f:
        # Header
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")

        # Global properties (object ID 0)
        f.write(f"0,ReferenceTime={ref_time}\n")
        f.write("0,ReferenceLongitude=0\n")
        f.write("0,ReferenceLatitude=0\n")

        for i in range(n_frames):
            t = float(t0 + i * dt)
            f.write(f"#{t:.2f}\n")
            for v in range(n_vehicles):
                obj_id = v + 1
                x = float(trajectories[v, i, 0])
                y = float(trajectories[v, i, 1])
                z = float(trajectories[v, i, 2])
                lon = x / metres_per_deg
                lat = y / metres_per_deg
                alt = z
                if i == 0:
                    f.write(f"{obj_id},T={lon}|{lat}|{alt},Name=Vehicle_{obj_id},Type=Air+FixedWing\n")
                else:
                    f.write(f"{obj_id},T={lon}|{lat}|{alt}\n")

    print(f"  ACMI exported to: {filename}  ({n_vehicles} vehicles, {n_frames} frames)")


# -- 5. Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JAX flight dynamics simulation")
    parser.add_argument("-o", "--output", help="Export trajectory to ACMI file for TacView")
    parser.add_argument("-n", "--num-sims", type=int, default=1,
                        help="Number of parallel simulations (default: 1)")
    args = parser.parse_args()

    n_sims = args.num_sims

    # Initial conditions
    position_0    = jnp.array([0.0, 0.0, 1000.0])   # metres
    velocity_0    = jnp.array([100.0, 0.0, 0.0])     # m/s  (due east)
    initial_state = jnp.concatenate([position_0, velocity_0])

    # Simulation parameters
    dt      = 0.1    # seconds
    n_steps = 100    # -> 10 seconds of flight
    t0      = 0.0

    print("=" * 55)
    print("  JAX Flight Dynamics Simulation (free flight / RK4)")
    print("=" * 55)
    print("Backend: " + jax.default_backend())
    print("Devices:")
    for device in jax.devices():
        print(device)
    print("=" * 55)
    print(f"  Initial position : {position_0}")
    print(f"  Initial velocity : {velocity_0}")
    print(f"  dt = {dt} s  |  steps = {n_steps}  |  T = {dt*n_steps} s")
    print(f"  Vehicles         : {n_sims}")
    print("-" * 55)

    if n_sims == 1:
        trajectory = simulate(initial_state, t0, dt, n_steps)

        # Print every 10th step
        print(f"{'Step':>6}  {'Time':>6}  {'x':>10}  {'y':>10}  {'z':>10}  "
              f"{'vx':>8}  {'vy':>8}  {'vz':>8}")
        print("-" * 75)
        for i in range(0, n_steps + 1, 10):
            s = trajectory[i]
            t = t0 + i * dt
            print(f"{i:6d}  {t:6.1f}  "
                  f"{s[0]:10.3f}  {s[1]:10.3f}  {s[2]:10.3f}  "
                  f"{s[3]:8.3f}  {s[4]:8.3f}  {s[5]:8.3f}")

        # Sanity check: analytic solution for free flight
        T             = n_steps * dt
        expected_pos  = position_0 + velocity_0 * T
        simulated_pos = trajectory[-1, :3]
        error         = jnp.linalg.norm(simulated_pos - expected_pos)

        print("-" * 75)
        print(f"\n  Analytic final position : {expected_pos}")
        print(f"  Simulated final position: {simulated_pos}")
        print(f"  L2 error                : {error:.2e}")
        print("\n  Trajectory shape:", trajectory.shape)
        print("=" * 55)

        # For ACMI export, wrap as (1, n_steps+1, 6)
        trajectories = trajectory[None, :, :]
    else:
        # Generate randomized initial states: add ±50 m position and ±10 m/s velocity offsets
        key = jax.random.PRNGKey(42)
        pos_key, vel_key = jax.random.split(key)
        pos_offsets = jax.random.normal(pos_key, shape=(n_sims, 3)) * 50.0
        vel_offsets = jax.random.normal(vel_key, shape=(n_sims, 3)) * 10.0
        positions = position_0 + pos_offsets                      # (N, 3)
        velocities = velocity_0 + vel_offsets                     # (N, 3)
        initial_states = jnp.concatenate([positions, velocities], axis=1)  # (N, 6)

        trajectories = simulate_batch(initial_states, t0, dt, n_steps)  # (N, n_steps+1, 6)

        print(f"\n  Simulated {n_sims} vehicles  |  trajectories shape: {trajectories.shape}")
        print("=" * 55)

    if args.output:
        export_acmi(trajectories, dt, t0, args.output)

    return trajectories


if __name__ == "__main__":
    trajectory = main()
