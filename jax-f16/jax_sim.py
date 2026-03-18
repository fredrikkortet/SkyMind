"""
F-16 Fighter Plane Flight Dynamics Simulation using JAX.

Uses the high-fidelity F-16 nonlinear plant model from the fighterplane module
with aerodynamic lookup tables and quaternion-based attitude representation.
"""

import argparse
from datetime import datetime, timezone
from functools import partial

import jax
import jax.numpy as jnp
import jax.random

from fighterplane import FighterPlaneState, FighterPlaneControlState, update


# -- 1. Simulation Loop (via jax.lax.scan) ------------------------------------

@partial(jax.jit, static_argnums=(3,))
def simulate(initial_state: FighterPlaneState, action: FighterPlaneControlState,
             dt: float, n_steps: int):
    """
    Run the F-16 simulation for n_steps using jax.lax.scan.

    Args:
        initial_state: FighterPlaneState (scalar fields)
        action:        FighterPlaneControlState (constant control input)
        dt:            fixed timestep
        n_steps:       number of integration steps

    Returns:
        trajectory: FighterPlaneState with fields of shape (n_steps + 1,)
    """
    def step(state, _):
        next_state = update(state, action, dt)
        return next_state, next_state

    _, trajectory = jax.lax.scan(step, initial_state, None, length=n_steps)

    # Prepend the initial state so trajectory[0] == initial_state
    trajectory = jax.tree.map(
        lambda init, traj: jnp.concatenate([jnp.atleast_1d(init), traj]),
        initial_state, trajectory
    )
    return trajectory


# Batched version: map simulate over N initial states -> N trajectories
simulate_batch = jax.vmap(simulate, in_axes=(0, None, None, None))


# -- 2. ACMI Export ------------------------------------------------------------

def export_acmi(trajectories, n_vehicles, dt, t0, filename):
    """
    Export trajectories to an ACMI 2.1 text file for TacView.

    Args:
        trajectories: FighterPlaneState with fields of shape (n_steps+1,) or (N, n_steps+1)
        n_vehicles:   number of vehicles
        dt:           timestep
        t0:           initial time
        filename:     output path

    Converts north/east (metres) to lat/lon using a simple equator approximation.
    """
    ref_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metres_per_deg = 111_320

    if n_vehicles == 1:
        n_frames = trajectories.north.shape[0]
    else:
        n_frames = trajectories.north.shape[1]

    with open(filename, "w") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        f.write(f"0,ReferenceTime={ref_time}\n")
        f.write("0,ReferenceLongitude=0\n")
        f.write("0,ReferenceLatitude=0\n")

        for i in range(n_frames):
            t = float(t0 + i * dt)
            f.write(f"#{t:.2f}\n")
            for v in range(n_vehicles):
                obj_id = v + 1
                if n_vehicles == 1:
                    east = float(trajectories.east[i])
                    north = float(trajectories.north[i])
                    alt = float(trajectories.altitude[i])
                    roll_deg = float(jnp.degrees(trajectories.roll[i]))
                    pitch_deg = float(jnp.degrees(trajectories.pitch[i]))
                    yaw_deg = float(jnp.degrees(trajectories.yaw[i]))
                else:
                    east = float(trajectories.east[v, i])
                    north = float(trajectories.north[v, i])
                    alt = float(trajectories.altitude[v, i])
                    roll_deg = float(jnp.degrees(trajectories.roll[v, i]))
                    pitch_deg = float(jnp.degrees(trajectories.pitch[v, i]))
                    yaw_deg = float(jnp.degrees(trajectories.yaw[v, i]))

                lon = east / metres_per_deg
                lat = north / metres_per_deg

                if i == 0:
                    f.write(
                        f"{obj_id},T={lon}|{lat}|{alt}|{roll_deg}|{pitch_deg}|{yaw_deg},"
                        f"Name=F16_{obj_id},Type=Air+FixedWing\n"
                    )
                else:
                    f.write(f"{obj_id},T={lon}|{lat}|{alt}|{roll_deg}|{pitch_deg}|{yaw_deg}\n")

    print(f"  ACMI exported to: {filename}  ({n_vehicles} vehicles, {n_frames} frames)")


# -- 3. Helpers ----------------------------------------------------------------

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (rad) to stored quaternion (NED-to-body convention)."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)

    # Body-to-NED quaternion (ZYX order)
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    # Store as NED-to-body (conjugate)
    return q0, -q1, -q2, -q3


# -- 4. Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JAX F-16 flight dynamics simulation")
    parser.add_argument("-o", "--output", help="Export trajectory to ACMI file for TacView")
    parser.add_argument("-n", "--num-sims", type=int, default=1,
                        help="Number of parallel simulations (default: 1)")
    args = parser.parse_args()

    n_sims = args.num_sims

    # F-16 initial conditions (approximate trim at 10,000 ft / 502 ft/s)
    altitude_m = 3048.0      # ~10,000 ft
    vt_ms      = 153.0       # m/s (~502 ft/s)
    alpha_rad  = 0.039       # ~2.2 deg angle of attack
    pitch_rad  = 0.039       # match alpha for approximate level flight

    q0, q1, q2, q3 = euler_to_quaternion(0.0, pitch_rad, 0.0)

    # Approximate trim thrust (lbf) for level flight at this speed/altitude
    # action.throttle maps to T = throttle * 0.225 * 76300 / 0.3048 ≈ throttle * 56324
    initial_T = 2109.0       # ~0.0375 * 56324

    # Simulation parameters
    dt      = 0.01   # seconds (100 Hz — Euler integration needs small steps)
    n_steps = 1000   # -> 10 seconds of flight
    t0      = 0.0

    print("=" * 60)
    print("  JAX F-16 Flight Dynamics Simulation")
    print("=" * 60)
    print(f"  Backend : {jax.default_backend()}")
    for device in jax.devices():
        print(f"  Device  : {device}")
    print("-" * 60)
    print(f"  Altitude : {altitude_m:.0f} m  ({altitude_m / 0.3048:.0f} ft)")
    print(f"  Airspeed : {vt_ms:.1f} m/s  ({vt_ms / 0.3048:.1f} ft/s)")
    print(f"  Alpha    : {jnp.degrees(alpha_rad):.1f} deg")
    print(f"  dt = {dt} s  |  steps = {n_steps}  |  T = {dt * n_steps:.1f} s")
    print(f"  Vehicles : {n_sims}")
    print("-" * 60)

    if n_sims == 1:
        initial_state = FighterPlaneState(
            north=0.0, east=0.0, altitude=altitude_m,
            roll=0.0, pitch=pitch_rad, yaw=0.0,
            vt=vt_ms, alpha=alpha_rad, beta=0.0,
            P=0.0, Q=0.0, R=0.0,
            q0=q0, q1=q1, q2=q2, q3=q3,
            T=initial_T, el=-0.9, ail=0.0, rud=0.0,
            ax=0.0, ay=0.0, az=1.0,
        )

        action = FighterPlaneControlState(
            throttle=0.0375, elevator=-0.02, aileron=0.0, rudder=0.0,
            leading_edge_flap=0.0,
        )

        trajectory = simulate(initial_state, action, dt, n_steps)

        # Print every 100th step
        print(f"{'Step':>6}  {'Time':>6}  {'North':>10}  {'East':>10}  {'Alt':>10}  "
              f"{'Vt':>8}  {'Alpha':>7}  {'Pitch':>7}")
        print("-" * 80)
        for i in range(0, n_steps + 1, 100):
            t = t0 + i * dt
            print(f"{i:6d}  {t:6.2f}  "
                  f"{float(trajectory.north[i]):10.1f}  "
                  f"{float(trajectory.east[i]):10.1f}  "
                  f"{float(trajectory.altitude[i]):10.1f}  "
                  f"{float(trajectory.vt[i]):8.1f}  "
                  f"{float(jnp.degrees(trajectory.alpha[i])):7.2f}  "
                  f"{float(jnp.degrees(trajectory.pitch[i])):7.2f}")

        print("-" * 80)
        print(f"\n  Final position: N={float(trajectory.north[-1]):.1f} m, "
              f"E={float(trajectory.east[-1]):.1f} m, "
              f"Alt={float(trajectory.altitude[-1]):.1f} m")
        print(f"  Final airspeed: {float(trajectory.vt[-1]):.1f} m/s")
        print("=" * 60)

        trajectories = trajectory
    else:
        key = jax.random.PRNGKey(42)
        alt_key, vt_key, alpha_key = jax.random.split(key, 3)

        alts   = altitude_m + jax.random.normal(alt_key, (n_sims,)) * 100.0
        vts    = vt_ms + jax.random.normal(vt_key, (n_sims,)) * 10.0
        alphas = alpha_rad + jax.random.normal(alpha_key, (n_sims,)) * 0.01
        pitches = alphas

        q0s, q1s, q2s, q3s = euler_to_quaternion(
            jnp.zeros(n_sims), pitches, jnp.zeros(n_sims)
        )

        initial_states = FighterPlaneState(
            north=jnp.zeros(n_sims),
            east=jnp.zeros(n_sims),
            altitude=alts,
            roll=jnp.zeros(n_sims),
            pitch=pitches,
            yaw=jnp.zeros(n_sims),
            vel_x=jnp.zeros(n_sims),
            vel_y=jnp.zeros(n_sims),
            vel_z=jnp.zeros(n_sims),
            vt=vts,
            status=jnp.zeros(n_sims, dtype=jnp.int32),
            blood=jnp.full(n_sims, 100.0),
            alpha=alphas,
            beta=jnp.zeros(n_sims),
            P=jnp.zeros(n_sims),
            Q=jnp.zeros(n_sims),
            R=jnp.zeros(n_sims),
            q0=q0s, q1=q1s, q2=q2s, q3=q3s,
            T=jnp.full(n_sims, initial_T),
            el=jnp.full(n_sims, -0.9),
            ail=jnp.zeros(n_sims),
            rud=jnp.zeros(n_sims),
            ax=jnp.zeros(n_sims),
            ay=jnp.zeros(n_sims),
            az=jnp.ones(n_sims),
        )

        action = FighterPlaneControlState(
            throttle=0.0375, elevator=-0.02, aileron=0.0, rudder=0.0,
            leading_edge_flap=0.0,
        )

        trajectories = simulate_batch(initial_states, action, dt, n_steps)

        print(f"\n  Simulated {n_sims} vehicles")
        print("=" * 60)

    if args.output:
        export_acmi(trajectories, n_sims, dt, t0, args.output)

    return trajectories


if __name__ == "__main__":
    main()
