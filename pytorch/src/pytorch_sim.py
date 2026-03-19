"""
Simple Flight Dynamics Simulation using PyTorch.

State vector: [x, y, z, vx, vy, vz]  (shape: (6,))
Dynamics: free flight (no forces), i.e. constant velocity.
"""

import argparse
from datetime import datetime, timezone

import torch


# -- 1. Dynamics --------------------------------------------------------------

def flight_dynamics(state: torch.Tensor, t: float) -> torch.Tensor:
    """
    Time derivative of state under free-flight (no forces).

    Args:
        state: (..., 6) tensor  [x, y, z, vx, vy, vz]
        t:     scalar time (unused here, kept for generality)

    Returns:
        dstate/dt: (..., 6) tensor  [vx, vy, vz, 0, 0, 0]
    """
    velocity = state[..., 3:]                    # extract vx, vy, vz
    acceleration = torch.zeros_like(velocity)   # no forces -> zero acceleration
    return torch.cat([velocity, acceleration], dim=-1)


# -- 2. RK4 Integrator --------------------------------------------------------

def rk4_step(f, state: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """
    Single RK4 integration step.

    Args:
        f:     dynamics function  f(state, t) -> dstate/dt
        state: current state (..., 6)
        t:     current time
        dt:    timestep

    Returns:
        next_state: (..., 6)
    """
    k1 = f(state,              t)
    k2 = f(state + dt/2 * k1,  t + dt/2)
    k3 = f(state + dt/2 * k2,  t + dt/2)
    k4 = f(state + dt   * k3,  t + dt)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# -- 3. Simulation Loop -------------------------------------------------------

def simulate(initial_state: torch.Tensor, t0: float, dt: float, n_steps: int):
    """
    Run the simulation for n_steps.

    Args:
        initial_state: (6,) tensor
        t0:            initial time
        dt:            fixed timestep
        n_steps:       number of integration steps

    Returns:
        trajectory: (n_steps + 1, 6) tensor  (includes the initial state)
    """
    device = initial_state.device
    dtype = initial_state.dtype
    
    # Pre-allocate trajectory tensor
    trajectory = torch.zeros((n_steps + 1, 6), device=device, dtype=dtype)
    trajectory[0] = initial_state
    
    state = initial_state.clone()
    t = t0
    
    for i in range(n_steps):
        state = rk4_step(flight_dynamics, state, t, dt)
        trajectory[i + 1] = state
        t += dt
    
    return trajectory


# Batched version: vectorize simulate over N initial states -> (N, n_steps+1, 6)
def simulate_batch(initial_states: torch.Tensor, t0: float, dt: float, n_steps: int):
    """
    Run simulations for a batch of initial states.

    Args:
        initial_states: (N, 6) tensor
        t0:             initial time
        dt:             fixed timestep
        n_steps:        number of integration steps

    Returns:
        trajectories: (N, n_steps + 1, 6) tensor
    """
    device = initial_states.device
    dtype = initial_states.dtype
    n_vehicles = initial_states.shape[0]
    
    # Pre-allocate trajectories tensor
    trajectories = torch.zeros((n_vehicles, n_steps + 1, 6), device=device, dtype=dtype)
    trajectories[:, 0] = initial_states
    
    states = initial_states.clone()
    t = t0
    
    for i in range(n_steps):
        # RK4 step for all vehicles in batch
        k1 = flight_dynamics(states, t)
        k2 = flight_dynamics(states + dt/2 * k1, t + dt/2)
        k3 = flight_dynamics(states + dt/2 * k2, t + dt/2)
        k4 = flight_dynamics(states + dt * k3, t + dt)
        
        states = states + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        trajectories[:, i + 1] = states
        t += dt
    
    return trajectories


# -- 4. ACMI Export -----------------------------------------------------------

def export_acmi(trajectories, dt, t0, filename):
    """
    Export one or more trajectories to an ACMI 2.1 text file for TacView.

    Args:
        trajectories: (N, n_steps+1, 6) tensor — N vehicles
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

    # Move to CPU for file I/O
    trajectories_cpu = trajectories.cpu().numpy()

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
                x = float(trajectories_cpu[v, i, 0])
                y = float(trajectories_cpu[v, i, 1])
                z = float(trajectories_cpu[v, i, 2])
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
    parser = argparse.ArgumentParser(description="PyTorch flight dynamics simulation")
    parser.add_argument("-o", "--output", help="Export trajectory to ACMI file for TacView")
    parser.add_argument("-n", "--num-sims", type=int, default=1,
                        help="Number of parallel simulations (default: 1)")
    parser.add_argument("--device", default="cpu",
                        help="Device to use (cpu or cuda, default: cpu)")
    args = parser.parse_args()

    n_sims = args.num_sims
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float32

    # Initial conditions
    position_0    = torch.tensor([0.0, 0.0, 1000.0], device=device, dtype=dtype)   # metres
    velocity_0    = torch.tensor([100.0, 0.0, 0.0], device=device, dtype=dtype)     # m/s  (due east)
    initial_state = torch.cat([position_0, velocity_0])

    # Simulation parameters
    dt      = 0.1    # seconds
    n_steps = 100    # -> 10 seconds of flight
    t0      = 0.0

    print("=" * 55)
    print("  PyTorch Flight Dynamics Simulation (free flight / RK4)")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
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
        error         = torch.norm(simulated_pos - expected_pos)

        print("-" * 75)
        print(f"\n  Analytic final position : {expected_pos}")
        print(f"  Simulated final position: {simulated_pos}")
        print(f"  L2 error                : {error:.2e}")
        print("\n  Trajectory shape:", trajectory.shape)
        print("=" * 55)

        # For ACMI export, wrap as (1, n_steps+1, 6)
        trajectories = trajectory.unsqueeze(0)
    else:
        # Generate randomized initial states: add ±50 m offsets to position
        torch.manual_seed(42)
        pos_offsets = torch.randn(n_sims, 3, device=device, dtype=dtype) * 50.0
        positions = position_0.unsqueeze(0) + pos_offsets               # (N, 3)
        velocities = velocity_0.unsqueeze(0).expand(n_sims, -1)        # (N, 3)
        initial_states = torch.cat([positions, velocities], dim=1)     # (N, 6)

        trajectories = simulate_batch(initial_states, t0, dt, n_steps)  # (N, n_steps+1, 6)

        print(f"\n  Simulated {n_sims} vehicles  |  trajectories shape: {trajectories.shape}")
        print("=" * 55)

    if args.output:
        export_acmi(trajectories, dt, t0, args.output)

    return trajectories


if __name__ == "__main__":
    trajectory = main()
