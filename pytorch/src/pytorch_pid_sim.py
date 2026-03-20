"""
PID-Controlled F16 Simulation.

Replaces the simplified rigid-body dynamics in pid_sim.py with the real
F16Model (envs/models/F16_model.py), which uses F16Dynamics + torchdiffeq
for a full nonlinear 6-DOF integration step.

The PID Controller stack (RollController, PitchController, YawController,
TECS, L1Controller) drives the F16Model's control surfaces each timestep.

State vector (F16Model internal, shape [N, 12]):
    0  x        North position     ft
    1  y        East position      ft
    2  z        Altitude (up)      ft
    3  roll     Roll angle         rad
    4  pitch    Pitch angle        rad
    5  yaw      Yaw angle          rad
    6  vt       True airspeed      ft/s
    7  alpha    Angle of attack    rad
    8  beta     Sideslip angle     rad
    9  p        Roll rate          rad/s
    10 q        Pitch rate         rad/s
    11 r        Yaw rate           rad/s

Control vector (F16Model internal, shape [N, 5]):
    0  T    Thrust               lbf
    1  el   Elevator deflection  deg
    2  ail  Aileron deflection   deg
    3  rud  Rudder deflection    deg
    4  lef  Leading-edge flap    deg  (always 0 here)

The Controller.get_action() output [throttle, elevator, aileron, rudder]
is normalised to [-1, 1] and passed into F16Model.update(), which applies
its own actuator lag and scaling exactly as designed.

Usage:
    python pytorch_pid_sim.py                     # single F16, loiter demo
    python pytorch_pid_sim.py -n 4               # batch of 4
    python pytorch_pid_sim.py -o flight.acmi     # TacView export
    python pytorch_pid_sim.py --mode waypoint    # fly to a waypoint
    python pytorch_pid_sim.py --mode heading     # heading hold
    python pytorch_pid_sim.py --mode level       # wings-level
    python pytorch_pid_sim.py --steps 1000       # longer run
"""

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import torch

# ---------------------------------------------------------------------------
# Path setup — adjust to your repo layout
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "envs"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "envs", "models"))

from envs.models.F16_model import F16Model
from algorithms.pid.controller import Controller


# ===========================================================================
# 1. Minimal env shim
# ===========================================================================

class AircraftEnv:
    """
    Minimal environment that wires F16Model to the PID Controller.

    The Controller calls env.model.get_*() for state observations.
    F16Model.update(action) advances physics one dt step using torchdiffeq.
    """
    def __init__(self, f16: F16Model):
        self.model = f16


# ===========================================================================
# 2. F16Model factory
# ===========================================================================

def make_f16_config(
    dt: float = 0.02,
    altitude_ft: float = 19_500.0,
    airspeed_fts: float = 1_100.0,
    init_thrust: float = 10_000.0,
    solver: str = "euler",
) -> SimpleNamespace:
    """Build a minimal config namespace for F16Model.__init__()."""
    return SimpleNamespace(
        num_states   = 12,
        num_controls = 5,
        dt           = dt,
        solver       = solver,
        airspeed     = 0,
        max_altitude = altitude_ft + 500,
        min_altitude = altitude_ft - 500,
        max_vt       = airspeed_fts + 100,
        min_vt       = airspeed_fts - 100,
        init_state   = {"init_T": init_thrust},
    )


def make_f16(
    n: int,
    device: torch.device,
    altitude_ft: float = 19_500.0,
    airspeed_fts: float = 1_100.0,
    dt: float = 0.02,
) -> F16Model:
    """
    Instantiate and initialise an F16Model batch of size n.

    Sets altitude (z) and TAS (vt) from the given initial conditions.
    All other states start at zero (trimmed straight-and-level).
    """
    config = make_f16_config(dt=dt, altitude_ft=altitude_ft,
                             airspeed_fts=airspeed_fts)
    f16 = F16Model(config, n=n, device=str(device), random_seed=42)

    # Set initial conditions directly (bypass random reset)
    f16.s[:, 2] = altitude_ft
    f16.s[:, 6] = airspeed_fts
    f16.u[:, 0] = config.init_state["init_T"]

    if n > 1:
        torch.manual_seed(42)
        f16.s[:, :2] += torch.randn(n, 2, device=device) * 50.0

    f16.recent_s = f16.s.clone()
    f16.recent_u = f16.u.clone()
    return f16


# ===========================================================================
# 3. Main simulation loop
# ===========================================================================

def simulate_pid(
    n: int,
    altitude_ft: float,
    airspeed_fts: float,
    hgt_dem: torch.Tensor,
    TAS_dem: torch.Tensor,
    dt: float,
    n_steps: int,
    nav_mode: str,
    nav_kwargs: dict,
    device: torch.device,
) -> tuple:
    """
    Run the PID-controlled F16 simulation.

    Returns:
        trajectories: (N, n_steps+1, 12) — full F16 state history
        actions:      (N, n_steps,   4)  — [throttle, el, ail, rud] in [-1,1]
    """
    f16 = make_f16(n, device, altitude_ft=altitude_ft,
                   airspeed_fts=airspeed_fts, dt=dt)
    env = AircraftEnv(f16)

    controller = Controller(
        airspeed_min = 800,
        airspeed_max = 2000,
        dt           = dt,
        n            = n,
        device       = str(device),
    )

    trajectories = torch.zeros((n, n_steps + 1, 12), device=device)
    all_actions  = torch.zeros((n, n_steps,      4), device=device)
    trajectories[:, 0] = f16.s.clone()

    for i in range(n_steps):

        # Outer loop: navigation → roll_dem, yaw_rate_dem
        if nav_mode == "loiter":
            controller.update_loiter(**nav_kwargs, env=env)
        elif nav_mode == "waypoint":
            nav_kwargs["state"]  = f16.s   # update with live state
            nav_kwargs["estate"] = f16.s
            controller.update_waypoint(**nav_kwargs, env=env)
        elif nav_mode == "heading":
            controller.update_heading_hold(**nav_kwargs, env=env)
        elif nav_mode == "level":
            controller.update_level_flight(env=env)
        else:
            raise ValueError(f"Unknown nav_mode: {nav_mode!r}")

        # TECS: altitude + airspeed → pitch_dem, throttle_dem
        controller.cal_pitch_throttle(hgt_dem, TAS_dem, env)

        # Inner loop: attitude stabilisation → aileron, elevator, rudder
        controller.stabilize(env)

        # Get normalised control output [-1, 1]
        action = controller.get_action()        # (N, 4)
        all_actions[:, i] = action.detach()

        # Advance physics: F16Model applies actuator lag + full odeint step
        f16.update(action)

        trajectories[:, i + 1] = f16.s.clone()

    return trajectories, all_actions


# ===========================================================================
# 4. ACMI export
# ===========================================================================

def export_acmi(trajectories: torch.Tensor, dt: float, t0: float, filename: str):
    """Export to ACMI 2.1 for TacView. F16: x=north, y=east, z=up (ft)."""
    ref_time  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    FT_TO_M   = 0.3048
    M_PER_DEG = 111_320.0

    traj = trajectories.cpu().numpy()
    N, n_frames, _ = traj.shape

    with open(filename, "w") as f:
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.1\n")
        f.write(f"0,ReferenceTime={ref_time}\n")
        f.write("0,ReferenceLongitude=0\n")
        f.write("0,ReferenceLatitude=0\n")

        for i in range(n_frames):
            f.write(f"#{float(t0 + i * dt):.2f}\n")
            for v in range(N):
                obj_id    = v + 1
                lat       = (float(traj[v, i, 0]) * FT_TO_M) / M_PER_DEG
                lon       = (float(traj[v, i, 1]) * FT_TO_M) / M_PER_DEG
                alt       =  float(traj[v, i, 2]) * FT_TO_M
                yaw_deg   = math.degrees(float(traj[v, i, 5])) % 360
                pitch_deg = math.degrees(float(traj[v, i, 4]))
                roll_deg  = math.degrees(float(traj[v, i, 3]))
                pose      = f"T={lon:.6f}|{lat:.6f}|{alt:.1f}|{yaw_deg:.1f}|{pitch_deg:.1f}|{roll_deg:.1f}"
                if i == 0:
                    f.write(f"{obj_id},{pose},Name=F16_{obj_id},Type=Air+FixedWing\n")
                else:
                    f.write(f"{obj_id},{pose}\n")

    print(f"  ACMI exported → {filename}  ({N} F16s, {n_frames} frames)")


# ===========================================================================
# 5. CLI
# ===========================================================================

def print_table(trajectories, actions, dt, t0, step_every=10):
    traj = trajectories[0].cpu()
    acts = actions[0].cpu()
    hdr = (f"{'Step':>6}  {'t(s)':>6}  {'x(ft)':>8}  {'y(ft)':>8}  "
           f"{'z(ft)':>7}  {'roll°':>6}  {'pitch°':>7}  {'yaw°':>6}  "
           f"{'vt':>6}  {'thr':>5}  {'el':>5}  {'ail':>5}  {'rud':>5}")
    print(hdr)
    print("-" * len(hdr))
    for i in range(0, traj.shape[0], step_every):
        s = traj[i]
        a = acts[min(i, acts.shape[0] - 1)]
        print(
            f"{i:6d}  {t0 + i*dt:6.1f}  "
            f"{s[0]:8.0f}  {s[1]:8.0f}  {s[2]:7.0f}  "
            f"{math.degrees(s[3].item()):6.1f}  "
            f"{math.degrees(s[4].item()):7.1f}  "
            f"{math.degrees(s[5].item()):6.1f}  "
            f"{s[6]:6.1f}  "
            f"{a[0]:5.2f}  {a[1]:5.2f}  {a[2]:5.2f}  {a[3]:5.2f}"
        )


def main():
    p = argparse.ArgumentParser(description="PID-controlled F16 simulation")
    p.add_argument("-o", "--output",   help="Export ACMI for TacView")
    p.add_argument("-n", "--num-sims", type=int,   default=1)
    p.add_argument("--device",                     default="cpu")
    p.add_argument("--mode",           default="loiter",
                   choices=["loiter", "waypoint", "heading", "level"])
    p.add_argument("--dt",             type=float, default=0.02)
    p.add_argument("--steps",          type=int,   default=500)
    args = p.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    N, dt, t0 = args.num_sims, args.dt, 0.0

    ALT_FT, TAS_FTS = 19_500.0, 1_100.0   # nominal F16 cruise conditions

    print("=" * 70)
    print("  PID-Controlled F16 Simulation  ·  F16Model + torchdiffeq")
    print("=" * 70)
    print(f"  Device : {device}  |  Aircraft: {N}  |  Mode: {args.mode}")
    print(f"  dt     : {dt} s    |  Steps: {args.steps}  |  T: {dt*args.steps:.1f} s")
    print(f"  Init   : alt={ALT_FT:.0f} ft  TAS={TAS_FTS:.0f} ft/s")
    print("-" * 70)

    hgt_dem = torch.full((N, 1), ALT_FT,  device=device)
    TAS_dem = torch.full((N, 1), TAS_FTS, device=device)

    if args.mode == "loiter":
        center = torch.zeros((N, 2), device=device)
        center[:, 0] = 3_000.0
        nav_kwargs = dict(
            center_WP        = center,
            radius           = torch.full((N, 1), 2_000.0, device=device),
            loiter_direction = torch.ones((N, 1),           device=device),
        )
    elif args.mode == "waypoint":
        prev_WP = torch.zeros((N, 2), device=device)
        next_WP = torch.zeros((N, 2), device=device)
        next_WP[:, 0] = 20_000.0
        nav_kwargs = dict(
            prev_WP  = prev_WP,
            next_WP  = next_WP,
            dist_min = torch.full((N, 1), 200.0, device=device),
            state    = torch.zeros((N, 12), device=device),
            estate   = torch.zeros((N, 12), device=device),
            eas2tas  = torch.ones((N, 1),   device=device),
        )
    elif args.mode == "heading":
        nav_kwargs = dict(
            navigation_heading = torch.full((N, 1), math.pi / 4, device=device),
        )
    else:
        nav_kwargs = {}

    print("  Simulating …")
    trajectories, actions = simulate_pid(
        n            = N,
        altitude_ft  = ALT_FT,
        airspeed_fts = TAS_FTS,
        hgt_dem      = hgt_dem,
        TAS_dem      = TAS_dem,
        dt           = dt,
        n_steps      = args.steps,
        nav_mode     = args.mode,
        nav_kwargs   = nav_kwargs,
        device       = device,
    )

    print(f"\n  Trajectory : {tuple(trajectories.shape)}")
    print(f"  Actions    : {tuple(actions.shape)}\n")
    print_table(trajectories, actions, dt, t0, max(1, args.steps // 10))

    s = trajectories[0, -1]
    print(f"\n  Final state (F16 0):")
    print(f"    Position : ({s[0]:.0f}, {s[1]:.0f}, {s[2]:.0f}) ft")
    print(f"    Attitude : roll={math.degrees(s[3].item()):.1f}°  "
          f"pitch={math.degrees(s[4].item()):.1f}°  "
          f"yaw={math.degrees(s[5].item()):.1f}°")
    print(f"    TAS      : {s[6]:.1f} ft/s  |  alpha={math.degrees(s[7].item()):.2f}°  "
          f"beta={math.degrees(s[8].item()):.2f}°")
    print("=" * 70)

    if args.output:
        export_acmi(trajectories, dt, t0, args.output)

    return trajectories, actions


if __name__ == "__main__":
    main()