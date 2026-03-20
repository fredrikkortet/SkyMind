"""
Gradient-based PID gain tuning for the F-16 simulation via JAX autodiff.

Packs all 14 tunable parameters (11 PID gains + 3 trim constants) into a
single JAX array, differentiates a trajectory loss w.r.t. those parameters,
and optimises with Adam.  Uses vmap over perturbed initial conditions for
robustness.

Key techniques for gradient stability:
  - State clamping inside the sim loop prevents NaN from the plant model
  - custom_jvp on the loss zeros out gradients if NaN still leaks through
  - Curriculum learning: starts with short horizons (50 steps) and extends
    to 1000 steps so that early optimisation sees non-divergent trajectories

Usage:
    python tune_pid.py
"""

import jax
import jax.numpy as jnp

# Enable float64 for numerical precision through long backpropagation chains
jax.config.update("jax_enable_x64", True)

from fighterplane import FighterPlaneControlState, FighterPlaneState, update
from jax_sim import PIDState, euler_to_quaternion

# ---------------------------------------------------------------------------
# Parameter layout — indices into the gains vector (14 parameters total)
# ---------------------------------------------------------------------------
IDX_KP_ALT = 0
IDX_KI_ALT = 1
IDX_KD_ALT = 2
IDX_KP_PITCH = 3
IDX_KD_PITCH = 4
IDX_KP_SPD = 5
IDX_KI_SPD = 6
IDX_KP_ROLL = 7
IDX_KD_ROLL = 8
IDX_KP_YAW = 9
IDX_KD_YAW = 10
IDX_TRIM_THROTTLE = 11
IDX_TRIM_ELEVATOR = 12
IDX_TRIM_PITCH = 13

PARAM_NAMES = [
    "KP_ALT",
    "KI_ALT",
    "KD_ALT",
    "KP_PITCH",
    "KD_PITCH",
    "KP_SPD",
    "KI_SPD",
    "KP_ROLL",
    "KD_ROLL",
    "KP_YAW",
    "KD_YAW",
    "TRIM_THROTTLE",
    "TRIM_ELEVATOR",
    "TRIM_PITCH",
]

# Current hand-tuned values (starting point for optimisation)
INITIAL_GAINS = jnp.array(
    [
        0.001364,  # KP_ALT
        -0.005066,  # KI_ALT
        -0.246749,  # KD_ALT
        -0.214935,  # KP_PITCH
        -0.878661,  # KD_PITCH
        0.876461,  # KP_SPD
        0.130524,  # KI_SPD
        -2.370678,  # KP_ROLL
        -1.057750,  # KD_ROLL
        0.822293,  # KP_YAW
        -1.802605,  # KD_YAW
        0.164266,  # TRIM_THROTTLE
        0.038325,  # TRIM_ELEVATOR
        -0.015158,  # TRIM_PITCH
    ],
    dtype=jnp.float64,
)

# ---------------------------------------------------------------------------
# Simulation targets & parameters
# ---------------------------------------------------------------------------
TARGET_ALT = 3048.0  # m
TARGET_VT = 200.0  # m/s
DT = 0.01  # 100 Hz


# ---------------------------------------------------------------------------
# State clamping — keeps the plant model in a region where it won't NaN
# ---------------------------------------------------------------------------
def clamp_state(state: FighterPlaneState) -> FighterPlaneState:
    """Clamp state fields to physically plausible ranges to prevent NaN."""
    return FighterPlaneState(
        north=state.north,
        east=state.east,
        altitude=jnp.clip(state.altitude, 0.0, 25000.0),
        roll=state.roll,
        pitch=jnp.clip(state.pitch, -jnp.pi / 2, jnp.pi / 2),
        yaw=state.yaw,
        vel_x=state.vel_x,
        vel_y=state.vel_y,
        vel_z=state.vel_z,
        vt=jnp.clip(state.vt, 20.0, 500.0),
        status=state.status,
        blood=state.blood,
        alpha=jnp.clip(state.alpha, jnp.radians(-20.0), jnp.radians(45.0)),
        beta=jnp.clip(state.beta, jnp.radians(-30.0), jnp.radians(30.0)),
        P=jnp.clip(state.P, -15.0, 15.0),
        Q=jnp.clip(state.Q, -15.0, 15.0),
        R=jnp.clip(state.R, -15.0, 15.0),
        q0=state.q0,
        q1=state.q1,
        q2=state.q2,
        q3=state.q3,
        T=jnp.clip(state.T, 0.0, 30000.0),
        el=state.el,
        ail=state.ail,
        rud=state.rud,
        ax=state.ax,
        ay=state.ay,
        az=state.az,
    )


# ---------------------------------------------------------------------------
# NaN-safe loss wrapper via custom_jvp
# ---------------------------------------------------------------------------
@jax.custom_jvp
def nan_safe(x):
    """Return x if finite, else 1e6. Gradient is zeroed when not finite."""
    return jnp.where(jnp.isfinite(x), x, 1e6)


@nan_safe.defjvp
def nan_safe_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    out = nan_safe(x)
    is_fin = jnp.isfinite(x)
    return out, jnp.where(is_fin, dx, 0.0)


# ---------------------------------------------------------------------------
# PID controller — identical logic to jax_sim.pid_controller but takes an
# explicit `gains` array so JAX can differentiate through it.
# ---------------------------------------------------------------------------
def pid_controller_parameterised(
    state: FighterPlaneState,
    pid_state: PIDState,
    gains: jax.Array,
    target_alt: float,
    target_vt: float,
    dt: float,
) -> tuple[FighterPlaneControlState, PIDState]:
    """PID controller with gains passed as an explicit differentiable array."""
    kp_alt = gains[IDX_KP_ALT]
    ki_alt = gains[IDX_KI_ALT]
    kd_alt = gains[IDX_KD_ALT]
    kp_pitch = gains[IDX_KP_PITCH]
    kd_pitch = gains[IDX_KD_PITCH]
    kp_spd = gains[IDX_KP_SPD]
    ki_spd = gains[IDX_KI_SPD]
    kp_roll = gains[IDX_KP_ROLL]
    kd_roll = gains[IDX_KD_ROLL]
    kp_yaw = gains[IDX_KP_YAW]
    kd_yaw = gains[IDX_KD_YAW]
    trim_throttle = gains[IDX_TRIM_THROTTLE]
    trim_elevator = gains[IDX_TRIM_ELEVATOR]
    trim_pitch = gains[IDX_TRIM_PITCH]

    # --- Altitude hold (outer loop) -> pitch correction ---
    alt_err = target_alt - state.altitude
    climb_rate = state.vt * jnp.sin(state.pitch - state.alpha)
    alt_integral = jnp.clip(pid_state.alt_integral + alt_err * dt, -500.0, 500.0)
    pitch_correction = kp_alt * alt_err + ki_alt * alt_integral + kd_alt * (-climb_rate)
    pitch_correction = jnp.clip(pitch_correction, jnp.radians(-10.0), jnp.radians(10.0))
    pitch_cmd = trim_pitch + pitch_correction

    # --- Pitch hold (inner loop) -> elevator ---
    pitch_err = pitch_cmd - state.pitch
    elevator = trim_elevator + kp_pitch * pitch_err + kd_pitch * (-state.Q)
    elevator = jnp.clip(elevator, -1.0, 1.0)

    # --- Speed hold -> throttle ---
    spd_err = target_vt - state.vt
    spd_integral = jnp.clip(pid_state.spd_integral + spd_err * dt, -50.0, 50.0)
    throttle = trim_throttle + kp_spd * spd_err + ki_spd * spd_integral
    throttle = jnp.clip(throttle, 0.0, 1.0)

    # --- Roll hold (wings level) -> aileron ---
    aileron = kp_roll * (-state.roll) + kd_roll * (-state.P)
    aileron = jnp.clip(aileron, -1.0, 1.0)

    # --- Yaw damper -> rudder ---
    rudder = kp_yaw * (-state.beta) + kd_yaw * (-state.R)
    rudder = jnp.clip(rudder, -1.0, 1.0)

    action = FighterPlaneControlState(
        throttle=throttle,
        elevator=elevator,
        aileron=aileron,
        rudder=rudder,
        leading_edge_flap=0.0,
    )
    new_pid_state = PIDState(alt_integral=alt_integral, spd_integral=spd_integral)
    return action, new_pid_state


# ---------------------------------------------------------------------------
# Differentiable simulation loop (with state clamping)
# ---------------------------------------------------------------------------
def simulate_with_gains(
    gains: jax.Array,
    initial_state: FighterPlaneState,
    target_alt: float,
    target_vt: float,
    dt: float,
    n_steps: int,
) -> FighterPlaneState:
    """Run the simulation for n_steps with state clamping, returning the trajectory."""

    def step(carry, _):
        state, pid_state = carry
        action, new_pid_state = pid_controller_parameterised(
            state, pid_state, gains, target_alt, target_vt, dt
        )
        next_state = update(state, action, dt)
        next_state = clamp_state(next_state)
        return (next_state, new_pid_state), next_state

    initial_pid_state = PIDState()
    _, trajectory = jax.lax.scan(
        step, (initial_state, initial_pid_state), None, length=n_steps
    )

    # Prepend initial state
    trajectory = jax.tree.map(
        lambda init, traj: jnp.concatenate([jnp.atleast_1d(init), traj]),
        initial_state,
        trajectory,
    )
    return trajectory


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def trajectory_loss(
    gains: jax.Array,
    initial_state: FighterPlaneState,
    target_alt: float,
    target_vt: float,
    dt: float,
    n_steps: int,
) -> jax.Array:
    """
    Scalar loss over a single trajectory.

    Components (all weighted and normalised):
      - Altitude tracking error
      - Speed tracking error
      - Wings-level penalty (roll^2)
      - Angular-rate damping (P^2 + Q^2 + R^2)
      - Time-weighted: later timesteps weighted more (steady-state matters)
    """
    traj = simulate_with_gains(gains, initial_state, target_alt, target_vt, dt, n_steps)

    n = n_steps + 1  # trajectory length (includes initial state)

    # Time weighting: linearly increasing so steady-state counts more
    t_weight = jnp.linspace(0.5, 1.5, n)

    # Altitude tracking (normalised by target altitude)
    alt_err = (traj.altitude - target_alt) / target_alt
    loss_alt = jnp.mean(t_weight * alt_err**2)

    # Speed tracking (normalised by target speed)
    spd_err = (traj.vt - target_vt) / target_vt
    loss_spd = jnp.mean(t_weight * spd_err**2)

    # Wings-level penalty
    loss_roll = jnp.mean(t_weight * traj.roll**2)

    # Angular rate damping
    loss_rates = jnp.mean(t_weight * (traj.P**2 + traj.Q**2 + traj.R**2))

    # Weighted sum
    loss = 10.0 * loss_alt + 10.0 * loss_spd + 1.0 * loss_roll + 0.1 * loss_rates

    # NaN/Inf protection via custom_jvp (zeros gradient when loss is non-finite)
    loss = nan_safe(loss)

    return loss


# ---------------------------------------------------------------------------
# Batched loss (parameterised by n_steps for curriculum)
# ---------------------------------------------------------------------------
def make_batched_loss(n_steps: int):
    """Create a batched loss function for a given horizon length."""

    def loss_fn(gains: jax.Array, initial_states: FighterPlaneState) -> jax.Array:
        per_ic_loss = jax.vmap(
            lambda ic: trajectory_loss(gains, ic, TARGET_ALT, TARGET_VT, DT, n_steps)
        )(initial_states)
        return jnp.mean(per_ic_loss)

    return loss_fn


# ---------------------------------------------------------------------------
# Perturbed initial conditions
# ---------------------------------------------------------------------------
def make_initial_conditions(n_batch: int = 5) -> FighterPlaneState:
    """Create a batch of slightly perturbed initial conditions."""
    key = jax.random.PRNGKey(0)
    k_alt, k_vt, k_roll = jax.random.split(key, 3)

    altitude_m = TARGET_ALT
    vt_ms = TARGET_VT
    alpha_rad = 0.039
    pitch_rad = 0.039
    initial_T = 2109.0

    alt_perturb = jax.random.normal(k_alt, (n_batch,)) * 50.0  # +/- ~50 m
    vt_perturb = jax.random.normal(k_vt, (n_batch,)) * 5.0  # +/- ~5 m/s
    roll_perturb = jax.random.normal(k_roll, (n_batch,)) * 0.05  # +/- ~3 deg

    alts = altitude_m + alt_perturb
    vts = vt_ms + vt_perturb
    rolls = roll_perturb

    q0s, q1s, q2s, q3s = euler_to_quaternion(
        rolls, jnp.full(n_batch, pitch_rad), jnp.zeros(n_batch)
    )

    return FighterPlaneState(
        north=jnp.zeros(n_batch),
        east=jnp.zeros(n_batch),
        altitude=alts,
        roll=rolls,
        pitch=jnp.full(n_batch, pitch_rad),
        yaw=jnp.zeros(n_batch),
        vel_x=jnp.zeros(n_batch),
        vel_y=jnp.zeros(n_batch),
        vel_z=jnp.zeros(n_batch),
        vt=vts,
        status=jnp.zeros(n_batch, dtype=jnp.int32),
        blood=jnp.full(n_batch, 100.0),
        alpha=jnp.full(n_batch, alpha_rad),
        beta=jnp.zeros(n_batch),
        P=jnp.zeros(n_batch),
        Q=jnp.zeros(n_batch),
        R=jnp.zeros(n_batch),
        q0=q0s,
        q1=q1s,
        q2=q2s,
        q3=q3s,
        T=jnp.full(n_batch, initial_T),
        el=jnp.full(n_batch, -0.9),
        ail=jnp.zeros(n_batch),
        rud=jnp.zeros(n_batch),
        ax=jnp.zeros(n_batch),
        ay=jnp.zeros(n_batch),
        az=jnp.ones(n_batch),
    )


# ---------------------------------------------------------------------------
# Adam optimiser (manual — no optax dependency)
# ---------------------------------------------------------------------------
def adam_init(params: jax.Array) -> tuple:
    """Initialise Adam state: (m, v, step)."""
    return jnp.zeros_like(params), jnp.zeros_like(params), 0


def adam_step(
    params: jax.Array,
    grads: jax.Array,
    state: tuple,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jax.Array, tuple]:
    """One Adam update step. Returns (new_params, new_state)."""
    m, v, t = state
    t = t + 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, (m, v, t)


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------
# Start short (where even bad gains survive), then extend to full horizon
CURRICULUM = [
    # (n_steps, n_opt_iterations, learning_rate)
    (50, 80, 3e-3),
    (200, 80, 1e-3),
    (500, 100, 5e-4),
    (1000, 100, 3e-4),
]


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Gradient-Based PID Gain Tuning (JAX Autodiff)")
    print("=" * 60)
    print(f"  Backend : {jax.default_backend()}")
    for d in jax.devices():
        print(f"  Device  : {d}")
    print(f"  float64 : {jnp.array(1.0).dtype == jnp.float64}")
    print(f"  Params  : {len(PARAM_NAMES)}")
    print(f"  Curriculum : {' -> '.join(f'{s} steps' for s, _, _ in CURRICULUM)}")
    print("-" * 60)

    # Initial conditions
    n_batch = 5
    ics = make_initial_conditions(n_batch)
    print(f"  Batch   : {n_batch} perturbed initial conditions")

    gains = INITIAL_GAINS.copy()
    opt_state = adam_init(gains)

    total_step = 0

    for phase, (n_steps, n_iters, lr) in enumerate(CURRICULUM):
        print(f"\n{'=' * 60}")
        print(
            f"  Phase {phase + 1}/{len(CURRICULUM)}: {n_steps} steps "
            f"({n_steps * DT:.1f} s), lr={lr}, {n_iters} iters"
        )
        print("=" * 60)

        loss_fn = make_batched_loss(n_steps)
        loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

        print("  Compiling...")
        loss_val, grad_val = loss_and_grad(gains, ics)
        grad_norm = jnp.linalg.norm(grad_val)
        print(
            f"  Phase start — loss: {float(loss_val):.6f}, "
            f"grad norm: {float(grad_norm):.6e}"
        )

        # Reset Adam for each phase (new loss landscape)
        opt_state = adam_init(gains)

        # Track best within this phase
        phase_best_loss = loss_val if jnp.isfinite(loss_val) else jnp.array(1e8)
        phase_gains = gains.copy()

        # Adaptive grad clip: use initial grad norm as reference
        grad_clip = jnp.where(
            jnp.isfinite(grad_norm) & (grad_norm > 0.0),
            jnp.maximum(grad_norm * 0.5, 1.0),
            1.0,
        )
        print(f"  Grad clip  : {float(grad_clip):.4f}")

        print(f"\n  {'Step':>5}  {'Loss':>12}  {'Grad norm':>12}  {'Best':>12}")
        print("  " + "-" * 47)

        for i in range(n_iters):
            loss_val, grad_val = loss_and_grad(gains, ics)

            # Skip update if gradient is NaN (sim still diverging)
            grad_norm = jnp.linalg.norm(grad_val)
            is_valid = jnp.isfinite(grad_norm) & (grad_norm > 0.0)

            # Gradient clipping
            clipped_grad = jnp.where(
                grad_norm > grad_clip, grad_val * grad_clip / grad_norm, grad_val
            )
            # Zero out if NaN
            clipped_grad = jnp.where(is_valid, clipped_grad, 0.0)

            gains, opt_state = adam_step(gains, clipped_grad, opt_state, lr=lr)

            # Track best within this phase
            if jnp.isfinite(loss_val) and float(loss_val) < float(phase_best_loss):
                phase_best_loss = loss_val
                phase_gains = gains.copy()

            if i % 10 == 0 or i == n_iters - 1:
                print(
                    f"  {total_step:5d}  {float(loss_val):12.6f}  "
                    f"{float(grad_norm):12.6e}  {float(phase_best_loss):12.6f}"
                )

            total_step += 1

        # Carry the best gains from this phase into the next
        gains = phase_gains
        print(f"  Phase best : {float(phase_best_loss):.6f}")

    # ---------------------------------------------------------------------------
    # Final evaluation at full 1000-step horizon
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Final evaluation (1000 steps)")
    print("=" * 60)

    final_loss_fn = make_batched_loss(1000)
    final_jit = jax.jit(final_loss_fn)

    # If last phase was not 1000 steps, we need to compile once more
    print("  Evaluating optimised gains at 1000 steps...")
    opt_loss_1000 = final_jit(gains, ics)
    init_loss_1000 = final_jit(INITIAL_GAINS, ics)
    print(f"  Init  loss (1000 steps) : {float(init_loss_1000):.6f}")
    print(f"  Opt   loss (1000 steps) : {float(opt_loss_1000):.6f}")
    print(
        f"  Improvement             : {float(init_loss_1000 - opt_loss_1000):.6f} "
        f"({float((1 - opt_loss_1000 / init_loss_1000) * 100):.1f}%)"
    )
    print()

    # Print gains formatted for copy-paste into jax_sim.py
    print("  Optimised gains (copy into jax_sim.py):")
    print("  " + "-" * 50)

    prefix_map = {
        "KP_ALT": "_KP_ALT",
        "KI_ALT": "_KI_ALT",
        "KD_ALT": "_KD_ALT",
        "KP_PITCH": "_KP_PITCH",
        "KD_PITCH": "_KD_PITCH",
        "KP_SPD": "_KP_SPD",
        "KI_SPD": "_KI_SPD",
        "KP_ROLL": "_KP_ROLL",
        "KD_ROLL": "_KD_ROLL",
        "KP_YAW": "_KP_YAW",
        "KD_YAW": "_KD_YAW",
        "TRIM_THROTTLE": "_TRIM_THROTTLE",
        "TRIM_ELEVATOR": "_TRIM_ELEVATOR",
        "TRIM_PITCH": "_TRIM_PITCH",
    }

    for name, val in zip(PARAM_NAMES, gains):
        var_name = prefix_map[name]
        print(f"  {var_name} = {float(val):.6f}")

    print("  " + "-" * 50)
    print()

    # Show comparison
    print("  Comparison (initial -> optimised):")
    print(f"  {'Parameter':<18} {'Initial':>12} {'Optimised':>12} {'Delta':>12}")
    print("  " + "-" * 56)
    for name, init_val, opt_val in zip(PARAM_NAMES, INITIAL_GAINS, gains):
        delta = float(opt_val - init_val)
        print(
            f"  {name:<18} {float(init_val):12.6f} {float(opt_val):12.6f} {delta:+12.6f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
