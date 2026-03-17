"""
PyTorch Space Flight Simulation - Complete Examples
=====================================================

This file demonstrates various usage patterns for the space flight simulator,
from simple single-object simulations to complex batch processing scenarios.
"""

import torch
import torch.nn.functional as F
from space_flight_sim import SpaceFlightEnvironment, SimulationConfig
from simulation_utils import (
    SimulationAnalyzer,
    SimulationExporter,
    RLAgentScaffold,
    create_synthetic_scenarios,
)


def example_1_basic_single_simulation():
    """
    Basic example: Run a single object flying through space.
    
    This demonstrates:
    - Creating a simulation environment
    - Resetting and running steps
    - Accessing observations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Single Simulation")
    print("=" * 80)
    
    # Create configuration for single environment
    config = SimulationConfig(
        num_envs=1,
        dt=0.1,
        max_steps=50,
        init_pos_range=(-100, 100),
        init_vel_range=(50, 100),
    )
    
    # Initialize environment
    env = SpaceFlightEnvironment(config)
    
    # Reset to get initial observation
    obs = env.reset(seed=42)
    
    print(f"Initial Position: {obs['position'].squeeze().numpy()}")
    print(f"Initial Velocity: {obs['velocity'].squeeze().numpy()}")
    print()
    
    # Run simulation steps
    for step in range(50):
        obs = env.step()
        if step % 10 == 0:
            print(f"Step {step:3d}: Position = {obs['position'].squeeze().numpy()}")
    
    print(f"\nFinal Position: {obs['position'].squeeze().numpy()}")
    print(f"Final Velocity: {obs['velocity'].squeeze().numpy()}")
    print(f"Total Time: {obs['time'].item():.1f} seconds")


def example_2_batch_processing():
    """
    Demonstrate batch processing with multiple parallel environments.
    
    This is the key feature of PyTorch-based simulation:
    - All 100 objects compute in parallel
    - Single GPU can handle thousands of environments
    - Each environment is completely independent
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Processing (100 Parallel Environments)")
    print("=" * 80)
    
    config = SimulationConfig(
        num_envs=100,
        dt=0.1,
        max_steps=200,
        init_pos_range=(-500, 500),
        init_vel_range=(-200, 200),
        verbose=True,
    )
    
    env = SpaceFlightEnvironment(config)
    obs = env.run_episode(max_steps=200)
    
    # Compute statistics across all environments
    stats = env.get_statistics()
    
    print(f"\n{'Metric':<35} {'Value':<20}")
    print("-" * 55)
    print(f"{'Mean Final Position (x, y, z)':<35} {stats['mean_final_position'].numpy()}")
    print(f"{'Std Final Position (x, y, z)':<35} {stats['std_final_position'].numpy()}")
    print(f"{'Mean Distance Traveled':<35} {stats['mean_distance_traveled'].item():.2f} m")
    print(f"{'Max Distance Traveled':<35} {stats['max_distance_traveled'].item():.2f} m")
    print(f"{'Min Distance Traveled':<35} {stats['min_distance_traveled'].item():.2f} m")


def example_3_custom_initial_conditions():
    """
    Set up custom initial conditions for simulation.
    
    This shows how to:
    - Manually set initial positions and velocities
    - Create specific scenarios (e.g., objects moving in same direction)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Initial Conditions")
    print("=" * 80)
    
    config = SimulationConfig(
        num_envs=3,
        dt=0.1,
        max_steps=100,
    )
    
    env = SpaceFlightEnvironment(config)
    env.reset()
    
    # Manually set initial conditions
    # Scenario: Three objects starting at different positions with parallel velocities
    env.position = torch.tensor([
        [-500.0, -500.0, -500.0],   # Object 1: Bottom-left-back
        [0.0, 0.0, 0.0],             # Object 2: Center
        [500.0, 500.0, 500.0],       # Object 3: Top-right-front
    ], dtype=torch.float32)
    
    env.velocity = torch.tensor([
        [100.0, 0.0, 0.0],   # Object 1: Moving along x-axis
        [100.0, 0.0, 0.0],   # Object 2: Moving along x-axis
        [100.0, 0.0, 0.0],   # Object 3: Moving along x-axis
    ], dtype=torch.float32)
    
    print("Initial Setup:")
    print(f"Positions:\n{env.position.numpy()}")
    print(f"Velocities:\n{env.velocity.numpy()}\n")
    
    # Run simulation
    env._record_state()  # Record initial state
    for _ in range(100):
        env.step()
    
    # Analyze
    print("After 100 steps (10 seconds):")
    print(f"Positions:\n{env.position.numpy()}")
    
    # They should maintain the same relative distances
    distances = torch.norm(env.position.unsqueeze(1) - env.position.unsqueeze(0), dim=2)
    print(f"\nPairwise distances (should be constant):\n{distances.numpy()}")


def example_4_export_and_analysis():
    """
    Run simulation and export data in multiple formats.
    
    Demonstrates:
    - Tacview ACMI export
    - JSON export
    - CSV export
    - Trajectory analysis
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Data Export and Analysis")
    print("=" * 80)
    
    config = SimulationConfig(
        num_envs=10,
        dt=0.1,
        max_steps=300,
        init_pos_range=(-1000, 1000),
        init_vel_range=(-150, 150),
    )
    
    env = SpaceFlightEnvironment(config)
    env.run_episode(max_steps=300)
    
    # Export to Tacview ACMI format
    env.export_to_tacview(
        "/mnt/user-data/outputs/example_batch.acmi",
        env_indices=[0, 1, 2, 3, 4],
        object_name="TestObject"
    )
    print("✓ Exported to Tacview ACMI format")
    
    # Export to JSON
    SimulationExporter.export_to_json(
        torch.stack(env.position_history),
        torch.stack(env.velocity_history),
        torch.stack(env.time_history),
        "/mnt/user-data/outputs/example_data.json"
    )
    print("✓ Exported to JSON format")
    
    # Export to CSV
    SimulationExporter.export_to_csv(
        torch.stack(env.position_history),
        torch.stack(env.velocity_history),
        torch.stack(env.time_history),
        "/mnt/user-data/outputs/example_data.csv"
    )
    print("✓ Exported to CSV format")
    
    # Trajectory analysis
    position_data = torch.stack(env.position_history)
    metrics = SimulationAnalyzer.compute_trajectory_metrics(position_data)
    
    print("\nTrajectory Metrics:")
    print(f"  Total Distance: {metrics['total_distance'].numpy()}")
    print(f"  Mean Speed: {metrics['mean_speed'].numpy()}")
    print(f"  Max Speed: {metrics['max_speed'].numpy()}")
    
    # Relative distances at final timestep
    distances = SimulationAnalyzer.compute_relative_distances(position_data)
    print(f"\nFinal Pairwise Distances (first 5x5):\n{distances[:5, :5].numpy()}")


def example_5_gpu_acceleration():
    """
    Demonstrate GPU acceleration for large-scale simulations.
    
    Shows:
    - How to use GPU if available
    - Performance comparison
    - Moving tensors between devices
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: GPU Acceleration (if available)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("CUDA not available, using CPU instead")
    
    config = SimulationConfig(
        num_envs=1000,
        dt=0.01,
        max_steps=500,
        device=device,
    )
    
    env = SpaceFlightEnvironment(config)
    env.reset()
    
    print(f"\nRunning 1000 parallel simulations...")
    import time
    
    start = time.time()
    for _ in range(500):
        env.step()
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f} seconds")
    print(f"Speed: {500 * 1000 / elapsed:,.0f} environment steps/second")


def example_6_rl_integration_scaffold():
    """
    Demonstrate the RL integration scaffold (future use).
    
    This shows where and how an RL agent would integrate:
    - Agent observes environment state
    - Agent computes actions
    - Actions affect simulation (placeholder)
    - Rewards computed from observations
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: RL Integration Scaffold (Future)")
    print("=" * 80)
    
    num_envs = 10
    
    config = SimulationConfig(
        num_envs=num_envs,
        dt=0.1,
        max_steps=100,
    )
    
    # Initialize simulation and agent
    env = SpaceFlightEnvironment(config)
    agent = RLAgentScaffold(num_envs=num_envs, action_dim=3)
    
    obs = env.reset()
    
    print("Integration loop (placeholder for RL training):\n")
    
    for step in range(10):  # Just show first 10 steps
        # Agent observes current state
        print(f"Step {step}:")
        print(f"  Observation keys: {list(obs.keys())}")
        
        # Agent computes actions
        actions = agent.compute_actions(obs, training=True)
        print(f"  Actions shape: {actions.shape}")
        
        # Execute step (currently no action effect - for future implementation)
        next_obs = env.step()
        
        # Compute rewards (placeholder)
        # In actual RL: reward = function of next_obs (e.g., distance to target)
        rewards = torch.zeros(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        
        # Agent learns from experience
        metrics = agent.update(obs, actions, rewards, next_obs, dones)
        print(f"  Training metrics: {metrics}\n")
        
        obs = next_obs


def example_7_scenario_based_testing():
    """
    Run predefined scenarios for testing and validation.
    
    Demonstrates:
    - Using the scenario library
    - Testing different simulation scales
    - Reproducibility with seeds
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Predefined Scenarios")
    print("=" * 80)
    
    scenarios = create_synthetic_scenarios()
    
    print("Available scenarios:")
    for name, params in scenarios.items():
        print(f"\n  {name}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    # Run a subset of scenarios
    print("\n" + "-" * 80)
    print("Running scenarios...\n")
    
    for scenario_name in ["single_object", "swarm"]:
        params = scenarios[scenario_name]
        
        config = SimulationConfig(
            num_envs=params["num_envs"],
            dt=params["dt"],
            max_steps=params["max_steps"],
        )
        
        env = SpaceFlightEnvironment(config)
        obs = env.run_episode(max_steps=params["max_steps"])
        
        stats = env.get_statistics()
        
        print(f"{scenario_name}:")
        print(f"  Environments: {params['num_envs']}")
        print(f"  Final step: {obs['step'][0].item()}")
        print(f"  Mean distance: {stats['mean_distance_traveled'].item():.1f}m")
        print()


def example_8_distributed_simulation():
    """
    Example of how distributed/parallel simulation could work.
    
    This is a scaffold for future distributed computing:
    - Multiple PyTorch processes
    - Data gathering across processes
    - Load balancing
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Distributed Simulation Architecture")
    print("=" * 80)
    
    print("""
This example shows the architecture for distributed simulation:

1. Master Process:
   - Manages num_envs environments split across workers
   - Collects trajectories and computes statistics
   - Handles data export and logging

2. Worker Processes:
   - Each runs a subset of parallel environments
   - Uses local GPU if available
   - Sends observations/data back to master

3. Communication:
   - PyTorch Distributed Data Parallel (DDP) for gradient sync
   - Ray for distributed RL training
   - Custom communication for data gathering

Current Implementation:
  - Single process, all environments on one device
  - Ready to scale to multiple GPUs via DataParallel
  - Architecture supports distributed extension

Example scaling path:
  1. Single GPU: 10,000+ parallel environments ✓
  2. Multi-GPU: torch.nn.DataParallel (same machine)
  3. Distributed: torch.distributed (multiple machines)
  4. RL training: Ray RLlib (distributed agents + environments)
""")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "=" * 80)
    print("PyTorch Space Flight Simulation - Complete Examples")
    print("=" * 80)
    
    example_1_basic_single_simulation()
    example_2_batch_processing()
    example_3_custom_initial_conditions()
    example_4_export_and_analysis()
    example_5_gpu_acceleration()
    example_6_rl_integration_scaffold()
    example_7_scenario_based_testing()
    example_8_distributed_simulation()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()
