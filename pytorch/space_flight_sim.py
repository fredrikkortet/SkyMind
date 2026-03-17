"""
PyTorch-based Space Flight Simulation
A simple, scalable physics simulation for objects flying through space.
Designed for batch processing and future RL agent integration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for a space flight simulation."""
    
    # Batch and environment parameters
    num_envs: int = 1  # Number of parallel environments
    device: str = "cpu"  # Device: "cpu" or "cuda"
    dtype: torch.dtype = torch.float32  # Data precision
    
    # Simulation parameters
    dt: float = 0.1  # Time step (seconds)
    max_steps: int = 1000  # Maximum steps per episode
    
    # Initial state parameters
    init_pos_range: Tuple[float, float] = (-1000, 1000)  # Position initialization range (meters)
    init_vel_range: Tuple[float, float] = (-100, 100)  # Velocity initialization range (m/s)
    
    # Physics parameters
    scale_factor: float = 1.0  # Scale factor for acceleration (future: for RL agents)
    
    # Logging
    verbose: bool = False


class SpaceFlightEnvironment:
    """
    A batch-processing space flight simulator using PyTorch.
    
    Supports:
    - Multiple parallel environments (batch processing)
    - Simple kinematics (position = position + velocity * dt)
    - Independent simulation state for each environment
    - Export to Tacview ACMI format
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the space flight simulation.
        
        Args:
            config: SimulationConfig object with simulation parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # ============================================================================
        # State Tensors (Batch Dimension First)
        # ============================================================================
        # Shape: [num_envs, 3] for each tensor
        # This enables efficient parallel computation across all environments
        
        self.position = torch.zeros(
            (config.num_envs, 3),
            dtype=config.dtype,
            device=self.device
        )
        
        self.velocity = torch.zeros(
            (config.num_envs, 3),
            dtype=config.dtype,
            device=self.device
        )
        
        # Time tracking
        self.time = torch.zeros(config.num_envs, dtype=config.dtype, device=self.device)
        self.step_count = torch.zeros(config.num_envs, dtype=torch.long, device=self.device)
        
        # History for export and analysis
        self.position_history = []  # List of [num_envs, 3] tensors
        self.velocity_history = []  # List of [num_envs, 3] tensors
        self.time_history = []  # List of [num_envs] tensors
        
        if config.verbose:
            print(f"Initialized SpaceFlightEnvironment:")
            print(f"  Num Environments: {config.num_envs}")
            print(f"  Device: {self.device}")
            print(f"  Data Type: {config.dtype}")
            print(f"  Time Step (dt): {config.dt}s")
    
    def reset(self, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Reset all environments to initial state.
        
        Positions and velocities are randomly sampled within configured ranges.
        Each environment is independent and gets its own random initialization.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing initial observations
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Uniformly sample initial positions and velocities
        min_pos, max_pos = self.config.init_pos_range
        min_vel, max_vel = self.config.init_vel_range
        
        self.position = torch.empty(
            (self.config.num_envs, 3),
            dtype=self.config.dtype,
            device=self.device
        ).uniform_(min_pos, max_pos)
        
        self.velocity = torch.empty(
            (self.config.num_envs, 3),
            dtype=self.config.dtype,
            device=self.device
        ).uniform_(min_vel, max_vel)
        
        # Reset time tracking
        self.time = torch.zeros(self.config.num_envs, dtype=self.config.dtype, device=self.device)
        self.step_count = torch.zeros(self.config.num_envs, dtype=torch.long, device=self.device)
        
        # Clear history
        self.position_history = []
        self.velocity_history = []
        self.time_history = []
        
        # Record initial state
        self._record_state()
        
        return self.get_observation()
    
    def step(self) -> Dict[str, torch.Tensor]:
        """
        Perform one simulation step across all environments.
        
        Uses simple kinematic update: position_new = position_old + velocity * dt
        This is a constant-velocity motion with no forces/acceleration.
        
        The update is vectorized across all environments simultaneously using PyTorch,
        making it efficient even for large numbers of parallel simulations.
        
        Returns:
            Dictionary containing current observations
        """
        # ============================================================================
        # Vectorized Kinematics Update
        # ============================================================================
        # All environments updated in parallel using tensor operations
        # position: [num_envs, 3]
        # velocity: [num_envs, 3]
        # dt: scalar
        
        self.position = self.position + self.velocity * self.config.dt
        self.time = self.time + self.config.dt
        self.step_count = self.step_count + 1
        
        # Record state for export
        self._record_state()
        
        return self.get_observation()
    
    def _record_state(self) -> None:
        """
        Record current state to history for later export.
        
        This stores snapshots at each timestep, enabling full trajectory export
        to Tacview format.
        """
        self.position_history.append(self.position.clone().detach())
        self.velocity_history.append(self.velocity.clone().detach())
        self.time_history.append(self.time.clone().detach())
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """
        Get current observation from the environment.
        
        Returns:
            Dictionary with keys:
                - "position": [num_envs, 3] position tensor
                - "velocity": [num_envs, 3] velocity tensor
                - "time": [num_envs] time tensor
                - "step": [num_envs] step count tensor
        """
        return {
            "position": self.position,
            "velocity": self.velocity,
            "time": self.time,
            "step": self.step_count,
        }
    
    def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Run a complete episode (reset + multiple steps).
        
        Args:
            max_steps: Maximum number of steps (uses config.max_steps if not specified)
            
        Returns:
            Final observation dictionary
        """
        max_steps = max_steps or self.config.max_steps
        self.reset()
        
        for _ in range(max_steps):
            self.step()
        
        if self.config.verbose:
            print(f"Episode complete: {max_steps} steps")
        
        return self.get_observation()
    
    def export_to_tacview(
        self,
        filepath: str,
        env_indices: Optional[List[int]] = None,
        object_name: str = "SpaceObject"
    ) -> None:
        """
        Export simulation data to Tacview ACMI format.
        
        ACMI (Advanced Combat Management Interface) is a standardized flight recording format.
        This exports position and velocity data for each environment in a format that can be
        visualized in Tacview (a 3D flight analysis tool).
        
        Args:
            filepath: Path to save the ACMI file
            env_indices: List of environment indices to export (None = all)
            object_name: Name prefix for objects in Tacview
            
        Raises:
            ValueError: If no history data is available
        """
        if not self.position_history:
            raise ValueError("No history data available. Run simulation first.")
        
        # Determine which environments to export
        if env_indices is None:
            env_indices = list(range(self.config.num_envs))
        
        # Stack history tensors to get [time_steps, num_envs, 3]
        position_data = torch.stack(self.position_history, dim=0).cpu().numpy()
        velocity_data = torch.stack(self.velocity_history, dim=0).cpu().numpy()
        time_data = torch.stack(self.time_history, dim=0).cpu().numpy()
        
        # Write ACMI file header
        with open(filepath, 'w') as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.2\n")
            f.write("RecordingType=FullEventPlayback\n")
            f.write(f"Title=PyTorch Space Flight Simulation\n")
            f.write(f"Date={datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Time={datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"Timezone=UTC\n")
            f.write(f"Duration={time_data[-1, 0]:.1f}\n")
            f.write("0,=F/A-18C Hornet\n")  # Default object type
            f.write("\n")  # End of header
            
            # Write timestamp and object data for each timestep
            current_time = 0.0
            for step_idx, (positions, velocities, times) in enumerate(
                zip(position_data, velocity_data, time_data)
            ):
                # Get the current time from first environment (all envs share time step)
                current_time = float(times[0])
                
                # Write timestamp line (format: #<time_in_seconds>)
                f.write(f"#{current_time:.2f}\n")
                
                # Write data for each requested environment
                for env_idx in env_indices:
                    if env_idx < len(positions):
                        pos = positions[env_idx]
                        vel = velocities[env_idx]
                        
                        # ACMI object line format (simplified):
                        # ObjectID,Callsign,Type,Coalition,Country,Pilot,Reg,Squawk,Rebuild,
                        # ,Type,GroupNumber,TaskName,FlightNumber,Callsign,Pilot,Color,
                        # Position(Lon,Lat,Alt)=decimal degrees and feet,
                        # Rotation(Roll,Pitch,Yaw)=degrees,
                        # Velocity(VelX,VelY,VelZ)=feet/s
                        
                        # For simplicity, we'll use a basic format:
                        # ObjectID,Callsign,Coalition,CountryID
                        # Then T for Trigger line with position and velocity
                        
                        obj_id = f"{env_idx + 1:04d}"
                        callsign = f"{object_name}_{env_idx + 1}"
                        
                        # Position in km (convert from meters)
                        x_km = float(pos[0]) / 1000.0
                        y_km = float(pos[1]) / 1000.0
                        z_km = float(pos[2]) / 1000.0
                        
                        # Velocity in km/s (convert from m/s)
                        vx_kms = float(vel[0]) / 1000.0
                        vy_kms = float(vel[1]) / 1000.0
                        vz_kms = float(vel[2]) / 1000.0
                        
                        # ACMI line: ObjectID,Coalition,Country,Callsign,Type=,FlightNumber=,Pilot=
                        # T for coordinates: Lon=,Lat=,Alt=,Roll=,Pitch=,Yaw=,VelX=,VelY=,VelZ=
                        
                        # Use X,Y,Z directly for space coordinates (not lat/lon)
                        line = (
                            f"{obj_id},Coalition=0,Country=0,Callsign={callsign},Type=SpaceObject,\n"
                            f"T={x_km:.2f}|{y_km:.2f}|{z_km:.2f}|0|0|0|{vx_kms:.4f}|{vy_kms:.4f}|{vz_kms:.4f}\n"
                        )
                        f.write(line)
                
                f.write("\n")  # Blank line between timesteps
        
        if self.config.verbose:
            print(f"Exported {len(env_indices)} environments to {filepath}")
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """
        Compute statistics across all environments.
        
        Useful for understanding simulation behavior across parallel environments.
        
        Returns:
            Dictionary with statistics:
                - "mean_final_position": Mean position at last step
                - "std_final_position": Std dev of positions at last step
                - "mean_final_velocity": Mean velocity at last step
                - "total_distance_traveled": Distance traveled by each env
        """
        if not self.position_history:
            raise ValueError("No history data available. Run simulation first.")
        
        # Final positions and velocities
        final_pos = self.position_history[-1]
        final_vel = self.velocity_history[-1]
        init_pos = self.position_history[0]
        
        # Distance traveled (straight line motion)
        distance = torch.norm(final_pos - init_pos, dim=1)
        
        stats = {
            "mean_final_position": final_pos.mean(dim=0),
            "std_final_position": final_pos.std(dim=0),
            "mean_final_velocity": final_vel.mean(dim=0),
            "std_final_velocity": final_vel.std(dim=0),
            "mean_distance_traveled": distance.mean(),
            "std_distance_traveled": distance.std(),
            "min_distance_traveled": distance.min(),
            "max_distance_traveled": distance.max(),
        }
        
        return stats
    
    def to_device(self, device: str) -> None:
        """
        Move all tensors to a different device (CPU/GPU).
        
        Useful for switching between CPU and GPU computation.
        
        Args:
            device: Device string ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.position = self.position.to(self.device)
        self.velocity = self.velocity.to(self.device)
        self.time = self.time.to(self.device)
        self.step_count = self.step_count.to(self.device)
        
        # Move history
        self.position_history = [p.to(self.device) for p in self.position_history]
        self.velocity_history = [v.to(self.device) for v in self.velocity_history]
        self.time_history = [t.to(self.device) for t in self.time_history]
        
        if self.config.verbose:
            print(f"Moved all tensors to {device}")


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PyTorch Space Flight Simulation - Example Usage")
    print("=" * 80)
    
    # ========================================================================
    # Example 1: Single Environment Simulation
    # ========================================================================
    print("\n[Example 1] Single Environment Simulation")
    print("-" * 80)
    
    config_single = SimulationConfig(
        num_envs=1,
        dt=0.1,
        max_steps=100,
        init_pos_range=(-100, 100),
        init_vel_range=(50, 100),
        verbose=True,
    )
    
    env_single = SpaceFlightEnvironment(config_single)
    obs_single = env_single.run_episode(max_steps=100)
    
    print(f"Final position: {obs_single['position'].squeeze().numpy()}")
    print(f"Final velocity: {obs_single['velocity'].squeeze().numpy()}")
    print(f"Total time: {obs_single['time'].item():.2f}s")
    
    # Export single environment to Tacview
    env_single.export_to_tacview("single_env.acmi", env_indices=[0])
    print(f"Exported single environment to 'single_env.acmi'")
    
    # ========================================================================
    # Example 2: Batch Processing - Multiple Parallel Environments
    # ========================================================================
    print("\n[Example 2] Batch Processing - 100 Parallel Environments")
    print("-" * 80)
    
    config_batch = SimulationConfig(
        num_envs=100,
        dt=0.1,
        max_steps=500,
        init_pos_range=(-500, 500),
        init_vel_range=(-200, 200),
        verbose=True,
    )
    
    env_batch = SpaceFlightEnvironment(config_batch)
    obs_batch = env_batch.run_episode(max_steps=500)
    
    # Compute and display statistics
    stats = env_batch.get_statistics()
    print("\nSimulation Statistics (across 100 environments):")
    print(f"  Mean final position: {stats['mean_final_position'].numpy()}")
    print(f"  Std final position:  {stats['std_final_position'].numpy()}")
    print(f"  Mean distance traveled: {stats['mean_distance_traveled'].item():.2f} meters")
    print(f"  Max distance traveled:  {stats['max_distance_traveled'].item():.2f} meters")
    
    # Export subset of environments
    env_batch.export_to_tacview("batch_env_subset.acmi", env_indices=[0, 1, 2, 3, 4])
    print(f"Exported 5 environments to 'batch_env_subset.acmi'")
    
    # ========================================================================
    # Example 3: Demonstrating Vectorization Performance
    # ========================================================================
    print("\n[Example 3] Vectorization and GPU Support (if available)")
    print("-" * 80)
    
    import time
    
    # Test with different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    
    print("Performance across different batch sizes (CPU):")
    for batch_size in batch_sizes:
        config_perf = SimulationConfig(
            num_envs=batch_size,
            dt=0.01,
            max_steps=1000,
            device="cpu",
        )
        
        env_perf = SpaceFlightEnvironment(config_perf)
        env_perf.reset()
        
        start = time.time()
        for _ in range(1000):
            env_perf.step()
        elapsed = time.time() - start
        
        print(f"  {batch_size:4d} envs, 1000 steps: {elapsed:.3f}s")
    
    # ========================================================================
    # Example 4: RL-Ready Architecture (no RL yet)
    # ========================================================================
    print("\n[Example 4] RL-Ready Architecture")
    print("-" * 80)
    print("The simulation supports future RL agent integration:")
    print("  - Vectorized batch processing ✓")
    print("  - Independent environment state ✓")
    print("  - GPU acceleration ready ✓")
    print("  - Structured observation format ✓")
    print("  - Modular design for reward/action integration ✓")
    print("\nTo add an RL agent:")
    print("  1. Create agent with action space (e.g., acceleration commands)")
    print("  2. Add 'apply_actions()' method to SpaceFlightEnvironment")
    print("  3. Define reward function based on observations")
    print("  4. Use config.scale_factor for control magnitude")
    
    print("\n" + "=" * 80)
    print("Simulation complete! Check the .acmi files for Tacview visualization.")
    print("=" * 80)
