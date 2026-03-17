"""
Utility functions for space flight simulation.

Includes:
- ACMI file validation and reading
- Visualization helpers
- Analysis tools
- RL agent integration scaffold
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class ACMIValidator:
    """Validate and parse ACMI files."""
    
    @staticmethod
    def validate_file(filepath: str) -> bool:
        """
        Check if an ACMI file is valid.
        
        Args:
            filepath: Path to ACMI file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return False
                
                # Check header
                if not lines[0].startswith("FileType="):
                    return False
                
                return True
        except Exception:
            return False
    
    @staticmethod
    def read_acmi(filepath: str) -> Dict:
        """
        Parse an ACMI file and extract data.
        
        Args:
            filepath: Path to ACMI file
            
        Returns:
            Dictionary containing parsed data
        """
        data = {
            "header": {},
            "objects": {},
            "timesteps": [],
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_time = 0.0
        current_objects = {}
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Header lines
            if '=' in line and not line.startswith('#'):
                if not line.startswith('T='):  # Not a trajectory line
                    key, value = line.split('=', 1)
                    data["header"][key] = value
            
            # Timestamp marker
            elif line.startswith('#'):
                current_time = float(line[1:])
            
            # Object/trajectory line
            elif ',' in line or line.startswith('T='):
                # Parse object data
                pass
        
        return data


class SimulationAnalyzer:
    """Analysis tools for simulation data."""
    
    @staticmethod
    def compute_trajectory_metrics(
        position_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute various trajectory metrics.
        
        Args:
            position_history: [time_steps, num_envs, 3] position tensor
            
        Returns:
            Dictionary of metrics
        """
        # Displacement vectors between consecutive steps
        displacements = position_history[1:] - position_history[:-1]
        
        # Distance traveled per step
        distances = torch.norm(displacements, dim=2)
        
        # Cumulative distance
        total_distances = distances.sum(dim=0)
        
        # Speed (distance per timestep)
        speeds = distances
        
        metrics = {
            "total_distance": total_distances,
            "mean_speed": speeds.mean(dim=0),
            "max_speed": speeds.max(dim=0)[0],
            "min_speed": speeds.min(dim=0)[0],
            "final_position": position_history[-1],
            "initial_position": position_history[0],
            "displacement": position_history[-1] - position_history[0],
        }
        
        return metrics
    
    @staticmethod
    def compute_relative_distances(
        position_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distances between all objects at final timestep.
        
        Args:
            position_history: [time_steps, num_envs, 3] position tensor
            
        Returns:
            [num_envs, num_envs] distance matrix
        """
        final_positions = position_history[-1]  # [num_envs, 3]
        
        # Compute pairwise distances
        # Using broadcasting: expand dimensions and compute L2 norm
        diff = final_positions.unsqueeze(1) - final_positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        
        return distances


class RLAgentScaffold:
    """
    Scaffold for future RL agent integration.
    
    This provides the interface for connecting RL agents to the simulation.
    Future implementations will fill in the actual learning algorithm.
    """
    
    def __init__(self, num_envs: int, action_dim: int = 3):
        """
        Initialize RL agent scaffold.
        
        Args:
            num_envs: Number of parallel environments
            action_dim: Dimension of action space (e.g., 3 for x,y,z acceleration)
        """
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.step_count = 0
    
    def compute_actions(
        self,
        observations: Dict[str, torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        """
        Compute actions based on observations.
        
        This is a scaffold - actual implementation would use neural networks.
        
        Args:
            observations: Dict from environment.get_observation()
            training: Whether in training or inference mode
            
        Returns:
            [num_envs, action_dim] action tensor
        """
        # Placeholder: return zero actions (no control)
        device = observations["position"].device
        actions = torch.zeros(
            (self.num_envs, self.action_dim),
            device=device,
            dtype=observations["position"].dtype,
        )
        return actions
    
    def update(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: Dict[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update agent based on experience.
        
        This is a scaffold - actual implementation would use loss functions,
        backpropagation, etc.
        
        Args:
            observations: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_observations: Next observations
            dones: Episode termination flags
            
        Returns:
            Dictionary with training metrics
        """
        self.step_count += 1
        
        return {
            "total_reward": rewards.sum().item(),
            "mean_reward": rewards.mean().item(),
            "steps": self.step_count,
        }


class SimulationExporter:
    """Export simulation data in various formats."""
    
    @staticmethod
    def export_to_json(
        position_history: torch.Tensor,
        velocity_history: torch.Tensor,
        time_history: torch.Tensor,
        filepath: str,
    ) -> None:
        """
        Export simulation data to JSON format.
        
        Args:
            position_history: [time_steps, num_envs, 3]
            velocity_history: [time_steps, num_envs, 3]
            time_history: [time_steps, num_envs]
            filepath: Output file path
        """
        num_envs = position_history.shape[1]
        
        data = {
            "environments": []
        }
        
        for env_idx in range(num_envs):
            env_data = {
                "id": env_idx,
                "timesteps": []
            }
            
            for t in range(len(position_history)):
                timestep = {
                    "time": float(time_history[t, env_idx]),
                    "position": position_history[t, env_idx].tolist(),
                    "velocity": velocity_history[t, env_idx].tolist(),
                }
                env_data["timesteps"].append(timestep)
            
            data["environments"].append(env_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def export_to_csv(
        position_history: torch.Tensor,
        velocity_history: torch.Tensor,
        time_history: torch.Tensor,
        filepath: str,
    ) -> None:
        """
        Export simulation data to CSV format.
        
        Args:
            position_history: [time_steps, num_envs, 3]
            velocity_history: [time_steps, num_envs, 3]
            time_history: [time_steps, num_envs]
            filepath: Output file path
        """
        import csv
        
        num_envs = position_history.shape[1]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["time", "env_id"]
            header.extend([f"pos_{axis}" for axis in ["x", "y", "z"]])
            header.extend([f"vel_{axis}" for axis in ["x", "y", "z"]])
            writer.writerow(header)
            
            # Data
            for t in range(len(position_history)):
                for env_idx in range(num_envs):
                    row = [
                        float(time_history[t, env_idx]),
                        env_idx,
                    ]
                    row.extend(position_history[t, env_idx].tolist())
                    row.extend(velocity_history[t, env_idx].tolist())
                    writer.writerow(row)


def create_synthetic_scenarios() -> Dict[str, Dict]:
    """
    Create predefined simulation scenarios for testing.
    
    Returns:
        Dictionary mapping scenario names to configuration dicts
    """
    scenarios = {
        "single_object": {
            "num_envs": 1,
            "dt": 0.1,
            "max_steps": 100,
            "init_pos_range": (-100, 100),
            "init_vel_range": (50, 100),
        },
        "swarm": {
            "num_envs": 50,
            "dt": 0.1,
            "max_steps": 500,
            "init_pos_range": (-1000, 1000),
            "init_vel_range": (-100, 100),
        },
        "large_scale": {
            "num_envs": 10000,
            "dt": 0.01,
            "max_steps": 1000,
            "init_pos_range": (-5000, 5000),
            "init_vel_range": (-500, 500),
        },
        "fine_grained": {
            "num_envs": 100,
            "dt": 0.01,  # Finer timestep
            "max_steps": 2000,
            "init_pos_range": (-500, 500),
            "init_vel_range": (-200, 200),
        },
    }
    
    return scenarios


if __name__ == "__main__":
    print("Utility functions module loaded.")
    print("Available classes:")
    print("  - ACMIValidator: Validate and read ACMI files")
    print("  - SimulationAnalyzer: Analyze trajectory metrics")
    print("  - RLAgentScaffold: RL agent integration interface")
    print("  - SimulationExporter: Export data in various formats")
    print("\nAvailable functions:")
    print("  - create_synthetic_scenarios(): Get predefined test scenarios")
