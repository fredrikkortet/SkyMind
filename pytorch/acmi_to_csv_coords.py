#!/usr/bin/env python3
"""
Convert ACMI files to CSV with latitude, longitude, altitude format.
"""

import csv
from pathlib import Path
import math


class CoordinateConverter:
    """Convert Cartesian coordinates to latitude/longitude/altitude."""
    
    EARTH_RADIUS_M = 6371000  # Earth radius in meters
    
    def __init__(self, ref_latitude=32.72722222, ref_longitude=-116.6847222, ref_altitude=34):
        """
        Initialize converter with reference point.
        
        Args:
            ref_latitude: Reference latitude in degrees
            ref_longitude: Reference longitude in degrees
            ref_altitude: Reference altitude in meters (or feet if alt_unit='ft')
        """
        self.ref_lat = ref_latitude
        self.ref_lon = ref_longitude
        self.ref_alt = ref_altitude
        
        # Precompute reference point in radians
        self.ref_lat_rad = math.radians(ref_latitude)
        self.ref_lon_rad = math.radians(ref_longitude)
    
    def ecef_to_latlon(self, x_m, y_m, z_m):
        """
        Convert local ENU (East-North-Up) coordinates relative to reference point
        to latitude, longitude, altitude.
        
        Args:
            x_m: X offset in meters (East)
            y_m: Y offset in meters (North)
            z_m: Z offset in meters (Up = altitude)
        
        Returns:
            tuple: (latitude, longitude, altitude)
        """
        # Using simplified flat-earth approximation for small distances
        # Calculate meters per degree at reference latitude
        lat_meters_per_degree = 111320  # Approximately constant
        lon_meters_per_degree = 111320 * math.cos(self.ref_lat_rad)
        
        # Convert offsets to degrees
        delta_lat = y_m / lat_meters_per_degree
        delta_lon = x_m / lon_meters_per_degree
        
        # Calculate new coordinates
        latitude = self.ref_lat + delta_lat
        longitude = self.ref_lon + delta_lon
        altitude = self.ref_alt + z_m
        
        return latitude, longitude, altitude


def parse_acmi_file(acmi_path):
    """Parse ACMI file and extract telemetry data."""
    data = []
    current_time = None
    current_objects = {}
    
    with open(acmi_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        # Parse object properties
        if ',' in line and not line.startswith('#') and not line.startswith('T='):
            parts = line.split(',')
            if parts[0].isdigit():
                obj_id = parts[0]
                obj_props = {}
                for prop in parts[1:]:
                    if '=' in prop:
                        k, v = prop.split('=', 1)
                        obj_props[k] = v
                current_objects[obj_id] = obj_props
        
        # Parse timestamp markers
        elif line.startswith('#'):
            try:
                current_time = float(line[1:])
            except ValueError:
                continue
        
        # Parse telemetry lines
        elif line.startswith('T='):
            if current_time is not None and current_objects:
                telem_str = line[2:]
                values = telem_str.split('|')
                
                if len(values) >= 9:
                    # Get last object
                    for obj_id, obj_props in current_objects.items():
                        record = {
                            'time': current_time,
                            'obj_id': obj_id,
                            'callsign': obj_props.get('Callsign', ''),
                            'x': float(values[0]),
                            'y': float(values[1]),
                            'z': float(values[2]),
                            'roll': float(values[3]),
                            'pitch': float(values[4]),
                            'yaw': float(values[5]),
                        }
                        data.append(record)
    
    return data


def convert_to_csv_with_coords(acmi_path, output_path, converter, scale_x1000=True):
    """
    Convert ACMI file to CSV with latitude, longitude, altitude.
    
    Args:
        acmi_path: Path to ACMI file
        output_path: Path to output CSV file
        converter: CoordinateConverter instance
        scale_x1000: If True, scales X,Y,Z by 1000 (for normalized values)
    """
    data = parse_acmi_file(acmi_path)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Longitude', 'Latitude', 'Altitude', 
                      'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in data:
            # Scale coordinates if needed
            scale = 1000 if scale_x1000 else 1
            x_m = record['x'] * scale
            y_m = record['y'] * scale
            z_m = record['z'] * scale
            
            # Convert to lat/lon/alt
            lat, lon, alt = converter.ecef_to_latlon(x_m, y_m, z_m)
            
            # Write row
            row = {
                'Time': f"{record['time']:.2f}",
                'Longitude': f"{lon:.8f}",
                'Latitude': f"{lat:.8f}",
                'Altitude': f"{alt:.2f}",
                'Roll (deg)': f"{record['roll']:.5f}",
                'Pitch (deg)': f"{record['pitch']:.5f}",
                'Yaw (deg)': f"{record['yaw']:.5f}",
            }
            writer.writerow(row)
    
    print(f"CSV created: {output_path}")
    print(f"Total records: {len(data)}")


def main():
    pytorch_dir = Path(__file__).parent
    
    # Initialize converter with reference point
    # You can modify these coordinates to match your simulation area
    converter = CoordinateConverter(
        ref_latitude=32.72722222,
        ref_longitude=-116.6847222,
        ref_altitude=34
    )
    
    # Convert ACMI files
    acmi_files = [
        (pytorch_dir / 'single_env.acmi', pytorch_dir / 'single_env_coords.csv'),
        (pytorch_dir / 'batch_env_subset.acmi', pytorch_dir / 'batch_env_subset_coords.csv'),
    ]
    
    for acmi_file, output_file in acmi_files:
        if acmi_file.exists():
            print(f"\nProcessing: {acmi_file.name}")
            convert_to_csv_with_coords(acmi_file, output_file, converter, scale_x1000=True)
        else:
            print(f"File not found: {acmi_file}")


if __name__ == '__main__':
    main()
