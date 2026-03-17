#!/usr/bin/env python3
"""
Convert ACMI files (Tacview format) to CSV for easier viewing and analysis.
"""

import re
import csv
from pathlib import Path


def parse_acmi_file(acmi_path):
    """
    Parse ACMI file and extract telemetry data.
    
    Returns:
        tuple: (headers_dict, data_list) where data_list contains dicts with telemetry at each timestep
    """
    headers = {}
    data = []
    current_time = None
    current_objects = {}
    
    with open(acmi_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Parse header lines
        if '=' in line and not line.startswith('#') and not line.startswith('T='):
            if ',' not in line.split('=')[0]:  # Simple key=value pairs in header
                key, value = line.split('=', 1)
                headers[key] = value
            else:
                # Object property line (e.g., "0001,Coalition=0,Country=0,...")
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
                # Extract telemetry data
                telem_str = line[2:]  # Remove 'T='
                values = telem_str.split('|')
                
                if len(values) >= 9:
                    # Get the last object ID we saw
                    for obj_id, obj_props in current_objects.items():
                        record = {
                            'Time': current_time,
                            'ObjectID': obj_id,
                            'Callsign': obj_props.get('Callsign', ''),
                            'Coalition': obj_props.get('Coalition', ''),
                            'Country': obj_props.get('Country', ''),
                            'Type': obj_props.get('Type', ''),
                            'X': values[0],
                            'Y': values[1],
                            'Z': values[2],
                            'Roll': values[3],
                            'Pitch': values[4],
                            'Yaw': values[5],
                            'Speed_X': values[6],
                            'Speed_Y': values[7],
                            'Speed_Z': values[8],
                        }
                        data.append(record)
    
    return headers, data


def write_csv(output_path, headers, data):
    """Write parsed data to CSV file."""
    if not data:
        print("No data to write!")
        return
    
    # Define CSV columns
    fieldnames = [
        'Time', 'ObjectID', 'Callsign', 'Coalition', 'Country', 'Type',
        'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Speed_X', 'Speed_Y', 'Speed_Z'
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data rows
        for row in data:
            # Filter to only include fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)
    
    print(f"CSV file created: {output_path}")
    print(f"Total records: {len(data)}")


def main():
    pytorch_dir = Path(__file__).parent
    
    # Convert both ACMI files
    acmi_files = [
        pytorch_dir / 'single_env.acmi',
        pytorch_dir / 'batch_env_subset.acmi'
    ]
    
    for acmi_file in acmi_files:
        if acmi_file.exists():
            print(f"\nProcessing: {acmi_file.name}")
            headers, data = parse_acmi_file(acmi_file)
            
            csv_path = acmi_file.with_suffix('.csv')
            write_csv(csv_path, headers, data)
        else:
            print(f"File not found: {acmi_file}")


if __name__ == '__main__':
    main()
