import re
import numpy as np
import argparse
import os
import sys

def process_relations_with_slope(filename, output_filename, S=1.0):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse the header information
    if not lines[0].strip() == "HNF":
        raise ValueError("First line must be 'HNF'.")
    size_parts = lines[1].split(',')
    n_i, n_j = map(int, size_parts)
    
    # Parse lattice info
    lattice_line = lines[2]
    coord_matches = re.findall(r'\((-?[0-9.eE+-]+),\s*(-?[0-9.eE+-]+)\)', lattice_line)
    (a, b), (_, _), (e, f) = [(float(x), float(y)) for x, y in coord_matches]
    
    # Process and write the modified file
    with open(output_filename, 'w') as f:
        # Write header unchanged
        f.write(lines[0])
        f.write(lines[1])
        f.write(lines[2])
        
        current_position = None
        for line in lines[3:]:
            line = line.strip()
            if not line:
                f.write('\n')
                continue
                
            if line.startswith('G,'):
                # Grid point line - write unchanged and store position
                f.write(line + '\n')
                match = re.match(r'G,(\d+),(\d+),\s*\(([^,]+),([^)]+)\)', line)
                if match:
                    x = float(match.group(3))
                    y = float(match.group(4))
                    current_position = (x, y)
            else:
                # Process slope and relations line
                if current_position is None:
                    raise ValueError("Found module line before any grid point.")
                
                parts = line.split(',')
                slope = float(parts[0])
                
                # Parse relations
                relations = []
                for coord_str in parts[1:]:
                    coord_str = coord_str.strip()
                    coord_match = re.match(r'\(([^;]+);([^)]+)\)', coord_str)
                    if coord_match:
                        x = float(coord_match.group(1))
                        y = float(coord_match.group(2))
                        relations.append((x, y))
                
                if len(relations) == 0:
                    f.write(line + '\n')
                    continue
                
                # relations should be sorted by x-coordinate
                # relations.sort(key=lambda rel: rel[0])
                
                # Convert to relative coordinates
                pos_x, pos_y = current_position
                relative_relations = [(x - pos_x, y - pos_y) for x, y in relations]
                
                # Find transition point where slope changes from >= S to < S
                left_idx = None
                right_idx = None
                vertical_slope = False
                for i, (rel_x, rel_y) in enumerate(relative_relations):
                    if rel_x != 0:  # Avoid division by zero
                        current_slope = rel_y / rel_x
                        if current_slope < S:
                            right_idx = i
                            break
                        else: 
                            left_idx = i
                            vertical_slope = False
                    else:
                        vertical_slope = True
                        transition_idx = i
                
                # Calculate new point with slope exactly S
                candidates = []

                if left_idx is not None:
                    # Left candidate: keep y, adjust x to have slope S
                    left_rel = relative_relations[left_idx]
                    left_x, left_y = left_rel
                    left_candidate = (left_y / S, left_y)
                    candidates.append(left_candidate)

                if right_idx is not None:
                    # Right candidate: keep x, adjust y to have slope S
                    right_rel = relative_relations[right_idx]
                    right_x, right_y = right_rel
                    right_candidate = (right_x, S * right_x)
                    candidates.append(right_candidate)

                # Check if we have any candidates
                if not candidates:
                    raise ValueError("No valid left or right relation found for transition calculation")

                # Choose the smallest candidate
                chosen_relative = min(candidates)

                # Convert back to absolute coordinates
                new_relation = (chosen_relative[0] + pos_x, chosen_relative[1] + pos_y)

                # Write the modified line with calculated relation
                f.write(f"{slope},({new_relation[0]:.14f};{new_relation[1]:.14f})\n")

def create_output_filename(input_filename):
    """Create output filename by adding '_diagonal' before the extension"""
    base, ext = os.path.splitext(input_filename)
    return f"{base}_diagonal{ext}"

def main():
    parser = argparse.ArgumentParser(description='Process grid file with diagonal slope filtering')
    parser.add_argument('input_file', nargs='?', 
                       default='/home/wsljan/AIDA/Persistence-Algebra/test_presentations/two_circles_2_dim1_minpres.sky',
                       help='Input file path')
    parser.add_argument('-s', '--slope', type=float, default=1.0,
                      help='Slope parameter S (Steigung) (default: 1.0)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Create output filename
    output_file = create_output_filename(args.input_file)
    
    try:
        print(f"Processing file: {args.input_file}")
        print(f"Using slope S = {args.slope}")
        print(f"Output file: {output_file}")
        process_relations_with_slope(args.input_file, output_file, args.slope)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()