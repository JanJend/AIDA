import re
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


def mp_landscape(filename, output_filename, theta=0, k=1):
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
    
    # Initialize diagonal functions
    num_diagonals = n_i + n_j - 1
    f_diagonals = [dict() for _ in range(num_diagonals)]
    
    def get_diagonal_index(i, j):
        return i - j + (n_j - 1)
    
    def update_diagonal(diag_idx, x, y, r1, r2):
        d = r1 / 2
        m_x = x + d
        m_y = y + d
        c = x - y
        
        t_min = x - d
        t_max = x + 2*d
        num_samples = max(100, int(4 * d * 10))
        t_values = np.linspace(t_min, t_max, num_samples)
        
        for t in t_values:
            l1_dist = abs(m_x - t) + abs(m_y - (t - c))
            height = max(0, d - l1_dist)
            
            if t in f_diagonals[diag_idx]:
                f_diagonals[diag_idx][t] = max(f_diagonals[diag_idx][t], height)
            else:
                f_diagonals[diag_idx][t] = height
    
    # Process file (same as before)
    current_position = None
    current_grid_indices = None
    pending_lines = []
    
    for line in lines[3:]:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('G,'):
            if current_position and pending_lines:
                valid_lines = [(t, r1, r2) for t, r1, r2 in pending_lines if t > theta]
                
                if len(valid_lines) >= k:
                    valid_lines.sort(key=lambda x: x[1], reverse=True)
                    _, r1, r2 = valid_lines[k - 1]
                    diag_idx = get_diagonal_index(current_grid_indices[0], current_grid_indices[1])
                    update_diagonal(diag_idx, current_position[0], current_position[1], r1, r2)
            
            pending_lines = []
            match = re.match(r'G,(\d+),(\d+),\s*\(([^,]+),([^)]+)\)', line)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                x = float(match.group(3))
                y = float(match.group(4))
                current_position = (x, y)
                current_grid_indices = (i, j)
        else:
            parts = line.split(',')
            theta_val = float(parts[0])
            for coord_str in parts[1:]:
                coord_str = coord_str.strip()
                coord_match = re.match(r'\(([^;]+);([^)]+)\)', coord_str)
                if coord_match:
                    r1 = float(coord_match.group(1))
                    r2 = float(coord_match.group(2))
                    pending_lines.append((theta_val, r1, r2))
    
    # Process last grid point
    if current_position and pending_lines:
        valid_lines = [(t, r1, r2) for t, r1, r2 in pending_lines if t > theta]
        if len(valid_lines) >= k:
            valid_lines.sort(key=lambda x: x[1], reverse=True)
            _, r1, r2 = valid_lines[k - 1]
            diag_idx = get_diagonal_index(current_grid_indices[0], current_grid_indices[1])
            update_diagonal(diag_idx, current_position[0], current_position[1], r1, r2)
    
    # Pre-process diagonal functions for fast lookup
    diag_funcs = []
    for diag_idx in range(num_diagonals):
        if f_diagonals[diag_idx]:
            t_vals = np.array(sorted(f_diagonals[diag_idx].keys()))
            heights = np.array([f_diagonals[diag_idx][t] for t in t_vals])
            diag_funcs.append((t_vals, heights))
        else:
            diag_funcs.append((None, None))
    
    # Create 3D plot with optimized grid filling
    resolution = 5  # Reduce from 10 to 5 for speed
    x_coords = np.linspace(0, n_i - 1, n_i * resolution)
    y_coords = np.linspace(0, n_j - 1, n_j * resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros_like(X)
    
    # Vectorized approach: process each diagonal at once
    for diag_idx in range(num_diagonals):
        t_vals, heights = diag_funcs[diag_idx]
        if t_vals is None:
            continue
        
        # Find all grid points on this diagonal
        c = diag_idx - (n_j - 1)  # diagonal constant: x - y = c
        
        # Mask for points on this diagonal (within tolerance)
        mask = np.abs((X - Y) - c) < 0.5 / resolution
        
        if np.any(mask):
            # Get x-coordinates of points on this diagonal
            x_on_diag = X[mask]
            
            # Vectorized interpolation
            z_on_diag = np.interp(x_on_diag, t_vals, heights, left=0, right=0)
            Z[mask] = z_on_diag
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('Height')
    ax.set_title(f'MP Landscape (theta={theta}, k={k})')
    fig.colorbar(surf, shrink=0.5)
    plt.savefig(output_filename)
    plt.show()

def create_output_filename(input_filename, dimension, theta):
    """Create output filename by adding '_landscape' before the extension"""
    base, ext = os.path.splitext(input_filename)
    return f"{base}_landscape_k{dimension}_theta{theta}.png"

def main():
    parser = argparse.ArgumentParser(description='Process diagonal hnf')
    parser.add_argument('input_file', nargs='?', 
                       default='tests/test_presentations/two_circles_diagonal_1.0.sky',
                       help='Input file path')
    parser.add_argument('-k', '--dimension', type=int, default=1,
                      help='Dimension parameter k (default: 1)')
    parser.add_argument('-t', '--theta', type=float, default=0.0,
                      help='Theta parameter (default: 0.0)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Create output filename
    output_file = create_output_filename(args.input_file, args.dimension, args.theta)
    
    try:
        print(f"Processing file: {args.input_file}")
        print(f"Using dimension k = {args.dimension}")
        print(f"Output file: {output_file}")
        mp_landscape(args.input_file, output_file, args.theta, args.dimension)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()