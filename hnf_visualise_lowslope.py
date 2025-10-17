from fileinput import filename
import re
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

def parse_grid_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip first line "HNF"
    if not lines[0].strip() == "HNF":
        raise ValueError("First line must be 'HNF'.")

    # Parse second line: grid size
    size_parts = lines[1].split(',')
    if len(size_parts) != 2:
        raise ValueError("Second line must contain two comma-separated values.")

    n_i, n_j = map(int, size_parts)

     # Parse lattice info: (a,b), (c,d), (e,f)
    lattice_line = lines[2]
    coord_matches = re.findall(r'\((-?[0-9.eE+-]+),\s*(-?[0-9.eE+-]+)\)', lattice_line)

    if len(coord_matches) != 3:
        raise ValueError("Expected 3 (x, y) coordinate pairs in third line.")

    (a, b), (_, _), (e, f) = [(float(x), float(y)) for x, y in coord_matches]

    # Generate lattice_coords from (a + i*e, b + j*f)
    lattice_coords = [
        (a + i * e, b + j * f)
        for i in range(n_i)
        for j in range(n_j)
    ]

    # Now initialize the grid array
    grid_data = np.empty((n_i, n_j), dtype=object)

    # Process the rest of the lines
    current_point = None

    for line in lines[3:]:  # Start processing after the third line
        line = line.strip()
        if not line:
            continue

        if line.startswith('G,'):
            # New grid point
            match = re.match(r'G,(\d+),(\d+),\s*\(([^,]+),([^)]+)\)', line)
            if not match:
                raise ValueError(f"Invalid grid point format: {line}")
            i = int(match.group(1))
            j = int(match.group(2))
            x = float(match.group(3))
            y = float(match.group(4))
            current_point = (i, j)
            grid_data[i, j] = {
                'position': (x, y),
                'modules': []
            }
        else:
            if current_point is None:
                raise ValueError("Found module line before any grid point.")

            # Now process the slope and relations
            parts = line.split(',')
            slope = float(parts[0])

            relations = []
            for coord_str in parts[1:]:
                # Strip spaces and check for valid (x;y) format for relations
                coord_str = coord_str.strip()
                coord_match = re.match(r'\(([^;]+);([^)]+)\)', coord_str)
                if not coord_match:
                    raise ValueError(f"Invalid relation coordinate format: {coord_str}")
                x = float(coord_match.group(1))
                y = float(coord_match.group(2))
                relations.append((x, y))

            # Add the slope and relations to the current grid point
            i, j = current_point
            grid_data[i, j]['modules'].append((slope, relations))

    return n_i, n_j, lattice_coords, grid_data


def generate_grayscale_images(n_i, n_j, grid_data, lattice_coords, output_file, m=3):
    # Determine the grid size (in pixels)
    image_width = n_j * m
    image_height = n_i * m

    # Initialize empty arrays for the images
    img1 = np.zeros((n_i, n_j))  # For hilbert function
    img2 = np.zeros((n_i, n_j))  # For square-slope-based grayscale
    img3 = np.zeros((n_i, n_j))  # For slope-based grayscale

    # Calculate max number of modules for normalization in img1
    max_modules = max(len(grid_data[i, j]['modules']) if grid_data[i, j] is not None else 0 for i in range(n_i) for j in range(n_j))

    for i in range(n_i):
        for j in range(n_j):
            if grid_data[i, j] is not None:
                # Image 1: Grayscale based on module count
                num_modules = len(grid_data[i, j]['modules'])
                img1[i, j] = num_modules / max_modules  # Normalize based on max number of modules
                # Image 2: Grayscale based on sum of squared and inverted slopes
                from math import log

                slope_sum = 0.0
                for slope, relations in grid_data[i, j]['modules']:
                    slope_sum += 1/slope # slope
                img2[i, j] = slope_sum
                slope_sum = 0.0
                for slope, relations in grid_data[i, j]['modules']:
                    slope_sum += 1/(slope**2)  # squared slopes
                img3[i, j] = slope_sum

    # Scale img2 so that the highest value becomes 1 (white)
    img2 = img2 / np.max(img2)  # Normalize to [0, 1] for proper grayscale representation
    img3 = img3 / np.max(img3)  # Normalize to [0, 1] for proper grayscale representation

    # Now, create the images by expanding them with factor m
    img1_resized = np.kron(img1, np.ones((m, m)))  # Resize by repeating each grid point's value
    img2_resized = np.kron(img2, np.ones((m, m)))  # Resize by repeating each grid point's value
    img3_resized = np.kron(img3, np.ones((m, m)))  # Resize by repeating each grid point's value

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    num_x_ticks = min(7, n_j)
    num_y_ticks = min(7, n_i)
    x_tick_indices = np.linspace(0, n_j - 1, num_x_ticks, dtype=int)
    y_tick_indices = np.linspace(0, n_i - 1, num_y_ticks, dtype=int)
    # Calculate tick positions in image coordinates
    x_tick_positions = x_tick_indices * m
    y_tick_positions = y_tick_indices * m

    # Image 1: Grayscale based on module count
    axes[0].imshow(img1_resized, cmap='gray', origin='upper', vmin=0, vmax=1)
    axes[0].set_title('Dimension')
    axes[0].invert_yaxis()  # Flip the Y-axis for image 1
    axes[0].set_xlabel('Scale')  # Set X-axis label
    axes[0].set_ylabel('CoDensity')  # Set Y-axis label

    # Set ticks and labels
    axes[0].set_xticks(x_tick_positions)
    axes[0].set_yticks(y_tick_positions)
    axes[0].set_xticklabels([f'{lattice_coords[i * n_j + 0][0]:.2f}' for i in x_tick_indices])
    axes[0].set_yticklabels([f'{lattice_coords[0 * n_j + j][1]:.2f}' for j in y_tick_indices])


    # Image 2: Grayscale based on slope sum (inverted and squared)
    axes[1].imshow(img2_resized, cmap='gray', origin='upper', vmin=0, vmax=1)
    axes[1].set_title('Inverse Slope Sum')
    axes[1].invert_yaxis()  # Flip the Y-axis for image 2
    axes[1].set_xlabel('Scale')  # Set X-axis label
    axes[1].set_ylabel('CoDensity')  # Set Y-axis label
    
    # Set ticks and labels
    axes[1].set_xticks(x_tick_positions)
    axes[1].set_yticks(y_tick_positions)
    axes[1].set_xticklabels([f'{lattice_coords[i * n_j + 0][0]:.2f}' for i in x_tick_indices])
    axes[1].set_yticklabels([f'{lattice_coords[0 * n_j + j][1]:.2f}' for j in y_tick_indices])

    # Image 3: Grayscale based on squared slope sum 
    axes[2].imshow(img3_resized, cmap='gray', origin='upper', vmin=0, vmax=1)
    axes[2].set_title('Inverse Slope^2 Sum')
    axes[2].invert_yaxis()  # Flip the Y-axis for image 2
    axes[2].set_xlabel('Scale')  # Set X-axis label
    axes[2].set_ylabel('CoDensity')  # Set Y-axis label

    # Set ticks and labels
    axes[2].set_xticks(x_tick_positions)
    axes[2].set_yticks(y_tick_positions)
    axes[2].set_xticklabels([f'{lattice_coords[i * n_j + 0][0]:.2f}' for i in x_tick_indices])
    axes[2].set_yticklabels([f'{lattice_coords[0 * n_j + j][1]:.2f}' for j in y_tick_indices])
    

    # Ensure equal aspect ratio for both images
    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')
    axes[2].set_aspect('equal', 'box')

    # Save images
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def create_output_filename(input_filename, intensity, magnitude):
    # Get the directory and filename from input
    input_dir = os.path.dirname(input_filename)
    base_name = os.path.basename(input_filename)
    base, ext = os.path.splitext(base_name)
    
    # Create new filename
    new_filename = f"{base}_low_theta_{intensity}_mag_{magnitude}.png"
    
    # Join with the input directory to keep it in the same folder
    return os.path.join(input_dir, new_filename)

def main():
    parser = argparse.ArgumentParser(description='Process grid file with diagonal slope filtering')
    parser.add_argument('input_file', nargs='?', 
                       default='/presentations/two_circles.sky',
                       help='Input file path')
    parser.add_argument('-i', '--intensity', type=float, default=1.0,
                      help='Intensity parameter (default: 1.0)')
    parser.add_argument('-m', '--magnitude', type=int, default=3,
                      help='Magnitude factor for image scaling (default: 3)')
    parser.add_argument('-o', '--output_file', type=str, default=None,
                      help='Output file path (optional)')
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = create_output_filename(args.input_file, args.intensity, args.magnitude)

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    try:
        print(f"Processing file: {args.input_file}")
        print(f"Using intensity = {args.intensity}")
        n_i, n_j, lattice_coords, grid_data = parse_grid_file(args.input_file)
        print(f"Grid size: {n_i} x {n_j}")
        generate_grayscale_images(n_i, n_j, grid_data, lattice_coords, args.output_file, m=3)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
