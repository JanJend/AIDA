import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import os
from pathlib import Path
from decimal import Decimal, getcontext

getcontext().prec = 20 

def Hilbert_function(generators, relations, syzygies=None, resolution=(300, 300), path=None, xlim=None, ylim=None):
    if syzygies is None:
        syzygies = []

    width, height = resolution

    # Set figure size to match resolution ratio
    fig_width = 6
    fig_height = fig_width * (height / width)
    plt.figure(figsize=(fig_width, fig_height))

    # Convert to numpy arrays for broadcasting
    generators = np.array([(float(x), float(y)) for (x, y) in generators])
    relations = np.array([(float(x), float(y)) for (x, y, _) in relations])
    syzygies = np.array([(float(x), float(y)) for (x, y, _) in syzygies])

    # Determine plot limits
    if xlim is not None and ylim is not None:
        x_min, x_max = xlim
        y_min, y_max = ylim
    else:
        all_points = [generators, relations, syzygies]
        all_points = [p for p in all_points if p.size > 0]
        all_points = np.vstack(all_points) if all_points else np.zeros((1, 2))
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)

        # Add padding only if auto mode
        padding = 0.1
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding * 2

    # Create grid
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    xx, yy = np.meshgrid(x, y)
    hilbert_vals = np.zeros_like(xx, dtype=int)

    # Add syzygies and generators
    if syzygies.size == 0:
        combined = generators
    else:
        combined = np.vstack([generators, syzygies])

    for px, py in combined:
        hilbert_vals += ((xx >= px) & (yy >= py)).astype(int)

    # Subtract relations
    for rx, ry in relations:
        hilbert_vals -= ((xx >= rx) & (yy >= ry)).astype(int)

    hilbert_vals = np.maximum(hilbert_vals, 0)
    max_val = np.max(hilbert_vals)
    norm = Normalize(vmin=0, vmax=max_val)

    # Plot
    im = plt.imshow(hilbert_vals, cmap='Blues', origin='lower',
                    extent=(x_min, x_max, y_min, y_max), aspect='auto',
                    interpolation='nearest', norm=norm)
    

    cbar = plt.colorbar(im)
    cbar.set_label('Dimension')

    plt.text(x_min + 0.05 * (x_max - x_min), y_max - 0.05 * (y_max - y_min),
             f"Max value: {max_val}", color='black', fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hilbert Function')
    plt.legend()

    if path:
        output_path = path if path.endswith('.png') else f"{path}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('hilbert_function.png', dpi=300, bbox_inches='tight')
    plt.show()

def read_resolution(filepath):
    try:
        with open(filepath, 'r') as file:
            line = file.readline().strip()
            if "scc2020" not in line:
                print("Error: Expected 'scc2020' in first line.")
                return None

            header_type = file.readline().strip()
            if header_type != '3':
                print("Error: Expected a '3' on the second line for resolution format.")
                return None

            # Read dimensions: no_syzygies, no_relations, no_generators
            line = file.readline().strip()
            no_syz, no_rel, no_gen = map(int, line.split())

            syzygies = []
            relations = []
            generators = []

            # Read syzygies
            for i in range(no_syz):
                line = file.readline().strip()
                entry = parse_line(line, is_relation=True)
                if entry is not None:
                    syzygies.append(entry)
                else:
                    print(f"Warning: Failed to parse syzygy at line {i + 4}")

            # Read relations
            for i in range(no_rel):
                line = file.readline().strip()
                entry = parse_line(line, is_relation=True)
                if entry is not None:
                    relations.append(entry)
                else:
                    print(f"Warning: Failed to parse relation at line {i + 4 + no_syz}")

            # Read generators
            for i in range(no_gen):
                line = file.readline().strip()
                entry = parse_line(line, is_relation=False)
                if entry is not None:
                    generators.append(entry)
                else:
                    print(f"Warning: Failed to parse generator at line {i + 4 + no_syz + no_rel}")

            # Print counts
            print(f"Parsed {len(syzygies)} syzygies, {len(relations)} relations, {len(generators)} generators")

            return syzygies, relations, generators

    except FileNotFoundError:
        print(f"Error: Unable to open file {filepath}")
        return None


def parse_line(line, is_relation):
    parts = line.split(';')
    try:
        coords = parts[0].split()
        real1 = Decimal(coords[0])
        real2 = Decimal(coords[1])
        if is_relation:
            integers = list(map(int, parts[1].strip().split()))
            return (real1, real2, integers)
        else:
            return (real1, real2)
    except (ValueError, IndexError):
        print(f"Error parsing line: {line}")
        return None

"""
input = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/presentation/non_cyclic_summands/noisy_annulus_socg_largecomp_resolution.scc"
resolution_data = read_resolution(input)
if resolution_data:
    syzygies, relations, generators = resolution_data
    print(f"Syzygies: {len(syzygies)}, Relations: {len(relations)}, Generators: {len(generators)}")
    Hilbert_function(
        generators,
        relations,
        syzygies,
        resolution=(300, 300),
        path=input,
        xlim=(0.0, 0.7),
        ylim=(-1.0, 0.0)
    )
"""

Folder = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/presentation/non_cyclic_summands"


for file in Path(Folder).glob("*resolution.scc"):
    input_path = str(file)
    resolution_data = read_resolution(input_path)
    if resolution_data:
        syzygies, relations, generators = resolution_data
        print(f"{file.name}: Syzygies: {len(syzygies)}, Relations: {len(relations)}, Generators: {len(generators)}")
        Hilbert_function(
            generators,
            relations,
            syzygies,
            resolution=(300, 300),
            path=input_path,
            xlim=(0.0, 0.7),
            ylim=(-1.0, 0.0)
        )
