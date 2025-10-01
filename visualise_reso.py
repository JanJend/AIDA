from turtle import pos
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import Normalize
import os
from pathlib import Path
from decimal import Decimal, getcontext
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import LogLocator, FixedFormatter

getcontext().prec = 20 

def Hilbert_function(generators, relations, syzygies=None, resolution=(500, 500), path=None, xlim=None, ylim=None):
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

    masked = np.ma.masked_equal(hilbert_vals, 0)
    pos = hilbert_vals[hilbert_vals > 0]

    # pos = array of strictly positive Hilbert function values
    if pos.size > 0:
        vmin = pos.min()   # smallest strictly positive value
        vmax = pos.max()
    else:
        vmin = 1    
        vmax = 2 
    if vmin == vmax:
        vmax = vmin + 1
        

    # logarithmic normalization on positive values
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    # white -> light blue -> mid blue -> dark blue -> black
    cmap = LinearSegmentedColormap.from_list(
        "white_blue_black",
        ["#d4eaff", "#006aff", "black"]
    )
    cmap.set_bad("white")   # masked (zeros) -> white
    
    # plot
    im = plt.imshow(masked, cmap=cmap, origin='lower',
                    extent=(x_min, x_max, y_min, y_max), aspect='auto',
                    interpolation='nearest', norm=norm)

    cbar = plt.colorbar(im)
    cbar.set_label('Dimension')
    locator = LogLocator(base=10, subs=[1, 2, 3, 5], numticks=100)
    cbar.locator = locator
    cbar.update_ticks()
    
    # force formatter to label *all* ticks, not only 1,10,100
    formatter = LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
    cbar.ax.yaxis.set_major_formatter(formatter)

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
            print(f"Reading resolution from {filepath}")
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


def read_scc_block(file, block_index):
    """Read one scc2020 block from file at current position."""
    header_type = file.readline().strip()
    if header_type != '3':
        print(f"Error in block {block_index}: Expected '3' on second line.")
        return None

    line = file.readline().strip()
    no_syz, no_rel, no_gen = map(int, line.split())

    syzygies, relations, generators = [], [], []

    for i in range(no_syz):
        entry = parse_line(file.readline().strip(), is_relation=True)
        if entry: syzygies.append(entry)

    for i in range(no_rel):
        entry = parse_line(file.readline().strip(), is_relation=True)
        if entry: relations.append(entry)

    for i in range(no_gen):
        entry = parse_line(file.readline().strip(), is_relation=False)
        if entry: generators.append(entry)

    return syzygies, relations, generators

def read_sccsum(filepath, param=5):
    """Parse .sccsum file and read only the `param` largest blocks."""
    print(f"Reading .sccsum file from {filepath}, selecting {param} largest blocks")
    try:
        with open(filepath, 'r') as f:
            first = f.readline().strip()
            if first != "scc2020sum":
                print("Error: File must start with 'scc2020sum'")
                return []

            num_blocks = int(f.readline().strip())
            block_infos = []  # store (size, start_pos) for each block

            for block_idx in range(1, num_blocks + 1):
                block_start = f.tell()
                f.readline()  # skip empty line
                f.readline()  # skip [type] line

                header = f.readline().strip()
                if header != "scc2020":
                    print(f"Error: expected 'scc2020' at start of block {block_idx}")
                    sys.exit(1)
                length = int(f.readline().strip())
                if length != 3:
                    print(f"Error: expected '3' for length in block {block_idx}")
                    sys.exit(1)

                counts_line = f.readline().strip()
                try:
                    counts = [int(x) for x in counts_line.split()]
                    if len(counts) != 3:
                        sys.exit(1)
                except ValueError:
                    print(f"Error parsing counts at block {block_idx} with start {block_start}." 
                          "Expected three numbers, got '{counts_line}'")

                total_size = sum(counts)
                block_infos.append((total_size, block_start))

                # skip the rest of the block
                for _ in range(total_size):  # adjust if each item has more lines
                    f.readline()

            # select the param largest blocks
            block_infos.sort(reverse=True, key=lambda x: x[0])
            largest_blocks = block_infos[:param]

            results = []
            for _, start_pos in largest_blocks:
                f.seek(start_pos)
                f.readline()  # skip empty line
                f.readline()  # skip [type] line
                header = f.readline().strip()
                if header != "scc2020":
                    continue

                block_data = read_scc_block(f, None)  # optional: pass block index if needed
                if block_data:
                    results.append(block_data)

            return results

    except FileNotFoundError:
        print(f"Error: Unable to open file {filepath}")
        return []

if __name__ == "__main__":

    default_path = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/two_circles_2_dim1_minpres_resolution.scc"

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file.scc | input_file.sccsum> [optional_integer_for_sccsum]")        
        input_file = default_path
    else:   
        input_file = sys.argv[1]

    if input_file.endswith(".scc"):
        resolution_data = read_resolution(input_file)
        if resolution_data:
            syzygies, relations, generators = resolution_data
            print(f"File {input_file}: {len(syzygies)} syzygies, {len(relations)} relations, {len(generators)} generators")
            Hilbert_function(
                generators,
                relations,
                syzygies,
                resolution=(300, 300),
                path=Path(input_file).stem + "_HF",
            )

    elif input_file.endswith(".sccsum"):
        # Optional integer parameter
        param = int(sys.argv[2]) if len(sys.argv) > 2 else 5

        all_blocks = read_sccsum(input_file, param)

        # compute global x/y bounds across all selected blocks
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for syzygies, relations, generators in all_blocks:
            for pt in syzygies + relations + generators:
                x, y = pt[0], pt[1]  # adjust if your structure is different
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        # fallback if no points exist
        if x_min == float('inf'):
            x_min, x_max, y_min, y_max = 0, 1, 0, 1
        x_min, x_max = float(x_min), float(x_max)
        y_min, y_max = float(y_min), float(y_max)
        padding = 0.1
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding * 2
        print(f"Data range is in x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")

        for idx, (syzygies, relations, generators) in enumerate(all_blocks, start=1):
            print(f"Block {idx}: {len(syzygies)} syzygies, {len(relations)} relations, {len(generators)} generators")
            Hilbert_function(
                generators,
                relations,
                syzygies,
                resolution=(300, 300),
                path=Path(input_file).stem + "_HF" + str(idx),
                xlim=(x_min, x_max),
                ylim=(y_min, y_max)
            )

    else:
        print("Error: input must be either a .scc or .sccsum file")
        sys.exit(1)


"""
Folder = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/torus_100_0.10_dim1_decomposition"


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
            resolution=(500, 500),
            path=input_path,
        )
"""