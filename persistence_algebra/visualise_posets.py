import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def read_presentation(filepath):
    # Find the position of the dot in the file extension
    dot_position = filepath.rfind('.')

    if dot_position == -1:
        # No dot found, invalid file format
        print("Error: Invalid file format. File must have an extension (.scc or .firep).")
        return None

    try:
        with open(filepath, 'r') as file:
            relations = []
            generators = []
            extension = filepath[dot_position:]
            line = file.readline().strip()

            # Check the file extension and perform actions accordingly
            if extension == ".scc" or extension == ".firep":
                print(f"Reading presentation file: {filepath}")
                if "firep" in line:
                    print(f"Reading FIREP presentation file: {filepath}")
                    # Skip 2 lines for FIREP
                    file.readline()
                    file.readline()
                elif "scc2020" in line:
                    print(f"Reading SCC2020 presentation file: {filepath}")
                    # Skip 1 line for SCC2020
                    file.readline()
                else:
                    # Invalid file type
                    print("Error: Unsupported file type. Supported types are FIREP and SCC2020.")
                    return None
            else:
                # Invalid file extension
                print("Error: Unsupported file extension. Supported extensions are .scc and .firep.")
                return None

            # Parse the first line after skipping
            line = file.readline().strip()
            no_rel, no_gen, third_number = map(int, line.split())

            if third_number != 0:
                print("Error: Invalid format in the first line. Expecting exactly 3 numbers with the last one being 0.")
                return None

            counter = 0

            while counter < no_rel:
                line = file.readline().strip()
                relations.append(parse_line(line, is_relation=True))
                counter += 1

            while counter < no_rel + no_gen:
                line = file.readline().strip()
                generators.append(parse_line(line, is_relation=False))
                counter += 1

            return relations, generators

    except FileNotFoundError:
        print(f"Error: Unable to open file {filepath}")
        return None


def parse_line(line, is_relation):
    # Split the line by semicolon
    parts = line.split(';')

    # Extract the first two real numbers
    try:
        real1 = float(parts[0].split()[0])
        real2 = float(parts[0].split()[1])
        if is_relation:
            integers = list(map(int, parts[1].strip().split()))
            return (real1, real2, integers)
        else:
            return (real1, real2)
    except ValueError:
        print("Error: Unable to parse real numbers from the line.")
        return None


def visualize_presentation(generators, relations):
    # Unpack x and y coordinates for generators
    x_gen, y_gen = zip(*generators)

    # Unpack x and y coordinates for relations
    x_rel, y_rel, rel_indices = zip(*relations)

    # Plot the points for generators
    scatter_gen = plt.scatter(x_gen, y_gen, label='Generators', color='blue', marker='o')

    # Plot the points for relations
    scatter_rel = plt.scatter(x_rel, y_rel, label='Relations', color='red', marker='x')

    # Get the size of the markers
    marker_size_gen = scatter_gen.get_sizes()[0] if scatter_gen.get_sizes().size > 0 else 20
    marker_size_rel = scatter_rel.get_sizes()[0] if scatter_rel.get_sizes().size > 0 else 20

    # Calculate arrow head sizes relative to marker sizes
    head_width = marker_size_rel ** 0.5 / 200
    head_length = marker_size_rel ** 0.5 / 200

    # Draw lines from relations to generators
    for (x_r, y_r, indices) in zip(x_rel, y_rel, rel_indices):
        for idx in indices:
            x_g, y_g = generators[idx]
            plt.plot([x_r, x_g], [y_r, y_g], color='green', linewidth=0.5)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Visualization of Generators and Relations')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()



# Example usage
presentation_data = read_presentation("/home/wsljan/AIDA/persistence_algebra/test_presentations/full_rips_size_1_instance_5_min_pres.scc")
if presentation_data:
    relations, generators = presentation_data
    visualize_presentation(generators, relations)
