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
                relations.append(parse_line(line))
                counter += 1

            while counter < no_rel + no_gen:
                line = file.readline().strip()
                generators.append(parse_line(line))
                counter += 1

            return relations, generators

    except FileNotFoundError:
        print(f"Error: Unable to open file {filepath}")
        return None


def parse_line(line):
    # Split the line by semicolon
    parts = line.split(';')

    # Extract the first two real numbers
    try:
        real1 = float(parts[0].split()[0])
        real2 = float(parts[0].split()[1])
        return (real1, real2)
    except ValueError:
        print("Error: Unable to parse real numbers from the line.")
        return None

def visualize_presentation(generators, relations):
    # Merge generators and relations
    data = generators + relations

    # Unpack x and y coordinates
    x, y = zip(*data)

    # Plot the points
    plt.scatter(x, y, label='Data Points', color='blue', marker='o')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Visualization of Generators and Relations')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

def visualize_graph(degree_filepath, edge_filepath):
    try:
        degree_df = pd.read_csv(degree_filepath)
        edge_df = pd.read_csv(edge_filepath)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except pd.errors.EmptyDataError:
        print("Error: Empty data in one or both CSV files.")
        return

    G = nx.DiGraph()

    # Add nodes with positions
    for index, row in degree_df.iterrows():
        G.add_node(index, pos=(row['X'], row['Y']))

    # Add edges
    for _, row in edge_df.iterrows():
        source, target = row['Source'], row['Target']
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target)

    # Check for nodes without positions
    nodes_without_position = [node for node, data in G.nodes(data=True) if 'pos' not in data]
    if nodes_without_position:
        print(f"Warning: Nodes {nodes_without_position} have no position.")

    # Draw the graph
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=700, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=10)

    plt.show()

# Call the function with the correct file paths
visualize_graph('/home/wsljan/generalized_persistence/code/degreeList.csv', '/home/wsljan/generalized_persistence/code/edgesBeforeReduction.csv')


# Example usage
# presentation_data = read_presentation("/home/wsljan/compression_for_2_parameter_persistent_homology_data_and_benchmarks/comparison_with_rivet_datasets/noisy_circle_firep_8_0_min_pres.firep")
# if presentation_data:
#    relations, generators = presentation_data
#    visualize_presentation(generators, relations)
