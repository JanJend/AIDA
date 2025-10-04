
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def plot_3d_density(filename):
    # Read data from file
    data = np.loadtxt(filename)
    
    # Extract coordinates and density values
    x = data[:, 0]
    y = data[:, 1] 
    z = data[:, 2]
    density = data[:, 3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(x, y, z, c=density, cmap='viridis', s=20, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot with Density Color Coding')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename.txt>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        plot_3d_density(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")