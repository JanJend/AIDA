import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the file
with open('/home/wsljan/AIDA/tests/test_presentations/two_circles_landscape.txt', 'r') as f:
    # Parse header
    header = f.readline().split()
    n_x = int(header[2])
    n_y = int(header[3])
    
    # Read the grid values
    landscape = []
    for line in f:
        if line.strip():
            landscape.append([float(x) for x in line.split()])

# Convert to numpy array
Z = np.array(landscape)

# Create x, y coordinate grids
X, Y = np.meshgrid(np.arange(n_y), np.arange(n_x))

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sky Landscape Surface')

plt.show()