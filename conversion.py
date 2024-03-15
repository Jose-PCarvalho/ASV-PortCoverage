import numpy as np
import PIL.Image as img
import matplotlib.pyplot as plt
import PIL.ImageDraw as draw
import yaml
import sys

edge_map = np.load('maps/edge_map2.npy')
area_map = np.load('maps/area_map2.npy')
image = img.fromarray(edge_map)
image = image.rotate(-45, resample=img.NEAREST)
image.show()
rows = 30
cols = 30

# Calculate the size of each cell
cell_height = edge_map.shape[0] // rows
cell_width = edge_map.shape[1] // cols

# Create a new array to store the discretized image
edge_map = np.array(image)
discretized_image = np.zeros_like(np.array(image))

# Iterate over the cells and assign the average value to each cell
for i in range(rows):
    for j in range(cols):
        cell = edge_map[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width]
        cell_average = np.mean(cell)
        if cell_average >200:
            discretized_image[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width] = 255
        else:
            discretized_image[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width] = 0

# Create a PIL image from the discretized array

discretized_image_pil = img.fromarray(discretized_image.astype(np.uint8)).resize((cols, rows), img.NEAREST)

discretized_image_pil.show()

yaml_array = np.where(np.array(discretized_image_pil) == 0, -1, 0)
#arr = -1 * np.ones((30, 30),dtype=np.uint8)
#arr[:, 8:23] = yaml_array
#yaml_array = arr
print(np.count_nonzero(yaml_array))
with open('maps/map2.yaml', 'w') as f:
    sys.stdout = f
    print("map:")
    for i in range(yaml_array.shape[0]):
        print("- [", end=" ")
        for j in range(yaml_array.shape[1]):
            print(yaml_array[i][j], end=", ")
        print("]")
