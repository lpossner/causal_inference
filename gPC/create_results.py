import numpy as np


# seed RNG
seed = 1
np.random.seed(seed)

# load coordinates
coords = np.load("grid_coords.npy")

# run simulations with grid.coords
f = np.poly1d([1, 2, 3, 4])
results = f(coords)[:, np.newaxis]

# save results
np.save("results.npy", results)
