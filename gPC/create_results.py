import os
import numpy as np


# seed RNG
seed = 1
np.random.seed(seed)

# load coordinates
coords = np.load("data/coords.npy")

# run simulations with grid.coords
f = np.poly1d([1, 2, 3, 4])
results = f(coords)[:]

# save results
os.makedirs("data", exist_ok=True)
np.save("data/results.npy", results)
