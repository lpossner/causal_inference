import pygpc
import numpy as np
from collections import OrderedDict


# seed RNG
seed = 1
np.random.seed(seed)

# define the properties of the random variables
parameters = OrderedDict()
parameters["x1"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x2"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x3"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x4"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x5"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x6"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x7"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x8"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x9"] = pygpc.Norm(pdf_shape=[0, 1])
parameters["x10"] = pygpc.Norm(pdf_shape=[0, 1])

# create grid object
grid = pygpc.LHS(parameters_random=parameters, options={"seed": seed, "criterion": "ese"}, n_grid=1000)

# save grid.coords
np.save("grid_coords.npy", grid.coords)
