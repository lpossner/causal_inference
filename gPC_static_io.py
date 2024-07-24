"""
Algorithm: Static_IO
==============================
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
import matplotlib
# matplotlib.use("Qt5Agg")

from collections import OrderedDict

fn_results = 'data/static_IO'   # filename of output
save_session_format = ".pkl"   # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
np.random.seed(1)

#%%
# Setup input and output data
########################################################################################################################
# define the properties of the random variables
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])

# create grid object
grid = pygpc.LHS(parameters_random=parameters, options={"seed": None, "criterion": "ese"}, n_grid=1000)

# save grid.coords and grid.coords in .txt file
np.savetxt("grid_coords.txt", grid.coords)

# Run simulations
########################################################################################################################
# load coordinates to run simulations with
coords_loaded = np.loadtxt("grid_coords.txt")

# run simulations with grid.coords
results = 1

# save results
np.savetxt("results.txt")

# Run gPC
########################################################################################################################
# re-generate grid object from grid.coords
coords_loaded = np.loadtxt("grid_coords.txt")
grid = pygpc.LHS(parameters_random=parameters, coords=coords_loaded)

# read results data
results = np.loadtxt("results.txt")

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [10]
options["order_max"] = 10
options["interaction_order"] = 1
options["error_type"] = "loocv"
options["error_norm"] = "absolute"
options["n_samples_validation"] = None
options["fn_results"] = fn_results
options["save_session_format"] = save_session_format
options["backend"] = "omp"
options["verbose"] = True

# determine number of gPC coefficients (hint: compare it with the amount of output data you have)
# Tipp: 4x mehr sims als n_coeffs für stabile invertierung, Restfehler von Polynonordnung und Problem an sich
n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                       order_glob_max=options["order_max"],
                                       order_inter_max=options["interaction_order"],
                                       dim=len(parameters))

# define algorithm
algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results)

#%%
# Running the gpc
# ---------------

# initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC algorithm
session, coeffs, results = session.run()

#%%
# Postprocessing
# --------------

# read session
session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

# Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="standard")

# get a summary of the sensitivity coefficients
sobol, gsens = pygpc.get_sens_summary(fn_results, parameters)
print(sobol)
print(gsens)

# plot gPC approximation and IO data
pygpc.plot_gpc(session=session,
               coeffs=coeffs,
               random_vars=["x1", "x3"],
               output_idx=0,
               n_grid=[100, 100],
               coords=grid.coords,
               results=results)

# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
