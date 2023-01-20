import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib
import pandas as pd
import numpy as np
import pyelsa as elsa
import time
import os

import tomophantom
from tomophantom.supp.artifacts import _Artifacts_
from tomophantom import TomoP2D
from tomophantom.supp.libraryToDict import modelfile2Dtolist

@dataclass
class SolverTest:
    solver_class: elsa.Solver
    solver_name: str
    is_gmres: bool = False
    is_unmatched: bool = False
    extra_args: dict = field(default_factory=dict)

def mse(optimized: np.ndarray, original: np.ndarray) -> float:
    size = original.size
    diff = (original - optimized) ** 2
    return np.sum(diff) / size

# function we need because tomophantom creates sinograms correctly like elsa but in a weird format
# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# Noise added like TomoPhantom/Demos/Python/Recon
def add_tomophantom_poisson(sinogram, noise_amplitude = 10000):
    sinogram = np.asarray(sinogram)
    # forming dictionaries with artifact types
    _noise_ =  {'noise_type' : 'Poisson',
                'noise_amplitude' : noise_amplitude, # noise amplitude
                'noise_seed' : 0}

    noisy_sino = _Artifacts_(sinogram, **_noise_)

    return noisy_sino

def tomophantom_model(model, size, num_angles, arc=180, noise=False):

    # select a model number from the library
    N_size = size # set dimension of the phantom
    # one can specify an exact path to the parameters file
    # path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
    path = os.path.dirname(tomophantom.__file__)
    path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
    objlist = modelfile2Dtolist(path_library2D, model) # extract parameters using Python
    #This will generate a N_size x N_size phantom (2D)
    phantom_2D = TomoP2D.Object(size, objlist)
    phantom_2D = np.asarray(phantom_2D)

    # TomoPhantom starts at a different angle compared to elsa
    # the correct solution is to offset it to the correct angle
    if arc is not 180 and not 360:
        start = 90
    else:
        # for some models (11...) offsetting by 90 degrees is bugged, for some reason the xray sources seem to start way too close to the phantom when offsetting it
        # this is why in the 180 degree or 360 degree case which we mainly use for testing we manually fix the sinogram using numpy later
        start = 0
    
    
    # Generating a list of angles that tomophantom will generate the analytical phantom from
    angles = np.linspace(start, start+arc,num_angles,dtype='float32')
    P = int(np.sqrt(2)*N_size) #detectors
    
    # Generating the analytical sinogram
    sino_an = TomoP2D.ObjectSino(size, P, angles, objlist)
    # Tomophantom reverses x and y on sinogram, we turn it back
    sino_an = np.asarray(sino_an).T

    if arc is 180:
        # manually adjusting the sinogram because of the off starting point
        # this results in the correct sinogram as elsa would generate numerically for 180
        shape = np.shape(sino_an)
        reshaped = blockshaped(sino_an, shape[0], int(shape[1] / 2))
        sino_an = np.concatenate((np.flipud(reshaped[1]), reshaped[0]), axis=1)
        
        sino_an = np.flipud(sino_an)
        sino_an = np.fliplr(sino_an)
    if arc is 360:
        shape = np.shape(sino_an)
        reshaped = blockshaped(sino_an, shape[0], int(shape[1] / 4))
        sino_an = np.concatenate((reshaped[3], reshaped[0], reshaped[1], reshaped[2]), axis=1)
        sino_an = np.fliplr(sino_an)

    if noise:
        sino_an = add_tomophantom_poisson(sino_an)

    return phantom_2D, sino_an 

def angles_compare_solvers(solvers: List[SolverTest], iterations: int, problem_size: int, show_plot: bool, model_number: int, max_angles: int, start_angles: int, step_angles: int, mean_repeats: int,noise: bool,  do_3d: bool, save_as=None):

    distances = [[] for _ in solvers]
    times = [[] for _ in solvers]
    angles = [[] for _ in solvers]

    for angl in range(start_angles, max_angles, step_angles):
        instantiated_solvers, phantom = instantiate_solvers(solvers, problem_size, model_number, angl, do_3d=do_3d, noise=noise)

        optimal_phantom = np.array(phantom)
        
        print(f'Solving for iterations {iterations} for model {model_number} and angles {angl}')

        for j, solver in enumerate(instantiated_solvers):
            durations = [-1] * mean_repeats
            mses = [-1] * mean_repeats

            for x in range(mean_repeats):

                start = time.process_time()
                reconstruction = np.array(solver[0].solve(iterations))

                durations[x] = time.process_time() - start
                mses[x] = mse(reconstruction, optimal_phantom)

            times[j].append(np.mean(durations))
            distances[j].append(np.mean(mses))
            angles[j].append(angl)
        

    print(f'Done with optimizing starting to plot now')

    import matplotlib.pyplot as plt  # local imports so that we can switch to headless mode before importing

    # Plotting times
    fig, ax = plt.subplots()
    ax.set_xlabel('MSE')
    ax.set_ylabel(f'num_angles for {180}°')
    ax.set_title(f'Mean Square Error over number of angles for {iterations} iterations, model {model_number}')
    for dist, angl, solver in zip(distances, angles, instantiated_solvers):
        ax.plot(dist, angl, label=solver[1])
    ax.legend()

    if save_as:
        name = save_as + f'comp_angles_mse_{model_number}'
        if do_3d:
            name += '_3D'
        if noise:
            name += '_noise'
        name +='.png'
        plt.savefig(name)

    # n_iterations = list(filter(lambda x: x <= max_iterations, [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 150, 200]))

    # Plotting Iterations
    fig, ax = plt.subplots()

    ax.set_xlabel('execution time [s]')
    ax.set_ylabel(f'num_angles for {180}°')

    ax.set_title(f'execution time [s] over number of angles for {iterations} iterations, model {model_number}')

    for times, angl, solver in zip(times, angles, instantiated_solvers):
        ax.plot(times, angl, label=solver[1])
    ax.legend()

    if save_as:
        name = save_as + f'comp_angles_time_{model_number}'
        if do_3d:
            name += '_3D'
        if noise:
            name += '_noise'
        name +='.png'
        plt.savefig(name)

def evaluate_solvers_angles(max_iterations: int, show_plots: bool, problem_size: int, model_number: int, mean_repeats: int ,noise: bool, do_3d: bool,plots_dir: Optional[Path] = None):
    solvers = [
        SolverTest(elsa.ABGMRES, 'ABGMRES', is_gmres=True),
        SolverTest(elsa.BAGMRES, 'BAGMRES', is_gmres=True),
        SolverTest(elsa.CG, 'CG'),
        SolverTest(elsa.FGM, 'FGM'),
        SolverTest(elsa.OGM, 'OGM'),
        SolverTest(elsa.GradientDescent, 'Gradient Descent')  # with 1 / lipschitz as step size
    ]

    angles_compare_solvers(solvers, iterations=max_iterations, problem_size=problem_size, show_plot=show_plots, model_number=model_number, start_angles=20, max_angles=420, step_angles=20, mean_repeats=mean_repeats, noise=noise, do_3d=do_3d, save_as=plots_dir if plots_dir else None)

def evaluate_solvers_angles_unmatched(max_iterations: int, show_plots: bool, problem_size: int, model_number: int, mean_repeats: int ,noise: bool, do_3d: bool,plots_dir: Optional[Path] = None):
    solvers = [
        SolverTest(elsa.ABGMRES, 'ABGMRES', is_gmres=True),
        SolverTest(elsa.BAGMRES, 'BAGMRES', is_gmres=True),
        SolverTest(elsa.ABGMRES, 'unmatched ABGMRES', is_gmres=True, is_unmatched=True),
        SolverTest(elsa.BAGMRES, 'unmatched BAGMRES', is_gmres=True, is_unmatched=True),
        SolverTest(elsa.CG, 'CG')
    ]

    angles_compare_solvers(solvers, iterations=max_iterations, problem_size=problem_size, show_plot=show_plots, model_number=model_number, start_angles=20, max_angles=420, step_angles=20, mean_repeats=mean_repeats, noise=noise, do_3d=do_3d, save_as=plots_dir if plots_dir else None)

def main():

    matplotlib.use('Agg')

    plots_dir = os.getcwd() + "/solver_experiments/"

    elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)
    elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
    elsa.logger_pyelsa_problems.setLevel(elsa.LogLevel.OFF)
    elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)

    # SheppLogan, Defrise, Phantom with Gaussians
    model_numbers = [1, 3, 14]

    # --- Repeats for Means ---
    mean_repeats = 20
    
    # --- Matched 2D ---

    do_3d=False

    problem_size = 512
    max_iterations = 50

    noise = False

    for model in model_numbers:
        evaluate_solvers_angles(max_iterations=max_iterations, show_plots=True, plots_dir=plots_dir, model_number=model, mean_repeats=mean_repeats, noise=noise,  do_3d=do_3d, problem_size=problem_size)

    # With noise
    noise = True

    for model in model_numbers:
        evaluate_solvers_angles(max_iterations=max_iterations, show_plots=True, plots_dir=plots_dir, model_number=model, mean_repeats=mean_repeats, noise=noise,  do_3d=do_3d, problem_size=problem_size)

    # --- Unmatched 2D ---

    for model in model_numbers:
        evaluate_solvers_angles_unmatched(max_iterations=max_iterations, show_plots=True, plots_dir=plots_dir, model_number=model, mean_repeats=mean_repeats, noise=noise,  do_3d=do_3d, problem_size=problem_size)

    # With noise
    noise = True

    for model in model_numbers:
        evaluate_solvers_angles_unmatched(max_iterations=max_iterations, show_plots=True, plots_dir=plots_dir, model_number=model, mean_repeats=mean_repeats, noise=noise,  do_3d=do_3d, problem_size=problem_size)


if __name__ == '__main__':
    main()