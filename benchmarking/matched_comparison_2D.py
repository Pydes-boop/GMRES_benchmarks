import numpy as np
import matplotlib.pyplot as plt
import pyelsa as elsa
import time
import os

from dataclasses import dataclass, field
@dataclass
class SolverTest:
    solver_class: elsa.Solver
    solver_name: str
    linestyle: str
    is_gmres: bool = False
    is_unmatched: bool = False

def mse(optimized: np.ndarray, original: np.ndarray) -> float:
    size = original.size
    diff = (original - optimized) ** 2
    return np.sum(diff) / size

def solve(solver: SolverTest, projector_class_matched: elsa.JosephsMethodCUDA, projector_class_unmatched: elsa.JosephsMethodCUDA, sinogram: elsa.DataContainer, times, distances, num, optimal_phantom, nmax_iter, repeats):
    durations = [-1.0] * repeats
    mses = [-1.0] * repeats

    for current in range(repeats):
        
        if solver.is_gmres:
            if solver.is_unmatched:
                solv = solver.solver_class(projector_class_unmatched, sinogram)
            else:
                solv = solver.solver_class(projector_class_matched, sinogram)
        else:
            # setup reconstruction problem
            problem = elsa.WLSProblem(projector_class_matched, sinogram)
            solv = solver.solver_class(problem)
            
        start = time.process_time()
        x = np.asarray(solv.solve(nmax_iter))
        durations[current] = time.process_time() - start
        mses[current] = mse(x, optimal_phantom)

    times[num].append(np.mean(durations))
    distances[num].append(np.amin(mses))

### --- Iteration --- ###
elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_problems.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)

### --- Solvers --- ###

ls = ['-',':','--','-.']

solvers_matched = [
        SolverTest(elsa.ABGMRES, 'ABGMRES', is_gmres=True, linestyle=ls[0%4]),
        SolverTest(elsa.BAGMRES, 'BAGMRES', is_gmres=True, linestyle=ls[1%4]),
        SolverTest(elsa.CG, 'CG', linestyle=ls[2%4]),
        SolverTest(elsa.FGM, 'FGM', linestyle=ls[3%4]),
        SolverTest(elsa.OGM, 'OGM', linestyle=ls[5%4]),
        SolverTest(elsa.GradientDescent, 'Gradient Descent', linestyle=ls[6%4])  # with 1 / lipschitz as step size
    ]

solvers_unmatched = [
    SolverTest(elsa.ABGMRES, 'matched ABGMRES', is_gmres=True, linestyle=ls[0%4]),
    SolverTest(elsa.BAGMRES, 'matched BAGMRES', is_gmres=True, linestyle=ls[1%4]),
    SolverTest(elsa.ABGMRES, 'unmatched ABGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[2%4]),
    SolverTest(elsa.BAGMRES, 'unmatched BAGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[3%4]),
    SolverTest(elsa.CG, 'CG', linestyle=ls[5%4])
]

### --- Setup --- ###
size = 500
min_iter = 1
max_iter = 20
iter_steps = 1
repeats = 20

# only have to setup these things once because everything has the same settings
# settings from how the phantoms and sinograms were generated in ../experiments/
arc = 360
num_angles = 180
volume_descriptor = elsa.VolumeDescriptor(np.asarray([size, size]))
sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, volume_descriptor, arc, size * 100, size)
# setup operator for 2d X-ray transform
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=False)
projectorUnmatched = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)

# folder path
dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms"

save_path = os.path.dirname(os.path.abspath(__file__)) + "/"

# print(dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms")

# https://pynative.com/python-list-files-in-a-directory/
# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    res.append(os.path.join(dir_path, path))

dir_path += "/"

# Getting all the important phantoms, with and without noise
phantoms = [s for s in res if "phantom_tp" in s]
phantomsNoise = [s for s in phantoms if "noise_True" in s]
phantomsNoise.sort()
phantoms = [s for s in phantoms if "noise_False" in s]
phantoms.sort()

# Getting all the important sinograms, with and without noise
sinograms = [s for s in res if "sinogram_tp" in s]
sinogramsNoise = [s for s in sinograms if "noise_True" in s]
sinogramsNoise.sort()
sinograms = [s for s in sinograms if "noise_False" in s]
sinograms.sort()

### matched solvers test ####
### no noise ###

for ph, si in zip(phantoms, sinograms):
    
    distancesM = [[] for _ in solvers_matched]
    timesM = [[] for _ in solvers_matched]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers_matched):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=sinogram, times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, model ' + name)
    for dist, times, solver in zip(distancesM, timesM, solvers_matched):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "matched_model_" + name + "_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, model ' + name)
    for dist, solver in zip(distancesM, solvers_matched):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "matched_model_" + name + "_mse_iter.png", dpi=600)

    plt.close('all')

### matched solvers test ####
### noise ###

for ph, si in zip(phantomsNoise, sinogramsNoise):

    distancesM = [[] for _ in solvers_matched]
    timesM = [[] for _ in solvers_matched]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers_matched):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=sinogram, times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, model ' + name)
    for dist, times, solver in zip(distancesM, timesM, solvers_matched):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "matched_model_" + name + "_noisy_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, model ' + name)
    for dist, solver in zip(distancesM, solvers_matched):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "matched_model_" + name + "_noisy_mse_iter.png", dpi=600)

    plt.close('all')


### unmatched solvers test ###
### no noise ###

print("testing unmtached solvers now")
for ph, si in zip(phantoms, sinograms):

    distancesU = [[] for _ in solvers_unmatched]
    timesU = [[] for _ in solvers_unmatched]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers_unmatched):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=sinogram, times=timesU, distances=distancesU, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, model ' + name)
    for dist, times, solver in zip(distancesU, timesU, solvers_unmatched):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "unmatched_model_" + name + "_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, model ' + name)
    for dist, solver in zip(distancesU, solvers_unmatched):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "unmatched_model_" + name + "_mse_iter.png", dpi=600)

    plt.close('all')

### unmatched solvers test ###
### noise ###

print("testing unmtached solvers now")
for ph, si in zip(phantomsNoise, sinogramsNoise):

    distancesU = [[] for _ in solvers_unmatched]
    timesU = [[] for _ in solvers_unmatched]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers_unmatched):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=sinogram, times=timesU, distances=distancesU, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, model ' + name)
    for dist, times, solver in zip(distancesU, timesU, solvers_unmatched):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "unmatched_model_" + name + "_noisy_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, model ' + name)
    for dist, solver in zip(distancesU, solvers_unmatched):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "unmatched_model_" + name + "_noisy_mse_iter.png", dpi=600)

    plt.close('all')