import numpy as np
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

### --- Setup --- ###

# todo change iter depending on matched_comparison
min_iter = 1
max_iter = 51
iter_steps = 1
repeats = 10

problem_size = 64
arc=360
num_angles=180

# specific linestyles, otherwise its hard to differentiate
# stronger lines, like full line or -- are painted first
ls = [
    '-',
    '--',
    '-.',
    ':',
    (0, (1, 2, 1, 2, 3, 2, 3, 2)),
    (0, (3, 2, 1, 2, 1, 2, 1, 2))
]

solvers_matched = [
        SolverTest(elsa.ABGMRES, 'ABGMRES', is_gmres=True, linestyle=ls[0]),
        SolverTest(elsa.BAGMRES, 'BAGMRES', is_gmres=True, linestyle=ls[1]),
        SolverTest(elsa.CG, 'CG', linestyle=ls[2]),
        SolverTest(elsa.FGM, 'FGM', linestyle=ls[3]),
        SolverTest(elsa.OGM, 'OGM', linestyle=ls[4]),
        SolverTest(elsa.GradientDescent, 'Gradient Descent', linestyle=ls[5])  # with 1 / lipschitz as step size
    ]

solvers_unmatched = [
    SolverTest(elsa.ABGMRES, 'matched ABGMRES', is_gmres=True, linestyle=ls[0]),
    SolverTest(elsa.BAGMRES, 'matched BAGMRES', is_gmres=True, linestyle=ls[1]),
    SolverTest(elsa.ABGMRES, 'unmatched ABGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[2]),
    SolverTest(elsa.BAGMRES, 'unmatched BAGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[3]),
    SolverTest(elsa.CG, 'CG', linestyle=ls[4])
]

### --- Iteration --- ###
elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_problems.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)

### --- Setup Elsa --- ###
size = np.array([problem_size, problem_size, problem_size])
volume_descriptor = elsa.VolumeDescriptor(size)
sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(num_angles, volume_descriptor, arc, size[0] * 100, size[0])

# setup operator for 2d X-ray transform
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=False)
projectorUnmatched = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)

### load sinogram and phantom ###

# folder path
dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms"

save_path = os.path.dirname(os.path.abspath(__file__)) + "/matched_comparison_3D/"

# create new folder for runtime so pictures dont overwrite each other
timestr = time.strftime("%d%m%Y-%H%M%S")
save_path = save_path + "/" + timestr + "/"
os.mkdir(save_path)

# print(dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms")

# https://pynative.com/python-list-files-in-a-directory/
# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    res.append(os.path.join(dir_path, path))

dir_path += "/"

# Getting all the important phantoms, with and without noise
phantom = [s for s in res if "phantom_elsa_3D" in s]
phantomNoise = [s for s in phantom if "noise_True" in s]
phantom = [s for s in phantom if "noise_False" in s]
phantom = np.load(phantom[0])

# Getting all the important sinograms, with and without noise
sinogram = [s for s in res if "sinogram_elsa_3D" in s]
sinogramNoise = [s for s in sinogram if "noise_True" in s]
sinogram = [s for s in sinogram if "noise_False" in s]
sinogramNoise = np.load(sinogramNoise[0])
sinogram = np.load(sinogram[0])

import matplotlib.pyplot as plt

def solve(solver: SolverTest, projector_class_matched: elsa.JosephsMethodCUDA, projector_class_unmatched: elsa.JosephsMethodCUDA, sinogram: elsa.DataContainer, times, distances, num, optimal_phantom, nmax_iter, repeats):
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
    x = solv.solve(nmax_iter)
    finish = time.process_time() - start
    x = np.asarray(x)
    times[num].append(finish)
    distances[num].append(mse(x, optimal_phantom))

def average(list, solvers):
    l = [[] for _ in solvers]
    for elem in list:
        for i in range(len(elem)):
            l[i].append(elem[i])

    ret = [[] for _ in solvers]
    for i in range(len(l)):
        ret[i] = np.average(l[i], axis=0)
    
    return ret

def test(phantom, sinogram, solvers, experiment: str):

    distanceRep = []
    timesRep = []

    for i in range(repeats):
        distances = [[] for _ in solvers]
        times = [[] for _ in solvers]

        print("experimenting with model 3D for matched case")

        for iter in range(min_iter,max_iter,iter_steps):
            for j, solver in enumerate(solvers):
                solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=elsa.DataContainer(sinogram), times=times, distances=distances, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

        distanceRep.append(distances)
        timesRep.append(times)
    
    # average data that can be plotted
    dist = average(distanceRep, solvers)
    tim = average(timesRep, solvers)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    # Plotting times
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, 3D model')
    for d, t, solver in zip(dist, tim, solvers):
        ax.plot(t, d, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + experiment +"_model_3D_mse_times.pdf", bbox_inches='tight')

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, 3D model')
    for d, solver in zip(dist, solvers):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), d, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + experiment + "_model_3D_mse_iter.pdf", bbox_inches='tight')

    plt.close('all')

### Matched Case ###
### no noise ###

test(phantom, sinogram, solvers_matched, "matched")

### Matched Case ###
### noise ###

test(phantom, sinogramNoise, solvers_matched, "matched_noise")

### Unmatched Case ###
### no noise ###

test(phantom, sinogram, solvers_unmatched, "unmatched")

### Unmatched Case ###
### noise ###

test(phantom, sinogramNoise, solvers_unmatched, "unmatched_noise")