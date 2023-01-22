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
max_iter = 30
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
    (0, (3, 4, 1, 4, 1, 4))
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
    distances[num].append(np.mean(mses))

### load sinogram and phantom ###

# folder path
dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms"

save_path = os.path.dirname(os.path.abspath(__file__)) + "/matched_comparison_3D/"

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
print(sinogramNoise[0])
print(sinogram)
sinogramNoise = np.load(sinogramNoise[0])
sinogram = np.load(sinogram[0])

### Matched Case ###

### no noise ###

distancesM = [[] for _ in solvers_matched]
timesM = [[] for _ in solvers_matched]

print("experimenting with model 3D for matched case")

for iter in range(min_iter,max_iter,iter_steps):
    for j, solver in enumerate(solvers_matched):
        solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=elsa.DataContainer(sinogram), times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

print(f'Done with optimizing matched solver for current sino, starting to plot now')

import matplotlib.pyplot as plt

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('execution time [s]')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over execution time, 3D model')
for dist, times, solver in zip(distancesM, timesM, solvers_matched):
    ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "matched_model_3D_mse_times.png", dpi=600)

# Plotting Iterations
fig, ax = plt.subplots()
ax.set_xlabel('number of iterations')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of iterations, 3D model')
for dist, solver in zip(distancesM, solvers_matched):
    ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "matched_model_3D_mse_iter.png", dpi=600)

plt.close('all')

### Matched Case ###

### noise ###

distancesM = [[] for _ in solvers_matched]
timesM = [[] for _ in solvers_matched]

print("experimenting with model 3D for matched case")

for iter in range(min_iter,max_iter,iter_steps):
    for j, solver in enumerate(solvers_matched):
        solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=elsa.DataContainer(sinogramNoise), times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

print(f'Done with optimizing matched solver for current sino, starting to plot now')

import matplotlib.pyplot as plt

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('execution time [s]')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over execution time, 3D model')
for dist, times, solver in zip(distancesM, timesM, solvers_matched):
    ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "matched_model_3D_noisy_mse_times.png", dpi=600)

# Plotting Iterations
fig, ax = plt.subplots()
ax.set_xlabel('number of iterations')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of iterations, 3D model')
for dist, solver in zip(distancesM, solvers_matched):
    ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "matched_model_3D_noisy_mse_iter.png", dpi=600)

plt.close('all')

### Unmatched Case ###

### no noise ###

distancesU = [[] for _ in solvers_unmatched]
timesU = [[] for _ in solvers_unmatched]

print("experimenting with model 3D for matched case")

for iter in range(min_iter,max_iter,iter_steps):
    for j, solver in enumerate(solvers_unmatched):
        solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=elsa.DataContainer(sinogram), times=timesU, distances=distancesU, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

print(f'Done with optimizing matched solver for current sino, starting to plot now')

import matplotlib.pyplot as plt

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('execution time [s]')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over execution time, 3D model')
for dist, times, solver in zip(distancesM, timesM, solvers_unmatched):
    ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "unmatched_model_3D_mse_times.png", dpi=600)

# Plotting Iterations
fig, ax = plt.subplots()
ax.set_xlabel('number of iterations')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of iterations, 3D model')
for dist, solver in zip(distancesM, solvers_unmatched):
    ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "unmatched_model_3D_mse_iter.png", dpi=600)

### Unmatched Case ###

### no noise ###

distancesU = [[] for _ in solvers_unmatched]
timesU = [[] for _ in solvers_unmatched]

print("experimenting with model 3D for matched case")

for iter in range(min_iter,max_iter,iter_steps):
    for j, solver in enumerate(solvers_unmatched):
        solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=elsa.DataContainer(sinogramNoise), times=timesU, distances=distancesU, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

print(f'Done with optimizing matched solver for current sino, starting to plot now')

import matplotlib.pyplot as plt

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('execution time [s]')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over execution time, 3D model')
for dist, times, solver in zip(distancesM, timesM, solvers_unmatched):
    ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "unmatched_model_3D_noisy_mse_times.png", dpi=600)

# Plotting Iterations
fig, ax = plt.subplots()
ax.set_xlabel('number of iterations')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of iterations, 3D model')
for dist, solver in zip(distancesM, solvers_unmatched):
    ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "unmatched_model_3D_noisy_mse_iter.png", dpi=600)