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
nmax_iter = 20
size = 500
min_angles = 20
max_angles = 501
angles_steps = 2
repeats = 10

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

# colormap, colorblindsafe from https://personal.sron.nl/~pault/#sec:qualitative
# didnt look good
# cm = [
#     '#66CCEE',
#     '#228833',
#     '#CCBB44',
#     '#EE6677',
#     '#AA3377',
#     '#4477AA'
# ]

solvers = [
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

phantom = elsa.phantoms.modifiedSheppLogan(np.array([size, size]))
optimal_phantom = np.array(phantom)

distances = [[] for _ in solvers]
times = [[] for _ in solvers]
angles = list(range(min_angles,max_angles,angles_steps))

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


for num_angles in range(min_angles,max_angles,angles_steps):

    print("current angle: ", num_angles)
    
    volume_descriptor = phantom.getDataDescriptor()

    # settings
    arc = 180

    # generate circular trajectory
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, phantom.getDataDescriptor(), arc, size * 100, size)

    # setup operator for 2d X-ray transform
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=False)
    projectorUnmatched = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)

    # simulate the sinogram
    sinogram = projector.apply(phantom)

    for j, solver in enumerate(solvers):
        solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projectorUnmatched, sinogram=sinogram, times=times, distances=distances, num=j, optimal_phantom=optimal_phantom, nmax_iter=nmax_iter, repeats=repeats)

print(f'Done with optimizing starting to plot now')

import matplotlib.pyplot as plt  # local imports so that we can switch to headless mode before importing

dir_path = os.path.dirname(os.path.abspath(__file__)) + "/angles_comparison/"

# Plotting MSE
fig, ax = plt.subplots()
ax.set_xlabel('number of angles')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of angles')
for dist, solver in zip(distances, solvers):
    ax.plot(angles, dist, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(dir_path + "mse_num_angles.png", dpi=600)

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('number of angles')
ax.set_ylabel('execution time [s]')
ax.set_title(f'execution time over number of angles')
for times, solver in zip(times, solvers):
    ax.plot(angles, times, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(dir_path + "times_num_angles.png", dpi=600)