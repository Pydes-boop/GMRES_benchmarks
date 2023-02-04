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
min_angles = 10
max_angles = 501
angles_steps = 10
repeats = 20

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
    x = np.asarray(solv.solve(nmax_iter))
    times[num].append(time.process_time() - start)
    distances[num].append(mse(x, optimal_phantom))

def average(list, solvers):
    l = [[] for _ in solvers]
    for elem in list:
        for i in range(len(elem)):
            l[i].append(elem[i])

    ret = [[] for _ in solvers]
    mins = [[] for _ in solvers]
    maxs = [[] for _ in solvers]
    for i in range(len(l)):
        ret[i] = np.average(l[i], axis=0)
        mins[i] = np.amin(l[i], axis=0)
        maxs[i] = np.amax(l[i], axis=0)

    return ret, mins, maxs

angles = list(range(min_angles,max_angles,angles_steps))

distanceRep = []
timesRep = []

for i in range(repeats):

    print("Solving angles for repeat: " + str(i))

    distances = [[] for _ in solvers]
    times = [[] for _ in solvers]

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

    distanceRep.append(distances)
    timesRep.append(times)

# average data that can be plotted
dist, distmin, distmax = average(distanceRep, solvers)
tim, timmin, timmax = average(timesRep, solvers)

print(f'Done with optimizing starting to plot now')

import matplotlib.pyplot as plt  # local imports so that we can switch to headless mode before importing

save_path = os.path.dirname(os.path.abspath(__file__)) + "/angles_comparison/"

# create new folder for runtime so pictures dont overwrite each other
timestr = time.strftime("%d%m%Y-%H%M%S")
save_path = save_path + "/" + timestr + "/"
os.mkdir(save_path)

# Plotting MSE
fig, ax = plt.subplots()
ax.set_xlabel('number of angles')
ax.set_ylabel('MSE')
ax.set_title(f'Mean Square Error over number of angles')
for d, solver in zip(dist, solvers):
    ax.plot(angles, d, label=solver.solver_name, linestyle=solver.linestyle)
ax.legend()

plt.savefig(save_path + "mse_num_angles.png", dpi=1200, bbox_inches='tight')

# Plotting times
fig, ax = plt.subplots()
ax.set_xlabel('number of angles')
ax.set_ylabel('execution time [s]')
ax.set_title(f'execution time over number of angles')
for t, solver, mine, maxe in zip(tim, solvers, timmin, timmax):
    ax.plot(angles, t, label=solver.solver_name, linestyle=solver.linestyle)
    plt.fill_between(angles, mine, maxe, alpha=0.2)
ax.legend()

plt.savefig(save_path + "times_num_angles.png", dpi=1200, bbox_inches='tight')