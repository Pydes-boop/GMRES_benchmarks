import time
import numpy as np
import pyelsa as elsa
from scipy.fft import fft, ifft
import os

### Setting up Filter to apply to B ###
def ramp_filter(size):
    n = np.concatenate(
        (
            # increasing range from 1 to size/2, and again down to 1, step size 2
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # See "Principles of Computerized Tomographic Imaging" by Avinash C. Kak and Malcolm Slaney,
    # Chap 3. Equation 61, for a detailed description of these steps
    return 2 * np.real(fft(f))[:, np.newaxis]

def filter_sinogram(sinogram, sino_descriptor = None):
    np_sinogram = np.array(sinogram)
    
    sinogram_shape = np_sinogram.shape[0]

    # Add padding
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * sinogram_shape))))
    pad_width = ((0, projection_size_padded - sinogram_shape), (0, 0))
    padded_sinogram = np.pad(np_sinogram, pad_width, mode="constant", constant_values=0)

    # Ramp filter
    fourier_filter = ramp_filter(projection_size_padded)

    projection = fft(padded_sinogram, axis=0)  * fourier_filter
    filtered_sinogram = np.real(ifft(projection, axis=0)[:sinogram_shape, :])

    if not isinstance(sinogram,(list,np.ndarray, np.matrix)):
        # Go back into elsa space
        filtered_sinogram = elsa.DataContainer(filtered_sinogram, sino_descriptor)
    
    return filtered_sinogram

### log function ###
def log(text, logging):
    if logging:
        print(text)

### apply with or without filter ###
def apply(A, x, filter=False, sino_descriptor=None):
    if filter == False:
        # standard apply case
        if isinstance(A,(list,np.ndarray, np.matrix)):
            return np.asarray(np.dot(A, x)).reshape(-1)
        else:
            dc = elsa.DataContainer(x)
            return np.asarray(A.apply(dc))
    else:
        # if we want to filter we first apply the filter sinogram function and then do the standard apply case
        if isinstance(A,(list,np.ndarray, np.matrix)):
            return np.asarray(np.dot(A, filter_sinogram(x)))
        else:
            if sino_descriptor is not None:
                ret = apply(A, filter_sinogram(elsa.DataContainer(x), sino_descriptor)).reshape(-1)
            else:
                ret = apply(A, filter_sinogram(elsa.DataContainer(x))).reshape(-1)
        return ret

### BAGMRES using filter or not ###
def BAGMRES(A, B, b, x0, nmax_iter, epsilon = None, logging=False, filter=False, sino_descriptor=None):

    log("Starting preperations... ", logging=logging)

    ti = time.process_time()

    # Saving this shape as in tomographic reconstruction cases with elsa this is not a vector so we have to translate the shape
    x0_shape = np.shape(np.asarray(x0))

    # r0 = Bb - BAx
    r0 = apply(B, b, sino_descriptor=sino_descriptor, filter=filter).reshape(-1) - apply(B, apply(A, x0), sino_descriptor=sino_descriptor, filter=filter).reshape(-1)

    h = np.zeros((nmax_iter + 1, nmax_iter))
    w = [np.zeros(len(r0))] * nmax_iter
    e = np.zeros(nmax_iter + 1)
    y = [0] * nmax_iter

    e[0] = np.linalg.norm(r0)

    w[0] = r0 / np.linalg.norm(r0)

    elapsed_time = time.perf_counter() - ti
    log("Preperations done, took: " + str(elapsed_time) + "s", logging=logging)

    # --- 2. Iterate ---
    for k in range(nmax_iter):

        log("Iteration | residual | elapsed time | total time", logging=logging)

        t = time.perf_counter()

        # q = BAw_k
        q = apply(B, apply(A, np.reshape(w[k], x0_shape)), sino_descriptor=sino_descriptor, filter=filter).reshape(-1)

        for i in range(k+1):
            h[i, k] = apply(q.T, w[i])
            q = q - h[i, k] * w[i]
        
        h[k+1, k] = np.linalg.norm(q)

        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            w[k+1] = q/h[k+1, k]

        # Solving minimization problem using numpy leastsquares
        y = np.linalg.lstsq(h, e, rcond=None)[0]

        # transforming list of vectors to a matrix
        w_copy = np.reshape(np.asarray(w), (nmax_iter, len(w[0]))).T

        # applying estimated guess from our generated krylov subspace to our initial guess x0
        x = np.asarray(x0) + np.reshape(apply(w_copy, y), x0_shape)

        # calculating a residual
        r = np.asarray(b).reshape(-1) - (apply(A, np.reshape(x, x0_shape))).reshape(-1)

        elapsed_time = time.process_time() - t
        ti += elapsed_time
        log(str(k) + " | " + str(np.linalg.norm(r)) + " | " + str(elapsed_time)[:6] + " | " + str(ti)[:6], logging=logging)

        if epsilon is not None:
            if np.linalg.norm(np.asarray(r)) <= epsilon:
                print("Reached Convergence at: " + str(k) + "/" + str(nmax_iter))
                break

    return x

### BAGMRES using filter or not ###
def ABGMRES(A, B, b, x0, nmax_iter, epsilon = None, logging=False, filter=False, sino_descriptor=None):

    log("Starting preperations... ", logging=logging)

    ti = time.process_time()

    # Saving this shape as in tomographic reconstruction cases with elsa this is not a vector so we have to translate the shape
    x0_shape = np.shape(np.asarray(x0))
    b_shape = np.shape(np.asarray(b))

    # r0 = b - Ax
    r0 = np.asarray(b).reshape(-1) - apply(A, x0).reshape(-1)

    h = np.zeros((nmax_iter + 1, nmax_iter))
    w = [np.zeros(len(r0))] * nmax_iter
    e = np.zeros(nmax_iter + 1)
    y = [0] * nmax_iter

    e[0] = np.linalg.norm(r0)

    w[0] = r0 / np.linalg.norm(r0)

    elapsed_time = time.perf_counter() - ti
    log("Preperations done, took: " + str(elapsed_time) + "s", logging=logging)

    # --- 2. Iterate ---
    for k in range(nmax_iter):

        log("Iteration | residual | elapsed time | total time", logging=logging)

        t = time.perf_counter()

        # q = ABw_k
        q = np.asarray(apply(A, np.reshape(apply(B, np.reshape(w[k], b_shape), sino_descriptor=sino_descriptor, filter=filter), x0_shape))).reshape(-1)

        for i in range(k+1):
            h[i, k] = apply(q.T, w[i])
            q = q - h[i, k] * w[i]
        
        h[k+1, k] = np.linalg.norm(q)

        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            w[k+1] = q/h[k+1, k]

        # Solving minimization problem using numpy leastsquares
        y = np.linalg.lstsq(h, e, rcond=None)[0]

        # transforming list of vectors to a matrix
        w_copy = np.reshape(np.asarray(w), (nmax_iter, len(w[0]))).T

        # applying estimated guess from our generated krylov subspace to our initial guess x0
        x = np.asarray(x0) + np.reshape(apply(B, np.reshape(np.asarray(np.dot(w_copy, y)), b_shape), filter=filter, sino_descriptor=sino_descriptor), x0_shape)

        # calculating a residual
        r = np.asarray(b).reshape(-1) - np.asarray(apply(A, np.reshape(x, x0_shape))).reshape(-1)

        elapsed_time = time.process_time() - t
        ti += elapsed_time
        log(str(k) + " | " + str(np.linalg.norm(r)) + " | " + str(elapsed_time)[:6] + " | " + str(ti)[:6], logging=logging)

        if epsilon is not None:
            if np.linalg.norm(np.asarray(r)) <= epsilon:
                print("Reached Convergence at: " + str(k) + "/" + str(nmax_iter))
                break

    return x

### testing setup ###

from dataclasses import dataclass, field
from typing import Callable
@dataclass
class SolverTest:
    solver_class: Callable
    solver_name: str
    linestyle: str
    is_gmres: bool = False
    is_unmatched: bool = False
    filter: bool = False

def mse(optimized: np.ndarray, original: np.ndarray) -> float:
    size = original.size
    diff = (original - optimized) ** 2
    return np.sum(diff) / size

### --- Setup --- ###

# todo change iter depending on matched_comparison
min_iter = 1
max_iter = 20
iter_steps = 1
repeats = 10

problem_size = 500
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

solvers = [
    SolverTest(ABGMRES, 'unmatched ABGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[0]),
    SolverTest(BAGMRES, 'unmatched BAGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[1]),
    SolverTest(ABGMRES, 'unmatched ABGMRES with filter', is_gmres=True, is_unmatched=True, linestyle=ls[2], filter=True),
    SolverTest(BAGMRES, 'unmatched BAGMRES with filter', is_gmres=True, is_unmatched=True, linestyle=ls[3], filter=True),
]

### --- Iteration --- ###
elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_problems.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)

### --- Setup Elsa --- ###
size = np.array([problem_size, problem_size])
volume_descriptor = elsa.VolumeDescriptor(size)
sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(num_angles, volume_descriptor, arc, size[0] * 100, size[0])

# setup operator for 2d X-ray transform
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)

phantom = elsa.phantoms.modifiedSheppLogan(size)
x0 = elsa.DataContainer(np.zeros_like(np.asarray(phantom)), phantom.getDataDescriptor())

# folder path
dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = dir_path.replace("benchmarking", "experiments") + "/phantoms_sinograms"

save_path = os.path.dirname(os.path.abspath(__file__)) + "/filter_comparison/"

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

def solve(solver: SolverTest, projector_class_matched: elsa.JosephsMethodCUDA, projector_class_unmatched: elsa.JosephsMethodCUDA, sinogram: elsa.DataContainer, times, distances, num, optimal_phantom, nmax_iter, repeats, sino_descriptor, x0, filter=False):
    durations = [-1.0] * repeats
    mses = [-1.0] * repeats

    for current in range(repeats):   
        print("Applying " + solver.solver_name + " for " + str(nmax_iter)) 
        start = time.process_time()
        x = np.asarray(solver.solver_class(projector_class_unmatched, elsa.adjoint(projector_class_unmatched), sinogram, x0, nmax_iter, sino_descriptor=sino_descriptor, filter=solver.filter))
        durations[current] = time.process_time() - start
        mses[current] = mse(x, optimal_phantom)

    times[num].append(np.mean(durations))
    distances[num].append(np.mean(mses))

### Matched Case ###

### no noise ###

for ph, si in zip(phantoms, sinograms):
    
    distancesM = [[] for _ in solvers]
    timesM = [[] for _ in solvers]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projector, sinogram=sinogram, times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats, sino_descriptor=sino_descriptor, x0=x0, filter=solver.filter)


    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    import matplotlib.pyplot as plt

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]

    # Plotting times
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, GMRES with FBP')
    for dist, times, solver in zip(distancesM, timesM, solvers):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "fbp_model_" + str(name) +"_2D_noise_False_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, GMRES with FBP')
    for dist, solver in zip(distancesM, solvers):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "fbp_model_" + str(name) + "_2D_noise_False_mse_iter.png", dpi=600)

    plt.close('all')

for ph, si in zip(phantomsNoise, sinogramsNoise):
    
    distancesM = [[] for _ in solvers]
    timesM = [[] for _ in solvers]

    print("experimenting with model " + str(ph))
    phantom = np.load(ph)
    sinogram = elsa.DataContainer(np.load(si))

    for iter in range(min_iter,max_iter,iter_steps):
        for j, solver in enumerate(solvers):
            solve(solver=solver, projector_class_matched=projector, projector_class_unmatched=projector, sinogram=sinogram, times=timesM, distances=distancesM, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats, sino_descriptor=sino_descriptor, x0=x0, filter=solver.filter)


    print(f'Done with optimizing matched solver for current sino, starting to plot now')

    import matplotlib.pyplot as plt
    import os

    # Plotting times
    name = ph.split("phantom_tp_model_",1)[1]
    name = name.split("_noise",1)[0]

    # Plotting times
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time, GMRES with FBP')
    for dist, times, solver in zip(distancesM, timesM, solvers):
        ax.plot(times, dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "fbp_model_" + str(name) +"_2D_noise_True_mse_times.png", dpi=600)

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations, GMRES with FBP')
    for dist, solver in zip(distancesM, solvers):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), dist, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(save_path + "fbp_model_" + str(name) + "_2D_noise_True_mse_iter.png", dpi=600)

    plt.close('all')