import argparse
from pathlib import Path

import pyelsa as elsa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import scipy.io as spio
import os

from scipy.fft import fft, ifft

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

def filter_sinogram(sinogram):
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

    filtered_sinogram = elsa.DataContainer(filtered_sinogram, sinogram.getDataDescriptor())
    
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
                ret = apply(A, filter_sinogram(elsa.DataContainer(x))).reshape(-1)
            else:
                ret = apply(A, filter_sinogram(elsa.DataContainer(x))).reshape(-1)
        return ret

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


def loadmat(filename):
    """
    Credit to: https://stackoverflow.com/a/8832212
    """

    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            else:
                d[strg] = elem
        return d

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def apply_BHC(data):
    # these coefficients are generated from the full disk data
    coefficients = np.array([0.00130522, 0.9995882, -0.01443113, 0.07282656])
    corrected_data = np.polynomial.polynomial.polyval(np.asarray(data), coefficients)
    return elsa.DataContainer(corrected_data, data.getDataDescriptor())


def load_htc2022data(filename, dataset_name="CtDataFull", groundTruth=False):
    # read in matlab file
    mat = loadmat(filename)
    params = mat[dataset_name]["parameters"]

    # read important parameters
    ds2c = params["distanceSourceOrigin"]
    ds2d = params["distanceSourceDetector"]
    dc2d = ds2d - ds2c

    detpixel_spacing = params["pixelSizePost"]
    num_detpixel = params["numDetectorsPost"]
    angles = params["angles"]

    # Rought approximation of a volume size
    vol_npixels = int(num_detpixel / np.sqrt(2))
    vol_spacing = detpixel_spacing

    # Description of the desired volume
    volume_descriptor = elsa.VolumeDescriptor([vol_npixels] * 2, [vol_spacing] * 2)

    # read data
    scan_sinogram = mat[dataset_name]["sinogram"].astype("float32")
    
    sino_data = scan_sinogram.transpose(1, 0)
    sino_data = np.flip(sino_data, axis=1)

    print(len(angles))

    if groundTruth:
        sino_descriptor = elsa.CircleTrajectoryGenerator.fromAngularIncrement(
            721,
            volume_descriptor,
            0.5,
            ds2c,
            dc2d,
            [0],
            [0, 0],
            [num_detpixel] * 2,
            [detpixel_spacing] * 2,
        )
    else:    
        # flip it
        sino_data = sino_data.T
        # slicing every second projection away so we have 360 positions over 360 angles
        sino_data = sino_data[0::2]
        # halfing the array so we hav 181 positions over 180 angles
        sino_data = sino_data[0:181]
        # slicing every second projection away so we have 90 positions over 180 angles
        sino_data = sino_data[0::2]
        # flip it back
        sino_data = sino_data.T

        # not actually correct but do we care because sizes are correct?
        sino_descriptor = elsa.CircleTrajectoryGenerator.fromAngularIncrement(
            91,
            volume_descriptor,
            2,
            ds2c,
            dc2d,
            [0],
            [0, 0],
            [num_detpixel] * 2,
            [detpixel_spacing] * 2,
        )


    # sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
    #     angles,
    #     volume_descriptor,
    #     ds2c,
    #     dc2d,
    #     [0],
    #     [0, 0],
    #     [num_detpixel] * 2,
    #     [detpixel_spacing] * 2,
    # )

    print(np.shape(sino_data))
    dc = elsa.DataContainer(sino_data, sino_descriptor)
    print(np.shape(np.asarray(dc)))

    return (
        dc,
        volume_descriptor,
    )

def reconstruct(projector, sinogram, iter, solver: elsa.Solver, gmres=True):
    if gmres:
        solv = solver(projector, sinogram)
        rec = solv.solve(iter)
    else:
        problem = elsa.WLSProblem(projector,sinogram)
        solv = solver(problem)
        rec = solv.solve(iter)
    return np.asarray(rec)

def mse(optimized: np.ndarray, original: np.ndarray) -> float:
    size = original.size
    diff = (original - optimized) ** 2
    return np.sum(diff) / size

from dataclasses import dataclass
import time
from typing import Callable
@dataclass
class SolverTest:
    solver_class: elsa.Solver
    solver_callable: Callable
    solver_name: str
    linestyle: str
    is_gmres: bool = False
    is_unmatched: bool = False
    is_callable: bool = False

def solve(solver: SolverTest, projector_class_matched: elsa.JosephsMethodCUDA, projector_class_unmatched: elsa.JosephsMethodCUDA, sinogram: elsa.DataContainer, times, distances, num, optimal_phantom, nmax_iter, repeats):

    if solver.is_gmres:
        if solver.is_unmatched:
            solv = solver.solver_class(projector_class_unmatched, sinogram)
        else:
            solv = solver.solver_class(projector_class_matched, sinogram)
    elif not solver.is_callable:
        # setup reconstruction problem
        problem = elsa.WLSProblem(projector_class_matched, sinogram)
        solv = solver.solver_class(problem)
    
    if solver.is_callable:
        start = time.process_time()
        x = solver.solver_callable(projector_class_unmatched, elsa.adjoint(projector_class_unmatched), sinogram, x0, nmax_iter, sino_descriptor=sinogram.getDataDescriptor(), filter=True)
        finish = time.process_time() - start
    else:
        start = time.process_time()
        x = solv.solve(nmax_iter)
        finish = time.process_time() - start

    x = np.asarray(x)
    x = np.interp(x, (x.min(), x.max()), (0, 1))
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

def add_image(x, y, image, min, max, cmap, name):

    plt.figure()
    plt.axis("off")
    plt.imshow(image, cmap=cmap, vmax=max, vmin=min)
    rect = patches.Rectangle((99, 99), 99, 99, linewidth=2, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.savefig(name + ".png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    plt.figure()
    plt.axis("off")
    plt.imshow(image[99:197, 99:198], cmap=cmap, vmax=max, vmin=min)
    plt.savefig(name + "_zoomed.png", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--indir", type=Path, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-f", "--file", type=Path, default="htc2022_tc_full.mat")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--reg", type=float, default=1.0)

    args = parser.parse_args()

    inpath = Path(args.indir)

    print(
        f"Running reconstruction for {args.file} with {args.iters} iterations and {args.reg}"
    )

    if not inpath.is_dir():
        raise RuntimeError("Input path is not a directory")

    file = inpath / args.file

    if "limited" in str(args.file):
        dataset = "CtDataLimited"
    else:
        dataset = "CtDataFull"

    # generate groundTruth with FBP

    sinogram, volume_desc = load_htc2022data(file, dataset_name=dataset, groundTruth=True)
    sinogram = apply_BHC(sinogram)
    sinogram = filter_sinogram(sinogram)
    projector = elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True)
    groundTruth = np.asarray(projector.applyAdjoint(sinogram))
    phantom = np.interp(groundTruth, (groundTruth.min(), groundTruth.max()), (0, 1))

    x0 = elsa.DataContainer(np.zeros_like(np.asarray(phantom)))

    del projector

    elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)

    sinogram, volume_desc = load_htc2022data(file, dataset_name=dataset, groundTruth=False)
    sinogram = apply_BHC(sinogram)

    iter = 10

    projector = elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True)
    projectorSlow = elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=False)

    rec_slow = reconstruct(projectorSlow, sinogram, iter, elsa.ABGMRES, gmres=True)
    rec_slow = np.interp(rec_slow, (rec_slow.min(), rec_slow.max()), (0, 1))

    rec_fast = reconstruct(projector, sinogram, iter, elsa.ABGMRES, gmres=True)
    rec_fast = np.interp(rec_fast, (rec_fast.min(), rec_fast.max()), (0, 1))

    rec_cg = reconstruct(projector, sinogram, iter, elsa.CG, gmres=False)
    rec_cg = np.interp(rec_cg, (rec_cg.min(), rec_cg.max()), (0, 1))

    rec_filter = ABGMRES(projector, elsa.adjoint(projector), sinogram, x0, iter, sino_descriptor=sinogram.getDataDescriptor(), filter=True)
    rec_filter = np.interp(rec_filter, (rec_filter.min(), rec_filter.max()), (0, 1))

    fntsize = 8

    figure, ax = plt.subplots(nrows=5, ncols=2, figsize=(6.9, 14.7))

    cmap = "gray"
    print(np.shape(phantom))
    add_image(0, 0, phantom, 0.0, 1.0, cmap, f"{args.indir}/results/Results/plastic_disc_true")
    add_image(1, 0, rec_slow, 0.0, 1.0, cmap, f"{args.indir}/results/Results/plastic_disc_slow")
    add_image(2, 0, rec_fast, 0.0, 1.0, cmap, f"{args.indir}/results/Results/plastic_disc_fast")
    add_image(3, 0, rec_filter, 0.0, 1.0, cmap, f"{args.indir}/results/Results/plastic_disc_filter")
    add_image(4, 0, rec_cg, 0.0, 1.0, cmap, f"{args.indir}/results/Results/plastic_disc_cg")

    print("DONE!")

    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(np.asarray(sinogram), cmap="gray")
    # ax.axis("equal")
    # ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(
        # f"results/recon_{file.stem}_iters-{str(niters).zfill(3)}_reg-{reg:05.2f}.png",
        f"{args.indir}/results/sinogram-{file.stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    # generatedifferenceplot(rec_fast, groundTruth, "/results/difference_gmres_fast.png")

    # generatedifferenceplot(rec_slow, groundTruth, "/results/difference_gmres_slow.png")

    # generatedifferenceplot(rec_cg, groundTruth, "/results/difference_cg.png")

    # generatedifferenceplot(rec_filter, groundTruth, "/results/difference_filter.png")

    # generatedifferenceplot(rec_fast, rec_cg, "/results/difference_gmres_cg.png")

    # generatedifferenceplot(rec_fast, rec_filter, "/results/difference_gmres_filter.png")

    ls = [
    '-',
    '--',
    '-.',
    ':',
    (0, (1, 2, 1, 2, 3, 2, 3, 2)),
    (0, (3, 2, 1, 2, 1, 2, 1, 2))
    ]

    so = [
    SolverTest(elsa.ABGMRES, None,'matched ABGMRES', is_gmres=True, linestyle=ls[0]),
    SolverTest(elsa.ABGMRES, None,'unmatched ABGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[1]),
    SolverTest(elsa.CG, None,'CG', linestyle=ls[2], is_gmres=False)
    ]

    solvers = [
    SolverTest(elsa.ABGMRES, None,'matched ABGMRES', is_gmres=True, linestyle=ls[0]),
    SolverTest(elsa.ABGMRES, None,'unmatched ABGMRES', is_gmres=True, is_unmatched=True, linestyle=ls[1]),
    SolverTest(elsa.CG, None,'CG', linestyle=ls[2], is_gmres=False),
    SolverTest(elsa.CG, None,'unmatched ABGMRES with filter', linestyle=ls[3], is_gmres=False)
    ]

    min_iter = 1
    max_iter = 31
    iter_steps = 1
    repeats = 20

    distanceRep = []
    timesRep = []

    for i in range(repeats):
        print("Repeat: "+ str(i))

        distances = [[] for _ in solvers]
        times = [[] for _ in solvers]

        for iter in range(min_iter,max_iter,iter_steps):
            # print("Iteration: " + str(iter))
            for j, solver in enumerate(so):
                solve(solver=solver, projector_class_matched=elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True), projector_class_unmatched=elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=False), sinogram=sinogram, times=times, distances=distances, num=j, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

        # overwriting second CG with filtered ABGRMES, this is done seperately as the time measurements are weird when the python code is executed with the elsa code
        for iter in range(min_iter,max_iter,iter_steps):
            # print("Iteration: " + str(iter))
            solve(solver=SolverTest(None, ABGMRES,'unmatched ABMRES with filter', is_gmres=False, is_unmatched=True, linestyle=ls[3], is_callable=True), projector_class_matched=elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True), projector_class_unmatched=elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=False), sinogram=sinogram, times=times, distances=distances, num=3, optimal_phantom=phantom, nmax_iter=iter, repeats=repeats)

        distanceRep.append(distances)
        timesRep.append(times)

        time.sleep(60)

    # average data that can be plotted
    dist = average(distanceRep, solvers)
    tim = average(timesRep, solvers)

    print(f'Done with optimizing matched solver for current sino, starting to plot now')  

    # Plotting times
    fig, ax = plt.subplots()
    ax.set_xlabel('execution time [s]')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over execution time')
    for d, t, solver in zip(dist, tim, solvers):
        ax.plot(t, d, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.xlim(0, 0.5)
    plt.savefig(f"{args.indir}/results/" + "acryl_mse_times.pdf", bbox_inches='tight')

    # Plotting Iterations
    fig, ax = plt.subplots()
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'Mean Square Error over number of iterations')
    for d, solver in zip(dist, solvers):
        ax.plot(list(range(min_iter,max_iter,iter_steps)), d, label=solver.solver_name, linestyle=solver.linestyle)
    ax.legend()

    plt.savefig(f"{args.indir}/results/" + "acryl_mse_iter.pdf", bbox_inches='tight')

    plt.close('all')
