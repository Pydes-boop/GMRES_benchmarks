import matplotlib.pyplot as plt
import numpy as np
import pyelsa as elsa
import nvtx
import os.path
import argparse
import math
import time
import numpy as np
import pyelsa as elsa
from scipy.fft import fft, ifft
import os

@nvtx.annotate("readtifs()", color="purple")
def load_dataset(path=None, binning=None, step=1, reload=True):

    if path is None:
        path = os.path.dirname(os.path.abspath(__file__))

    # If wanted, load dataset from disk instead
    if not reload and os.path.exists(f"{path}/sinogram.npy"):
        print("Reading sinogram from disk")
        return np.load(f"{path}/sinogram.npy")

    num_images = 721
    projections = None

    for i in range(0, num_images, step):
        filename = f"{path}/20201111_walnut_projections/20201111_walnut_{str(i + 1).zfill(4)}.tif"

        print(f"Processing {filename}")
        # Load data from disk
        raw = plt.imread(filename).astype(np.float16)

        # Now we can allocate memory for the projection data
        if projections is None:
            shape = None
            lastdim = num_images // step
            if binning:
                shape = tuple(np.asarray(raw.shape) // binning) + (lastdim,)
            else:
                shape = raw.shape + (lastdim,)

            projections = np.empty(shape, dtype=np.float16).transpose(1, 0, 2)

        projections[:, :, i // step] = preprocess(raw, binning=binning).transpose(1, 0)

    projections /= np.max(projections)
    np.save(f"{path}/sinogram.npy", projections)

    print(np.shape(projections))

    return projections


def preprocess(proj, binning=None):
    # Correct for slight misalignment
    proj = np.roll(proj, 3, axis=1)

    # Take a part of the background
    background = proj[:500, :500]
    I0 = np.mean(background)

    # If binning should be performed, do it now
    if binning:
        proj = rebin2d(proj, binning)

    # reduce some noise
    proj[proj > I0] = I0

    # log-transform of projection and rebinning
    proj = -np.log(proj / I0)

    return proj


def rebin2d(arr, binfac):
    """Rebin 2D array arr to shape new_shape by averaging.

    Credits to: https://scipython.com/blog/binning-a-2d-array-in-numpy/
    """

    nx, ny = arr.shape
    bnx, bny = nx // binfac, ny // binfac
    shape = (bnx, arr.shape[0] // bnx, bny, arr.shape[1] // bny)
    return arr.reshape(shape).mean(-1).mean(1)


def create_sinogram(
    projections, volume_descriptor, binning=2, step=1
):
    num_angles = projections.shape[-1] // step - 1

    # Distances given by txt file
    dist_source_detector = 553.74
    dist_source_origin = 210.66
    dist_origin_detector = dist_source_detector - dist_source_origin

    # Spacing is also given
    detector_spacing = np.asarray([0.05, 0.05]) * binning
    detector_size = projections.shape[:-1]

    sino_descriptor = elsa.CircleTrajectoryGenerator.fromAngularIncrement(
        num_angles,
        volume_descriptor,
        0.5 * step,
        dist_source_origin,
        dist_origin_detector,
        [0, 0],
        [0, 0, 0],  # Offset of origin
        detector_size,
        detector_spacing,
    )

    # if youre running a newer Version of elsa fromAngularIncrement doesnt exist!
    # Try this to create the sino_descriptor instead
    # sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    #     721,
    #     volume_descriptor,
    #     360,
    #     dist_source_origin,
    #     dist_origin_detector,
    #     [0, 0],
    #     [0, 0, 0],  # Offset of origin
    #     detector_size,
    #     detector_spacing,
    # )


    return elsa.DataContainer(projections[:, :, :num_angles:step], sino_descriptor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--binning", type=int, default=4)
    parser.add_argument("-s", "--step", type=int, default=1)
    parser.add_argument("-w", "--weight", type=int, default=5)
    parser.add_argument("--no-reload", action="store_false")

    # Change Binning for different results or if it doesnt run on your system

    args = parser.parse_args()

    projections = load_dataset(
        binning=args.binning, reload=True, step=args.step
    )

    sz = int(math.ceil(np.max(projections.shape[:-1]) / math.sqrt(2.0)))
    volume_descriptor = elsa.VolumeDescriptor([sz] * 3, [0.125] * 3)
    sinogram = create_sinogram(
        projections,
        volume_descriptor,
        binning=args.binning,
        step=args.step,
    )

    sino_descriptor = sinogram.getDataDescriptor()

    del projections

    path = os.path.dirname(os.path.abspath(__file__))

    cmap = 'gray'
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)

    l = [10,15,25]

    for iters in l:
        solver = elsa.ABGMRES(projector, sinogram)
        recon = solver.solve(iters)
        path = os.path.dirname(os.path.abspath(__file__))
        np.save(f"{path}/reconstructions/abgmres_{args.binning}_{iters}_reconstruction.npy", recon)

        solver = elsa.BAGMRES(projector, sinogram)
        recon = solver.solve(iters)
        path = os.path.dirname(os.path.abspath(__file__))
        np.save(f"{path}/reconstructions/bagmres_{args.binning}_{iters}_reconstruction.npy", recon)

        problem = elsa.WLSProblem(projector, sinogram)
        solver = elsa.CG(problem)
        recon = np.asarray(solver.solve(iters))
        path = os.path.dirname(os.path.abspath(__file__))
        np.save(f"{path}/reconstructions/bagmres_{args.binning}_{iters}_reconstruction.npy", recon)

        del solver
        del problem
        del recon

        