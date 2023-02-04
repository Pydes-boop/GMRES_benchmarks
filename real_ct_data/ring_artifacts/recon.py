import argparse
from pathlib import Path

import pyelsa as elsa
import matplotlib.pyplot as plt
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

def generateplot(sinogram, iters, name, solver: elsa.Solver, projector: elsa.JosephsMethodCUDA, gmres=True):

    recon = reconstruct(projector, sinogram, iters, solver, gmres=gmres)

    recon = np.interp(recon, (recon.min(), recon.max()), (0, 1))

    ax = plt.subplot()
    im = ax.imshow(recon, cmap="gray")
    ax.axis("equal")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    del projector

    plt.colorbar(im, cax=cax)
    plt.savefig(f"{args.indir}" + name, dpi=600)
    plt.close('all')

    return recon

def generatedifferenceplot(rec1, rec2, name):
    ax = plt.subplot()

    diff = np.subtract(np.asarray(rec1), np.asarray(rec2))
    diff = np.interp(diff, (diff.min(), diff.max()), (0, 1))
    im = ax.imshow(diff, cmap="gray")
    ax.axis("equal")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(f"{args.indir}" + name, dpi=600)
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--indir", type=Path, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("-f", "--file", type=Path, default="htc2022_ta_full.mat")
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
    groundTruth = np.interp(groundTruth, (groundTruth.min(), groundTruth.max()), (0, 1))

    ax = plt.subplot()
    im = ax.imshow(groundTruth, cmap="gray")
    ax.axis("equal")
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(f"{args.indir}" + "/results/groundTruth.png", dpi=600)
    plt.close('all')

    elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)

    sinogram, volume_desc = load_htc2022data(file, dataset_name=dataset, groundTruth=False)
    sinogram = apply_BHC(sinogram)

    iter = 30

    rec_slow = generateplot(sinogram, iter, "/results/recon_fast_false.png", elsa.ABGMRES, elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=False))

    rec_fast = generateplot(sinogram, iter, "/results/recon_fast_true.png", elsa.ABGMRES, elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True))

    rec_cg = generateplot(sinogram, iter, "/results/recon_CG.png", elsa.CG, elsa.JosephsMethodCUDA(volume_desc, sinogram.getDataDescriptor(), fast=True), gmres=False)

    generatedifferenceplot(rec_fast, groundTruth, "/results/difference_gmres_fast.png")

    generatedifferenceplot(rec_slow, groundTruth, "/results/difference_gmres_slow.png")

    generatedifferenceplot(rec_cg, groundTruth, "/results/difference_cg.png")

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
    # plt.show()
