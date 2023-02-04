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

class IndexTracker:
    def __init__(self, rec):
        self.axs = []
        self.data = []
        self.slicingPerspective = 2
        self.slicesX, self.slicesY, self.slicesZ = rec.shape
        self.indX = self.slicesX // 2
        self.indY = self.slicesY // 2
        self.indZ = self.slicesZ // 2

    def add(self, ax, rec):
        self.axs.append(ax)
        ax.set_title(
            "scroll or use ←/→ keys to navigate \nd key to change perspective \nesc to close"
        )
        self.data.append((ax.imshow(rec[:, :, self.indZ], cmap="gray"), rec))
        self.update()
        return self

    def on_scroll(self, event):
        if event.button == "up":
            self.move(1)
        else:
            self.move(-1)
        self.update()

    def on_click(self, event):
        if event.key == "escape":
            plt.close()
            sys.exit()
        if event.key == "right":
            self.move(10)
        if event.key == "left":
            self.move(-10)
        if event.key == "d":
            self.slicingPerspective = (self.slicingPerspective + 1) % 3
        self.update()

    def getSplitValue(self):
        if self.slicingPerspective == 0:
            return self.indX
        if self.slicingPerspective == 1:
            return self.indY
        if self.slicingPerspective == 2:
            return self.indZ

    def move(self, steps):
        if self.slicingPerspective == 0:
            self.indX = (self.indX + steps) % self.slicesX
        if self.slicingPerspective == 1:
            self.indY = (self.indY + steps) % self.slicesY
        if self.slicingPerspective == 2:
            self.indZ = (self.indZ + steps) % self.slicesZ

    def getDimensionText(self):
        if self.slicingPerspective == 0:
            return "X"
        if self.slicingPerspective == 1:
            return "Y"
        if self.slicingPerspective == 2:
            return "Z"

    def getSliceData(self, rec):
        if self.slicingPerspective == 0:
            return rec[self.indX, :, :]
        if self.slicingPerspective == 1:
            return rec[:, self.indY, :]
        if self.slicingPerspective == 2:
            return rec[:, :, self.indZ]

    def update(self):
        splitValue = self.getSplitValue()
        dimension = self.getDimensionText()
        for ax in self.axs:
            ax.set_ylabel("slice %s in %s" % (splitValue, dimension))

        for (im, rec) in self.data:
            im.set_data(self.getSliceData(rec))
            im.axes.figure.canvas.draw_idle()
            im.axes.figure.canvas.draw()


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

    return projections


def preprocess(proj, binning=None):
    # Correct for slight misalignment
    proj = np.roll(proj, -3, axis=1)

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
    projections, volume_descriptor, binning=2, step=1, num_projections=720
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

    # new implementation
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
    parser.add_argument("-i", "--iterations", type=int, default=5)
    parser.add_argument("-b", "--binning", type=int, default=4)
    parser.add_argument("-s", "--step", type=int, default=1)
    parser.add_argument("-w", "--weight", type=int, default=5)
    parser.add_argument("--no-reload", action="store_false")

    args = parser.parse_args()

    num_images = 720
    projections = load_dataset(
        binning=args.binning, reload=False, step=args.step
    )

    print(args.binning)
    sz = int(math.ceil(np.max(projections.shape[:-1]) / math.sqrt(2.0)))
    volume_descriptor = elsa.VolumeDescriptor([sz] * 3, [0.125] * 3)
    sinogram = create_sinogram(
        projections,
        volume_descriptor,
        binning=args.binning,
        step=args.step,
        num_projections=num_images,
    )
    sino_descriptor = sinogram.getDataDescriptor()

    del projections

    nmax_iter = 3

    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)
    solver = elsa.BAGMRES(projector, sinogram)
    del projector
    recon = np.asarray(solver.solve(nmax_iter))
    del solver
    path = os.path.dirname(os.path.abspath(__file__))
    np.save(f"{path}/reconstruction.npy", recon)

    plt.figure(0)
    plt.imshow(recon[236], cmap="gray")
    plt.show()
    
    # fig, ax = plt.subplots(1, 1)
    # tracker = IndexTracker(np.asarray(recon)).add(ax, np.asarray(recon))
    # fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    # fig.canvas.mpl_connect('key_press_event', tracker.on_click)
    # plt.show()
