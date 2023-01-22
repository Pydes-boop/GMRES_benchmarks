import os
from tomophantom.supp.artifacts import _Artifacts_

import numpy as np
import pyelsa as elsa

# function we need because tomophantom creates sinograms correctly like elsa but in a weird format
# https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# Noise added like TomoPhantom/Demos/Python/Recon
def add_tomophantom_poisson(sinogram, noise_amplitude = 10000):
    sinogram = np.asarray(sinogram)
    # forming dictionaries with artifact types
    _noise_ =  {'noise_type' : 'Poisson',
                'noise_amplitude' : noise_amplitude, # noise amplitude
                'noise_seed' : 0}

    noisy_sino = _Artifacts_(sinogram, **_noise_)

    print(np.linalg.norm(np.subtract(noisy_sino, sinogram)))

    return noisy_sino

def generate_sinophantom(noise, problem_size, num_angles, arc, dir_path):

    ### --- Setup Elsa --- ###
    size = np.array([problem_size, problem_size, problem_size])
    phantom = elsa.phantoms.modifiedSheppLogan(size)
    volume_descriptor = elsa.VolumeDescriptor(size)
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(num_angles, volume_descriptor, arc, size[0] * 100, size[0])

    # setup operator for 2d X-ray transform
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=False)

    # simulate the sinogram
    sinogram = projector.apply(phantom)

    if noise:
        sinogram = add_tomophantom_poisson(np.asarray(sinogram))

    name = f'noise_{noise}_size_{problem_size}_arc_{arc}_angles_{num_angles}.npy'
    np.save(dir_path + 'phantoms_sinograms/sinogram_elsa_3D_' + name, np.asarray(sinogram))
    np.save(dir_path + 'phantoms_sinograms/phantom_elsa_3D_'+ name, np.asarray(phantom))

problem_size = 64
num_angles = 180
arc = 360
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"

# without noise
generate_sinophantom(noise=False, problem_size=problem_size, num_angles=num_angles, arc=arc, dir_path=dir_path)
# with noise
generate_sinophantom(noise=True, problem_size=problem_size, num_angles=num_angles, arc=arc, dir_path=dir_path)