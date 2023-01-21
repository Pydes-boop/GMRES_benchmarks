import os

import numpy as np
import matplotlib.pyplot as plt

import pyelsa as elsa

import tomophantom
from tomophantom import TomoP2D
from tomophantom.supp.libraryToDict import modelfile2Dtolist
from tomophantom.supp.artifacts import _Artifacts_

def add_tomophantom_poisson(sinogram, noise_amplitude=10000):
    sinogram = np.asarray(sinogram)
    # forming dictionaries with artifact types
    _noise_ =  {'noise_type' : 'Poisson',
                'noise_amplitude' : noise_amplitude, # noise amplitude
                'noise_seed' : 0}

    noisy_sino = _Artifacts_(sinogram, **_noise_)
    
    return noisy_sino

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

def tomophantom_model(model, size, num_angles, arc=180, noise=False):

    # select a model number from the library
    N_size = size # set dimension of the phantom
    # one can specify an exact path to the parameters file
    # path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
    path = os.path.dirname(tomophantom.__file__)
    path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
    objlist = modelfile2Dtolist(path_library2D, model) # extract parameters using Python
    #This will generate a N_size x N_size phantom (2D)
    phantom_2D = TomoP2D.Object(size, objlist)
    phantom_2D = np.asarray(phantom_2D)

    # create sinogram analytically
    angles_num = int(0.5*np.pi*N_size); # angles number

    # TomoPhantom starts at a different angle compared to elsa
    # the correct solution is to offset it to the correct angle
    if arc != 180 and arc != 360:
        start = 90
    else:
        # for some models (11...) offsetting by 90 degrees is bugged, for some reason the xray sources seem to start way too close to the phantom when offsetting it
        # this is why in the 180 degree or 360 degree case which we mainly use for testing we manually fix the sinogram using numpy later
        start = 0
    
    
    # Generating a list of angles that tomophantom will generate the analytical phantom from
    angles = np.linspace(start, start+arc,num_angles,dtype='float32')
    P = int(np.sqrt(2)*N_size) #detectors
    
    # Generating the analytical sinogram
    sino_an = TomoP2D.ObjectSino(size, P, angles, objlist)
    # Tomophantom reverses x and y on sinogram, we turn it back
    sino_an = np.asarray(sino_an).T

    if arc is 180:
        # manually adjusting the sinogram because of the off starting point
        # this results in the correct sinogram as elsa would generate numerically for 180
        shape = np.shape(sino_an)
        reshaped = blockshaped(sino_an, shape[0], int(shape[1] / 2))
        sino_an = np.concatenate((np.flipud(reshaped[1]), reshaped[0]), axis=1)
        
        sino_an = np.flipud(sino_an)
        sino_an = np.fliplr(sino_an)
    if arc is 360:
        shape = np.shape(sino_an)
        reshaped = blockshaped(sino_an, shape[0], int(shape[1] / 4))
        sino_an = np.concatenate((reshaped[3], reshaped[0], reshaped[1], reshaped[2]), axis=1)
        sino_an = np.fliplr(sino_an)

    if noise:
        noisy_sino = add_tomophantom_poisson(sino_an, noise_amplitude=10000)

        if num_angles in (20, 100, 180, 360, 420):
            plt.close('all')
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(sino_an, cmap=cmap)
            ax[0].set_title("sinogram", fontsize=8)
            ax[1].imshow(noisy_sino, cmap=cmap)
            ax[1].set_title("noisy sinogram", fontsize=8)
            ax[2].imshow(np.subtract(sino_an, noisy_sino), cmap=cmap)
            ax[2].set_title("difference sinogram-noisy", fontsize=8)
            fig.set_size_inches(16,8)
            plt.show()
            
        sino_an = noisy_sino

    return phantom_2D, sino_an 

elsa.logger_pyelsa_solvers.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_projectors.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_problems.setLevel(elsa.LogLevel.OFF)
elsa.logger_pyelsa_generators.setLevel(elsa.LogLevel.OFF)

model_number = 1
cmap = 'gray'

size = np.array([500, 500])
volume_descriptor = elsa.VolumeDescriptor(size)
arc = 360

for i in range(20,421,20):

    # settings
    num_angles = i
    print("generating angles: "+ str(num_angles))

    # generate circular trajectory
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, volume_descriptor, arc, size[0] * 100, size[0])

    # setup operator for 2d X-ray transform
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, fast=True)
    backprojector = elsa.adjoint(projector)

    ### using tomophantom
    phantom, sinogram = tomophantom_model(model_number, size[0], num_angles, arc, noise=True)

    sinogramElsa = projector.apply(elsa.DataContainer(phantom))

    solver = elsa.ABGMRES(projector,  backprojector, elsa.DataContainer(sinogram))
    rec = solver.solve(10)

    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].imshow(phantom, cmap=cmap)
    ax[0].set_title("phantom tomophantom model: " + str(model_number), fontsize=8)
    ax[1].imshow(sinogram, cmap=cmap)
    ax[1].set_title("analytical sinogram tomophantom", fontsize=8)
    ax[2].imshow(np.asarray(sinogram) - np.asarray(sinogramElsa), cmap=cmap)
    ax[2].set_title("difference sinogram elsa/tomophantom", fontsize=8)
    ax[3].imshow(rec, cmap=cmap)
    ax[3].set_title("Reconstruction tomophantom model: " + str(model_number), fontsize=8)
    fig.set_size_inches(16,8)
    
    if num_angles in (20, 100, 180, 360, 420):
        print("Showing Phantom, Sinograms and Reconstruction for angles: " + str(num_angles))
        plt.show()

    plt.close()