import tomophantom
import os
from tomophantom.supp.artifacts import _Artifacts_
from tomophantom import TomoP2D
from tomophantom.supp.libraryToDict import modelfile2Dtolist

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

    return noisy_sino

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

    # TomoPhantom starts at a different angle compared to elsa
    # the correct solution is to offset it to the correct angle
    if arc is not 180 and not 360:
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
        sino_an = add_tomophantom_poisson(sino_an)

    return phantom_2D, sino_an 

def generate_sinophantom(noise, problem_size, num_angles, arc, model, dir_path):
    # generating tomophantom sinogram with noise
    phantom, sinogram = tomophantom_model(model=model,size=problem_size,num_angles=num_angles,arc=arc,noise=noise)
    name = f'model_{model}_noise_{noise}_size_{problem_size}_arc_{arc}_angles_{num_angles}.npy'

    # Generating elsa sinogram for comparison, checking if everything is correct
    volume_descriptor = elsa.VolumeDescriptor(np.array([problem_size, problem_size]) )
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, volume_descriptor, arc, problem_size * 100, problem_size)
    projector = elsa.JosephsMethod(volume_descriptor, sino_descriptor)
    sinogramElsa = np.asarray(projector.apply(elsa.DataContainer(phantom)))
    if noise:
        sinogramElsa = add_tomophantom_poisson(np.asarray(sinogramElsa))

    np.save(dir_path + 'phantoms_sinograms/sinogram_1000_tp_' + name, np.asarray(sinogram))
    np.save(dir_path + 'phantoms_sinograms/sinogram_1000_elsa_'+name, np.asarray(sinogramElsa))
    np.save(dir_path + 'phantoms_sinograms/phantom_1000_tp_'+name, np.asarray(phantom))

problem_size = 1000
num_angles = 180
arc = 360
model_numbers = [1]
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/"

for model in model_numbers:
    # without noise
    generate_sinophantom(noise=False, problem_size=problem_size, num_angles=num_angles, arc=arc, model=model, dir_path=dir_path)
    # with noise
    generate_sinophantom(noise=True, problem_size=problem_size, num_angles=num_angles, arc=arc, model=model, dir_path=dir_path)