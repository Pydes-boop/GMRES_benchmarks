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

    print(np_sinogram.shape)
    print(projection_size_padded)
    print(pad_width)
    print(np.shape(padded_sinogram))
    print(sinogram_shape)

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

problem_size = 1000
arc=360
num_angles=180

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

res = []

# Iterate directory
for path in os.listdir(dir_path):
    res.append(os.path.join(dir_path, path))

dir_path += "/"

# Getting all the important phantoms, with and without noise
phantoms = [s for s in res if "phantom_1000_tp" in s]
phantomsNoise = [s for s in phantoms if "noise_True" in s]
phantomsNoise.sort()
phantoms = [s for s in phantoms if "noise_False" in s]
phantoms.sort()

# Getting all the important sinograms, with and without noise
sinograms = [s for s in res if "sinogram_1000_tp" in s]
sinogramsNoise = [s for s in sinograms if "noise_True" in s]
sinogramsNoise.sort()
sinograms = [s for s in sinograms if "noise_False" in s]
sinograms.sort()

from typing import Callable

def solve(sino_descriptor, nmax_iterations: int, solverClass: Callable, projector: elsa.JosephsMethodCUDA, sinogram: elsa.DataContainer, x0: elsa.DataContainer, filter: bool):
    x = np.asarray(solverClass(projector, elsa.adjoint(projector), sinogram, x0, nmax_iterations, sino_descriptor=sino_descriptor, filter=filter))
    return x

from matplotlib.colors import LinearSegmentedColormap

# From https://stackoverflow.com/questions/34859628/has-someone-made-the-parula-colormap-in-matplotlib

cm_data = [[0.2422, 0.1504, 0.6603],
[0.2444, 0.1534, 0.6728],
[0.2464, 0.1569, 0.6847],
[0.2484, 0.1607, 0.6961],
[0.2503, 0.1648, 0.7071],
[0.2522, 0.1689, 0.7179],
[0.254, 0.1732, 0.7286],
[0.2558, 0.1773, 0.7393],
[0.2576, 0.1814, 0.7501],
[0.2594, 0.1854, 0.761],
[0.2611, 0.1893, 0.7719],
[0.2628, 0.1932, 0.7828],
[0.2645, 0.1972, 0.7937],
[0.2661, 0.2011, 0.8043],
[0.2676, 0.2052, 0.8148],
[0.2691, 0.2094, 0.8249],
[0.2704, 0.2138, 0.8346],
[0.2717, 0.2184, 0.8439],
[0.2729, 0.2231, 0.8528],
[0.274, 0.228, 0.8612],
[0.2749, 0.233, 0.8692],
[0.2758, 0.2382, 0.8767],
[0.2766, 0.2435, 0.884],
[0.2774, 0.2489, 0.8908],
[0.2781, 0.2543, 0.8973],
[0.2788, 0.2598, 0.9035],
[0.2794, 0.2653, 0.9094],
[0.2798, 0.2708, 0.915],
[0.2802, 0.2764, 0.9204],
[0.2806, 0.2819, 0.9255],
[0.2809, 0.2875, 0.9305],
[0.2811, 0.293, 0.9352],
[0.2813, 0.2985, 0.9397],
[0.2814, 0.304, 0.9441],
[0.2814, 0.3095, 0.9483],
[0.2813, 0.315, 0.9524],
[0.2811, 0.3204, 0.9563],
[0.2809, 0.3259, 0.96],
[0.2807, 0.3313, 0.9636],
[0.2803, 0.3367, 0.967],
[0.2798, 0.3421, 0.9702],
[0.2791, 0.3475, 0.9733],
[0.2784, 0.3529, 0.9763],
[0.2776, 0.3583, 0.9791],
[0.2766, 0.3638, 0.9817],
[0.2754, 0.3693, 0.984],
[0.2741, 0.3748, 0.9862],
[0.2726, 0.3804, 0.9881],
[0.271, 0.386, 0.9898],
[0.2691, 0.3916, 0.9912],
[0.267, 0.3973, 0.9924],
[0.2647, 0.403, 0.9935],
[0.2621, 0.4088, 0.9946],
[0.2591, 0.4145, 0.9955],
[0.2556, 0.4203, 0.9965],
[0.2517, 0.4261, 0.9974],
[0.2473, 0.4319, 0.9983],
[0.2424, 0.4378, 0.9991],
[0.2369, 0.4437, 0.9996],
[0.2311, 0.4497, 0.9995],
[0.225, 0.4559, 0.9985],
[0.2189, 0.462, 0.9968],
[0.2128, 0.4682, 0.9948],
[0.2066, 0.4743, 0.9926],
[0.2006, 0.4803, 0.9906],
[0.195, 0.4861, 0.9887],
[0.1903, 0.4919, 0.9867],
[0.1869, 0.4975, 0.9844],
[0.1847, 0.503, 0.9819],
[0.1831, 0.5084, 0.9793],
[0.1818, 0.5138, 0.9766],
[0.1806, 0.5191, 0.9738],
[0.1795, 0.5244, 0.9709],
[0.1785, 0.5296, 0.9677],
[0.1778, 0.5349, 0.9641],
[0.1773, 0.5401, 0.9602],
[0.1768, 0.5452, 0.956],
[0.1764, 0.5504, 0.9516],
[0.1755, 0.5554, 0.9473],
[0.174, 0.5605, 0.9432],
[0.1716, 0.5655, 0.9393],
[0.1686, 0.5705, 0.9357],
[0.1649, 0.5755, 0.9323],
[0.161, 0.5805, 0.9289],
[0.1573, 0.5854, 0.9254],
[0.154, 0.5902, 0.9218],
[0.1513, 0.595, 0.9182],
[0.1492, 0.5997, 0.9147],
[0.1475, 0.6043, 0.9113],
[0.1461, 0.6089, 0.908],
[0.1446, 0.6135, 0.905],
[0.1429, 0.618, 0.9022],
[0.1408, 0.6226, 0.8998],
[0.1383, 0.6272, 0.8975],
[0.1354, 0.6317, 0.8953],
[0.1321, 0.6363, 0.8932],
[0.1288, 0.6408, 0.891],
[0.1253, 0.6453, 0.8887],
[0.1219, 0.6497, 0.8862],
[0.1185, 0.6541, 0.8834],
[0.1152, 0.6584, 0.8804],
[0.1119, 0.6627, 0.877],
[0.1085, 0.6669, 0.8734],
[0.1048, 0.671, 0.8695],
[0.1009, 0.675, 0.8653],
[0.0964, 0.6789, 0.8609],
[0.0914, 0.6828, 0.8562],
[0.0855, 0.6865, 0.8513],
[0.0789, 0.6902, 0.8462],
[0.0713, 0.6938, 0.8409],
[0.0628, 0.6972, 0.8355],
[0.0535, 0.7006, 0.8299],
[0.0433, 0.7039, 0.8242],
[0.0328, 0.7071, 0.8183],
[0.0234, 0.7103, 0.8124],
[0.0155, 0.7133, 0.8064],
[0.0091, 0.7163, 0.8003],
[0.0046, 0.7192, 0.7941],
[0.0019, 0.722, 0.7878],
[0.0009, 0.7248, 0.7815],
[0.0018, 0.7275, 0.7752],
[0.0046, 0.7301, 0.7688],
[0.0094, 0.7327, 0.7623],
[0.0162, 0.7352, 0.7558],
[0.0253, 0.7376, 0.7492],
[0.0369, 0.74, 0.7426],
[0.0504, 0.7423, 0.7359],
[0.0638, 0.7446, 0.7292],
[0.077, 0.7468, 0.7224],
[0.0899, 0.7489, 0.7156],
[0.1023, 0.751, 0.7088],
[0.1141, 0.7531, 0.7019],
[0.1252, 0.7552, 0.695],
[0.1354, 0.7572, 0.6881],
[0.1448, 0.7593, 0.6812],
[0.1532, 0.7614, 0.6741],
[0.1609, 0.7635, 0.6671],
[0.1678, 0.7656, 0.6599],
[0.1741, 0.7678, 0.6527],
[0.1799, 0.7699, 0.6454],
[0.1853, 0.7721, 0.6379],
[0.1905, 0.7743, 0.6303],
[0.1954, 0.7765, 0.6225],
[0.2003, 0.7787, 0.6146],
[0.2061, 0.7808, 0.6065],
[0.2118, 0.7828, 0.5983],
[0.2178, 0.7849, 0.5899],
[0.2244, 0.7869, 0.5813],
[0.2318, 0.7887, 0.5725],
[0.2401, 0.7905, 0.5636],
[0.2491, 0.7922, 0.5546],
[0.2589, 0.7937, 0.5454],
[0.2695, 0.7951, 0.536],
[0.2809, 0.7964, 0.5266],
[0.2929, 0.7975, 0.517],
[0.3052, 0.7985, 0.5074],
[0.3176, 0.7994, 0.4975],
[0.3301, 0.8002, 0.4876],
[0.3424, 0.8009, 0.4774],
[0.3548, 0.8016, 0.4669],
[0.3671, 0.8021, 0.4563],
[0.3795, 0.8026, 0.4454],
[0.3921, 0.8029, 0.4344],
[0.405, 0.8031, 0.4233],
[0.4184, 0.803, 0.4122],
[0.4322, 0.8028, 0.4013],
[0.4463, 0.8024, 0.3904],
[0.4608, 0.8018, 0.3797],
[0.4753, 0.8011, 0.3691],
[0.4899, 0.8002, 0.3586],
[0.5044, 0.7993, 0.348],
[0.5187, 0.7982, 0.3374],
[0.5329, 0.797, 0.3267],
[0.547, 0.7957, 0.3159],
[0.5609, 0.7943, 0.305],
[0.5748, 0.7929, 0.2941],
[0.5886, 0.7913, 0.2833],
[0.6024, 0.7896, 0.2726],
[0.6161, 0.7878, 0.2622],
[0.6297, 0.7859, 0.2521],
[0.6433, 0.7839, 0.2423],
[0.6567, 0.7818, 0.2329],
[0.6701, 0.7796, 0.2239],
[0.6833, 0.7773, 0.2155],
[0.6963, 0.775, 0.2075],
[0.7091, 0.7727, 0.1998],
[0.7218, 0.7703, 0.1924],
[0.7344, 0.7679, 0.1852],
[0.7468, 0.7654, 0.1782],
[0.759, 0.7629, 0.1717],
[0.771, 0.7604, 0.1658],
[0.7829, 0.7579, 0.1608],
[0.7945, 0.7554, 0.157],
[0.806, 0.7529, 0.1546],
[0.8172, 0.7505, 0.1535],
[0.8281, 0.7481, 0.1536],
[0.8389, 0.7457, 0.1546],
[0.8495, 0.7435, 0.1564],
[0.86, 0.7413, 0.1587],
[0.8703, 0.7392, 0.1615],
[0.8804, 0.7372, 0.165],
[0.8903, 0.7353, 0.1695],
[0.9, 0.7336, 0.1749],
[0.9093, 0.7321, 0.1815],
[0.9184, 0.7308, 0.189],
[0.9272, 0.7298, 0.1973],
[0.9357, 0.729, 0.2061],
[0.944, 0.7285, 0.2151],
[0.9523, 0.7284, 0.2237],
[0.9606, 0.7285, 0.2312],
[0.9689, 0.7292, 0.2373],
[0.977, 0.7304, 0.2418],
[0.9842, 0.733, 0.2446],
[0.99, 0.7365, 0.2429],
[0.9946, 0.7407, 0.2394],
[0.9966, 0.7458, 0.2351],
[0.9971, 0.7513, 0.2309],
[0.9972, 0.7569, 0.2267],
[0.9971, 0.7626, 0.2224],
[0.9969, 0.7683, 0.2181],
[0.9966, 0.774, 0.2138],
[0.9962, 0.7798, 0.2095],
[0.9957, 0.7856, 0.2053],
[0.9949, 0.7915, 0.2012],
[0.9938, 0.7974, 0.1974],
[0.9923, 0.8034, 0.1939],
[0.9906, 0.8095, 0.1906],
[0.9885, 0.8156, 0.1875],
[0.9861, 0.8218, 0.1846],
[0.9835, 0.828, 0.1817],
[0.9807, 0.8342, 0.1787],
[0.9778, 0.8404, 0.1757],
[0.9748, 0.8467, 0.1726],
[0.972, 0.8529, 0.1695],
[0.9694, 0.8591, 0.1665],
[0.9671, 0.8654, 0.1636],
[0.9651, 0.8716, 0.1608],
[0.9634, 0.8778, 0.1582],
[0.9619, 0.884, 0.1557],
[0.9608, 0.8902, 0.1532],
[0.9601, 0.8963, 0.1507],
[0.9596, 0.9023, 0.148],
[0.9595, 0.9084, 0.145],
[0.9597, 0.9143, 0.1418],
[0.9601, 0.9203, 0.1382],
[0.9608, 0.9262, 0.1344],
[0.9618, 0.932, 0.1304],
[0.9629, 0.9379, 0.1261],
[0.9642, 0.9437, 0.1216],
[0.9657, 0.9494, 0.1168],
[0.9674, 0.9552, 0.1116],
[0.9692, 0.9609, 0.1061],
[0.9711, 0.9667, 0.1001],
[0.973, 0.9724, 0.0938],
[0.9749, 0.9782, 0.0872],
[0.9769, 0.9839, 0.0805]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

cmap = "gray"
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as patches

def create_figure(filter):
    fig = []

    i = 0
    fig.append(solve(sino_descriptor=sino_descriptor, solverClass=BAGMRES, projector=projector, sinogram=elsa.DataContainer(np.load(sinogramsNoise[0])), x0=x0, filter=filter, nmax_iterations=2))
    fig.append(solve(sino_descriptor=sino_descriptor, solverClass=BAGMRES, projector=projector, sinogram=elsa.DataContainer(np.load(sinogramsNoise[0])), x0=x0, filter=filter, nmax_iterations=5))
    fig.append(solve(sino_descriptor=sino_descriptor, solverClass=BAGMRES, projector=projector, sinogram=elsa.DataContainer(np.load(sinogramsNoise[0])), x0=x0, filter=filter, nmax_iterations=10))
    fig.append(solve(sino_descriptor=sino_descriptor, solverClass=BAGMRES, projector=projector, sinogram=elsa.DataContainer(np.load(sinogramsNoise[0])), x0=x0, filter=filter, nmax_iterations=20))

    ax[0, i].imshow(fig[0], vmin=np.min(fig[0]), vmax=np.max(fig[0]), cmap=cmap)
    ax[1, i].imshow(fig[1], vmin=np.min(fig[1]), vmax=np.max(fig[1]), cmap=cmap)
    ax[2, i].imshow(fig[2], vmin=np.min(fig[2]), vmax=np.max(fig[2]), cmap=cmap)
    ax[3, i].imshow(fig[3], vmin=np.min(fig[3]), vmax=np.max(fig[3]), cmap=cmap)

    # Add rectangle to images
    rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
    ax[0, i].add_patch(rect)
    rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
    ax[1, i].add_patch(rect)
    rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
    ax[2, i].add_patch(rect)
    rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
    ax[3, i].add_patch(rect)

    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')
    ax[3, i].axis('off')

    i = 1
    ax[0, i].imshow(fig[0][400:600,400:600], vmin=np.min(fig[0]), vmax=np.max(fig[0]), cmap=cmap)
    ax[1, i].imshow(fig[1][400:600,400:600], vmin=np.min(fig[1]), vmax=np.max(fig[1]), cmap=cmap)
    ax[2, i].imshow(fig[2][400:600,400:600], vmin=np.min(fig[2]), vmax=np.max(fig[2]), cmap=cmap)
    im = ax[3, i].imshow(fig[3][400:600,400:600], vmin=np.min(fig[3]), vmax=np.max(fig[3]), cmap=cmap)

    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')
    ax[3, i].axis('off')

    return im

fntsize = 8

figure, ax = plt.subplots(nrows=4, ncols=2, figsize=(5.98, 12))

im = create_figure(True)

plt.subplots_adjust(wspace=0, hspace=0)

# ticklabels = ['0', '0.5', '1.0']
# cb = figure.colorbar(im, ax=ax.ravel().tolist(), fraction=0.084, pad=0.04)
# tick_locator = ticker.LinearLocator(numticks=3)
# cb.locator = tick_locator
# cb.update_ticks()
# cb.set_ticklabels(ticklabels)

plt.savefig("gmres_fbp_comparison_filter.png", dpi=300, bbox_inches='tight')
# plt.show()

fntsize = 8

figure, ax = plt.subplots(nrows=4, ncols=2, figsize=(6.9, 12))

im = create_figure(False)

plt.subplots_adjust(wspace=0, hspace=0)

ticklabels = ['0', '0.5', '1.0']
cb = figure.colorbar(im, ax=ax.ravel().tolist(), fraction=0.084)
tick_locator = ticker.LinearLocator(numticks=3)
cb.locator = tick_locator
cb.update_ticks()
cb.set_ticklabels(ticklabels)

plt.savefig("gmres_fbp_comparison_no_filter.png", dpi=300, bbox_inches='tight')
# plt.show()