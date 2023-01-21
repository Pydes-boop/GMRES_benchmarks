import os
import matplotlib.pyplot as plt
import numpy as np

def view_models(res, dir_path):
    dir_path +="/"
    phantoms = [s for s in res if "phantom" in s]
    for phantom in phantoms:
        phan = np.load(dir_path + phantom)
        sinotp = np.load(dir_path + phantom.replace("phantom", "sinogram"))
        sinoElsa = np.load(dir_path + phantom.replace("phantom_tp", "sinogram_elsa"))

        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=4)
        ax[0].imshow(phan, cmap=cmap)
        ax[0].set_title("phantom", fontsize=8)
        ax[1].imshow(sinotp, cmap=cmap)
        ax[1].set_title("sinogram tomophantom", fontsize=8)
        ax[2].imshow(sinoElsa, cmap=cmap)
        ax[2].set_title("sinogram elsa", fontsize=8)
        ax[3].imshow(np.subtract(sinotp, sinoElsa), cmap=cmap)
        ax[3].set_title("difference between analytical sinogram tp - elsa", fontsize=8)
        plt.suptitle(phantom.replace("_", " "))
        plt.show()

# folder path
dir_path = os.path.dirname(os.path.abspath(__file__))

# https://pynative.com/python-list-files-in-a-directory/
# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)

cmap = "viridis"

view_models(res=res, dir_path=dir_path)