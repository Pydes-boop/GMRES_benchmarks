import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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

import time
# folder path
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/reconstructions/"

save_path = os.path.dirname(os.path.abspath(__file__)) + "/"

# https://pynative.com/python-list-files-in-a-directory/
# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    res.append(os.path.join(dir_path, path))

# Getting all the important phantoms, with and without noise
res = [s for s in res if ".npy" in s]
res = [s for s in res if "gmres" in s]
# res = [s for s in res if "2_2" in s]
res.sort()

dir_path += "/"

cmap = "gray"

for file in res:
    print(file)
    recon = np.load(file)

    # Uncomment to view 3D Volume 

    # fig, ax = plt.subplots(1, 1)
    # tracker = IndexTracker(np.asarray(recon)).add(ax, np.asarray(recon))
    # fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    # fig.canvas.mpl_connect('key_press_event', tracker.on_click)
    # plt.show()

    figures, ax = plt.subplots()
    ax.axis("off")
    if recon.shape[0] > 419:
        view = recon[411]
        view = view[209:628, 209:628]
    else:
        view = recon[202]

    
    ax.imshow(view, cmap=cmap)

    plt.savefig(file.replace(".npy", ".png"), dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    figures, ax = plt.subplots()
    ax.axis("off")
    if recon.shape[0] > 419:
        view = recon[:,411]
        view = view[209:628, 209:628]
    else:
        view = recon[:,202]

    
    ax.imshow(view, cmap=cmap)

    plt.savefig(file.replace(".npy", ".png"), dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close('all')