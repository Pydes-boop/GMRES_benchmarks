from libraries.lib_noise import *
from libraries.lib_tomophantom_sinogram import *
from libraries.lib_parula_cmap import *
from libraries.lib_filtered_backprojection import *
from matplotlib import ticker
import matplotlib.patches as patches


#### Setup ####
cmap = "gray"
epsilon = None
size = np.array([1000, 1000])
model_number = 1

def ret_backproject(num_angles, arc, noise=False):
    ### using tomophantom
    phantom, sinogram = tomophantom_model(model_number, size[0], num_angles, arc, noise=False)
    sinogram = elsa.DataContainer(sinogram)
    # sinogram = np.reshape(sinogram, [num_angles, int(np.sqrt(2)*size[0])])
    # print(np.shape(sinogram))

    volume_descriptor = elsa.VolumeDescriptor(size)

    # generate circular trajectory
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, volume_descriptor, arc, size[0]*100, size[0])

    # setup operator for 2d X-ray transform
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)

    if noise:
        sinogram = add_tomophantom_poisson(sinogram)

    # from tomobar.methodsDIR import RecToolsDIR

    # angles = np.linspace(0, arc-0.1,num_angles,dtype='float32')
    # angles_rad = angles*(np.pi/180.0)
    # P = int(np.sqrt(2)*size[0]) #detectors

    # RecToolsDIR = RecToolsDIR(DetectorsDimH = P, # Horizontal detector dimension
    #                     DetectorsDimV = None,            # Vertical detector dimension
    #                     CenterRotOffset = None,          # Center of Rotation scalar
    #                     AnglesVec = angles_rad,          # A vector of projection angles in radians
    #                     ObjSize = size[0],                # Reconstructed object dimensions (scalar)
    #                     device_projector='cpu')
                        
    # filtered_backprojection = RecToolsDIR.FBP(sinogram) #perform FBP
    filtered_backprojection = projector.applyAdjoint(elsa.DataContainer(sinogram))
    # plt.figure(0)
    # plt.imshow(np.asarray(filtered_backprojection))
    # plt.show()


    filtered_backprojection = projector.applyAdjoint(elsa.DataContainer(filter_sinogram(sinogram, sino_descriptor)))
    filtered_backprojection = np.asarray(filtered_backprojection)
    filtered_backprojection = filtered_backprojection

    return filtered_backprojection

def CG(num_angles, arc, n, noise=False):
    ### using tomophantom
    phantom, sinogram = tomophantom_model(model_number, size[0], num_angles, arc, noise=False)
    sinogram = elsa.DataContainer(sinogram)
    # sinogram = np.reshape(sinogram, [num_angles, int(np.sqrt(2)*size[0])])
    # print(np.shape(sinogram))

    volume_descriptor = elsa.VolumeDescriptor(size)

    # generate circular trajectory
    sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
        num_angles, volume_descriptor, arc, size[0]*100, size[0])

    # setup operator for 2d X-ray transform
    projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)

    if noise:
        sinogram = add_tomophantom_poisson(sinogram)
    sinogram = elsa.DataContainer(sinogram)
    
    problem = elsa.WLSProblem(projector, sinogram)
    solver = elsa.CG(problem, 0.00005)
    reconstruction = solver.solve(n)

    return np.asarray(reconstruction)

arc = 180

fntsize = 8

figure, ax = plt.subplots(nrows=2, ncols=4, figsize=(13.2, 6))

i = 0

fig = []

fig.append(ret_backproject(1600,arc, True))
fig.append(ret_backproject(400,arc, True))
fig.append(ret_backproject(40,arc, True))
fig.append(CG(40, arc, 10, True))


i = 0
ax[i,0].imshow(fig[0], vmin=np.min(fig[0]), vmax=np.max(fig[0]), cmap=cmap)
rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
ax[i,0].add_patch(rect)
ax[i,1].imshow(fig[1], vmin=np.min(fig[1]), vmax=np.max(fig[1]), cmap=cmap)
rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
ax[i,1].add_patch(rect)
ax[i,2].imshow(fig[2], vmin=np.min(fig[2]), vmax=np.max(fig[2]), cmap=cmap)
rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
ax[i,2].add_patch(rect)
ax[i,3].imshow(fig[3], vmin=np.min(fig[3]), vmax=np.max(fig[3]), cmap=cmap)
rect = patches.Rectangle((400, 400), 200, 200, linewidth=1, edgecolor='r', facecolor='none')
ax[i,3].add_patch(rect)

ax[i,0].axis('off')
ax[i,1].axis('off')
ax[i,2].axis('off')
ax[i,3].axis('off')

i = 1
ax[i,0].imshow(fig[0][400:600,400:600], vmin=np.min(fig[0]), vmax=np.max(fig[0]), cmap=cmap)
ax[i,1].imshow(fig[1][400:600,400:600], vmin=np.min(fig[1]), vmax=np.max(fig[1]), cmap=cmap)
im = ax[i,2].imshow(fig[2][400:600,400:600], vmin=np.min(fig[2]), vmax=np.max(fig[2]), cmap=cmap)
ax[i,3].imshow(fig[3][400:600,400:600], vmin=np.min(fig[3]), vmax=np.max(fig[3]), cmap=cmap)

ax[i,0].axis('off')
ax[i,1].axis('off')
ax[i,2].axis('off')
ax[i,3].axis('off')

plt.subplots_adjust(wspace=0, hspace=0)

ticklabels = ['0', '0.5', '1.0']
cb = figure.colorbar(im,ax=ax.ravel().tolist(), fraction=0.046)
tick_locator = ticker.LinearLocator(numticks=3)
cb.locator = tick_locator
cb.update_ticks()
cb.set_ticklabels(ticklabels)

# cb.outline.set_visible(False)
# cb.set_ticks([])
# cb.set_ticks([0, 0.5, 1])

plt.savefig("FBP_examples.png", dpi=300, bbox_inches='tight')
plt.show()

# def save_fig(f, name, m):
#     plt.figure()
#     plt.imshow(f, cmap=cmap, vmax=m)
#     plt.axis("off")
#     plt.savefig(name, dpi=600)

# count = 1
# for f in fig:
#     save_fig(f, "FBP_Examples_" + str(count) + ".png", np.max(f))
#     save_fig(f[400:600,400:600], "FBP_Examples_" + str(count) + "_zoomed.png", np.max(f))
#     count = count + 1