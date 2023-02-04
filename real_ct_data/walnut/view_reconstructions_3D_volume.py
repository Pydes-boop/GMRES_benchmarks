import plotly.graph_objs as go
import plotly.express as px
import os
import numpy as np

def add_slices_horizontal(slice_num, array, surfaces):
    slices = []
    stepsize = int(np.shape(array)[0]/slice_num)

    for i in range(0, 419, stepsize):
        temp = array[i]
        temp = temp[:,0:419]
        slices.append(temp)

    for i, slice in enumerate(slices):
        rang = np.arange(slice.shape[0])

        # location = i * np.ones_like(slice)

        location = (i + 1) * stepsize * np.ones_like(slice)

        surface = go.Surface(x=rang,
                            y=rang,
                            z=location,
                            surfacecolor=slice,
                            # cmin/cmax needs to be manually set to be the same for every slice
                            # alternatively here the min / max value of the recon could be used directly
                            cmin = np.amin(slices), 
                            cmax = np.amax(slices),
                            # giving a colorscale, with transparency for lower values and solid colormap for higher values
                            # should basically be plasma colormap
                             colorscale=[[0,'rgba(13, 8, 135, 0)'],[0.05,'rgba(70, 3, 159, 0)'],[0.2,'rgba(114, 1, 168, 255)'],[0.3,'rgba(156, 23, 158, 255)'],[0.4,'rgba(189, 55, 134, 255)'],[0.5,'rgba(216, 87, 107, 255)'],[0.6,'rgba(237, 121, 83, 255)'],[0.7,'rgba(251, 159, 58, 255)'],[0.8,'rgba(253, 202, 38, 255)'],[1,'rgba(240, 249, 33, 255)']]
                            )
        surfaces.append(surface)
    return surfaces

def generate_volume(reco, slice_num, path, show=False, writeHtml=False, writePNG=True):
    surfaces = []
    surfaces = add_slices_horizontal(slice_num=slice_num, array=reco, surfaces=surfaces)

    # Set the axis labels and show the plot
    layout = go.Layout(scene=dict(xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z')
                                )

    layout.update(scene=dict(
                            xaxis_title=dict(font=dict(size=50)),
                            yaxis_title=dict(font=dict(size=50)),
                            zaxis_title=dict(font=dict(size=50)),
                            xaxis_showticklabels=False,
                            yaxis_showticklabels=False,
                            zaxis_showticklabels=False,
                            ))               

    fig = go.Figure(data=surfaces, layout=layout)

    fig.update_layout(width=2000, height=2000,
                    # autosize=False,
                    # title='Slices of Walnut Reconstruction', title_x=0.5, title_y=0.95,  
                    # margin=dict(l=65, r=60, b=65, t=20),
                    font=dict(
                        # family="Courier New, monospace",
                        # color="RebeccaPurple",
                        size=40,
                    )
                    )


    # Disable Background Completly
    # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

    import plotly.io as pio
    import time

    if writePNG:
        # write as image
        timestr = time.strftime("%d%m%Y-%H%M%S")
        pio.write_image(fig, file=f'{path}/3D_volumes/slices_walnut_{slice_num}_' + timestr + '.png')
        print("Wrote Image to: " + f'{path}/3D_volumes/slices_walnut_{slice_num}_' + timestr + '.png')

    if writeHtml:
        # write as html
        pio.write_html(fig, file=f'{path}/3D_volumes/index.html')
        print("Wrote HTML to: " + f'{path}/3D_volumes/index.html')


    if show:
        # Show 3D view in browser
        print("Showing 3D Volume")
        fig.show()

# Loading 3D reconstruction array from current folder
path = os.path.dirname(os.path.abspath(__file__))
# which reconstruction do you want to load?
reco = np.load(f"{path}/reconstructions/abgmres_4_10_reconstruction.npy")

# reinterpreting the array into range 0 to 1
np.interp(reco, (reco.min(), reco.max()), (0.0, 1.0))

# removing all noise below 0.02 so we can have true invisibility around the walnut
reco[reco < 0.02] = 0
reco = np.flip(reco, axis=2)

slice_num = 10

# Front of reco
generate_volume(reco=reco, slice_num=slice_num, path=path)
# Back of reco
generate_volume(reco=np.flip(reco, axis=1), slice_num=slice_num, path=path)

slice_num = 100

# Front of reco
generate_volume(reco=reco, slice_num=slice_num, path=path)
# Back of reco
generate_volume(reco=np.flip(reco, axis=1), slice_num=slice_num, path=path)

