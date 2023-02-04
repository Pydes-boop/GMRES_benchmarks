Code for loading dataset provided by David Frank and part of [elsa MR 320](https://gitlab.lrz.de/IP/elsa/-/merge_requests/320)

Dataset to run script can be dowloaded at: https://zenodo.org/record/6986012
Put all tif images (dataset) into 20201111_walnut_projections folder

Currently this script is generating the sinogram via fromAngularIncrement:

```
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
```

This might not be part of your elsa version, instead you can try to use createTrajectory instead of fromAngularIncrement:

```
sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    721,
    volume_descriptor,
    360,
    dist_source_origin,
    dist_origin_detector,
    [0, 0],
    [0, 0, 0],  # Offset of origin
    detector_size,
    detector_spacing,
)
```

3D Volume showcase was created by me

3D slices showcase is from: https://plotly.com/python/visualizing-mri-volume-slices/