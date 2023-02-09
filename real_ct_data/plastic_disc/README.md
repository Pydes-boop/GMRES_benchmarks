Code for loading dataset provided by David Frank

Dataset to run script can be dowloaded at: https://zenodo.org/record/6984868
Put dataset into same folder level as reconstruction python script.

Specific reconstructions here were run with the htc2022_tc_full.mat

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
