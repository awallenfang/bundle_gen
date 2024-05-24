# Bundle gen

A framework for generating wavelength light scattering tables of bundles of fibers based on singular fiber tables.

This was created as part of my bachelor thesis on light scattering in fibers.

## Description

Essentially this framework traces rays through a bundle structure and aggregates the resulting data into a new table.
For it to work the data for a single fiber has to be generated through methods like the following: https://github.com/mandyxmq/WaveFiber3d/

These can then be read into a high-dimensional table in the code and subsequently can be used for the rendering.

## Process

Once the light scattering tables have been generated the fiber bundle structure is created by placing parallel cylinders in 3d space.
This representation is loaded as a scene to then be traced through using the [Mitsuba Rendering Engine](https://mitsuba.readthedocs.io/en/latest/) and [Dr. JIT](https://drjit.readthedocs.io/en/latest/).

The light is then traced using common path tracing methods. Each ray that leaves the scene is used to slowly build the resulting table using its magnitude and orientation.

The resulting table is then aggregated and saved in a manner that allows it to use it further down the road.

## Running the Code

First my fork of the Mitsuba library has to be built, since slight changes to the cylinder interaction were required. For further info on this step head to the [Mitsuba Developer Guid](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html)

```bash
git clone --recursive git@github.com:awallenfang/mitsuba3.git
cd ./mitsuba3
mkdir build
cd build
cmake -GNinja ..
ninja
```

This Mitsuba version then has to be activated to be used by Python:
```bash
source setpath.sh
```

Once the Mitsuba fork was successfully activated the generated light scattering data should be placed in a folder called `fiber_model`. The representation and structure that is used here is the one by Mandy Xia et al.: https://github.com/mandyxmq/WaveFiber3d/

Changes to the structure or representation can be done in the __init__ method of the TabulatedBCRDF class. Or a new BRDF class can be created as well.

Once the data has been generated the bundle structure can be defined by filling the fiber list at the top of `main.py`.

Finally the code can be run by simply running `main.py`

```py
python main.py
```

## Results

The results and inner workings of the code are also described in my thesis that can be found in this repository: [Computation of BCSDFs for elliptical multi-fiber bundles](Computation_of_BCSDFs_for_elliptical_multi-fiber_bundles.pdf)