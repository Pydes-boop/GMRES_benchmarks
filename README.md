# GMRES benchmarks

Collection of benchmarks and experiments scripts that are used to evaluate AB-GMRES and BA-GMRES implementations for pyelsa. This collection of scripts includes all code that was used to generate benchmarks for the ABBA GMRES implementations in [elsa](https://gitlab.lrz.de/IP/elsa) that were implemented , as well as some experiments using and applying [tomophantom](https://github.com/dkazanc/TomoPhantom) with elsa.

Installation instructions for all dependencies can be found at the [bottom](#prerequisites-and-installation-instructions) of the page

If you are interested in how GMRES is implemented you can also find different python versions, with or without elsa bindings at [Pydes-boop/GMRES](https://github.com/Pydes-boop/GMRES), which also gives a short introduction into the GMRES algorithms and their use-cases.

## Potential Issues:

Experiments were run on older custom version of elsa with CG solver which has been removed since then, replace any occurence of CG with the new CGLS solver.

CircleTrajectoryGenerator is also from the experimental elsa version and has to be replaced with a fitting up-to-date one. Details can be found in [walnut readme](real_ct_data/walnut/README.md) but might be relevant to plastic disc experiments as well.

## Content

### Benchmarking
   

1.   [angles_comparison.py](benchmarking/angles_comparison.py)
     - compares scenarios with different amounts of angles starting at 20, up to 500 in steps of +20 for an arc of 180°
     - compares matched and unmatched versions of AB-, BA- GMRES against CG
     
1.   [filter_comparison_analytical.py](benchmarking/filter_comparison_analytical.py)
     - compares different solvers with filtered Versions of AB- , BA-GMRES 

3.   [matched_comparison_2D.py](benchmarking/matched_comparison_2D.py)
     - compares matched and unmatched versions of AB-, BA- GMRES against CG  
     
4.   [matched_comparison_3D.py](benchmarking/matched_comparison_3D.py)
     - compares different solvers with filtered Versions of AB- , BA-GMRES


### Experiments / Showcases

1.   [fbp_examples.py](experiments/fbp_examples.py)
     - generates different example images for reconstruction with FBP vs CG
     
2.   [gmres_fbp_visualcomparison.py](experiments/gmres_fbp_visualcomparison.py)
     - compares GMRES to GMRES with additional filtering in visual examples
     
3.   Sinogram generators for tests:
     - [sinogram_generator_3D.py](experiments/sinogram_generator_3D.py)
     - [sinogram_generator.py](experiments/sinogram_generator.py)
     - [tomophantom_sinogram.py](experiments/tomophantom_sinogram.py)
     

### Real CT Data testing:

1. [2D plastic disc reconstructions](real_ct_data/plastic_disc/):
     - [dataset](https://zenodo.org/record/6984868)
     - reconstructs plastic disc with CG and GMRES algorithms
     - visual image comparison and benchmark based on FBP as ground truth

2. [3D walnut reconstruction](real_ct_data/walnut/):
     - [dataset](https://zenodo.org/record/6986012)
     - reconstructs walnut with GMRES and CG
     - 3D visual comparison with volume and slice viewers
     - visual image comparison of single specific slices

## Prerequisites and installation instructions

### requirements for building [elsa](https://gitlab.lrz.de/IP/elsa) and [tomophantom](https://github.com/dkazanc/TomoPhantom):

```
# elsa (check [repository](https://gitlab.lrz.de/IP/elsa) for any possible changes)
- C++17 compliant compiler: GCC >= 9 or Clang >= 9
- CMake >= 3.14
- CUDA toolkit >= 10.2
```

```
# tomophantom (check [repository](https://github.com/dkazanc/TomoPhantom) for any possible changes)
- CMake >= 3.0
- a C compiler with OpenMP
- libm on linux
```

### Install necessary Python3 pip dependencies:

```bash
pip install numpy matplotlib scipy
```

### install [elsa](https://gitlab.lrz.de/IP/elsa) python bindings:

```bash
git clone https://gitlab.lrz.de/IP/elsa
cd elsa

# using the --verbose flag the console output should show you if CUDA is enabled

pip install . --verbose

# sometimes this has difficulty to find CUDAs thrust on your system, try linking your CUDA directory for CMake:

CMAKE_ARGS="-DThrust_DIR=/usr/local/cuda-11.8/targets/x86_64-linux/lib/cmake/thrust" pip install . --verbose
```

### install [tomophantom](https://github.com/dkazanc/TomoPhantom) python bindings:

```bash
git clone https://github.com/dkazanc/TomoPhantom.git
cd TomoPhantom
mkdir build
cd build
cmake ../ -DCONDA_BUILD=OFF -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install

# let Python find the shared library
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install

# in the case Python cant find the shared library / cant find or use tomophantom using the export (such was the case for me)
nano .bashrc

# add these lines to your .bashrc:

  # tomophantom
  export LD\_LIBRARY\_PATH="/home/usr/pathToRepo/TomoPhantom/build\_cmake/install/python:$LD\_LIBRARY\_PATH"
  export PYTHONPATH="${PYTHONPATH}:/home/usr/pathToRepo/TomoPhantom/build\_cmake/install/python"

# reload .bashrc and now tomophantom should be found by your compiler
source .bashrc

# even then then sometimes doesnt work / is system dependent, its annoying try getting it from conda ([conda cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html))

# install conda
wget URLtoMinicondaInstallation link
bash Miniconda3-latest-Linux-x86_64.sh
# follow installation instructions and restart bash
conda create --name elsa_tomophantom
conda activate elsa_tomophantom
conda install pip

# install python dependencies for elsa and install elsa here
# install tomophantom as conda package
conda install tomophantom -c ccpi

# or install tomophantom manually as conda package because it doesnt work sometimes, because of weird libgcc errors
pip install cython
cmake ../ -DCONDA_BUILD=ON -DBUILD_MATLAB_WRAPPER=OFF -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install

# tomophantom should now be listed when:
conda list
```
