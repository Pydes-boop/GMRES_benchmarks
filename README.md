# GMRES-experiments

Collection of testing and experiments scripts that are used to evaluate AB-GMRES and BA-GMRES implementations for pyelsa. This collection of scripts includes all code that was used to generate benchmarks for the ABBA GMRES implementations in [elsa](https://gitlab.lrz.de/IP/elsa) that were implemented , as well as some experiments using and applying [tomophantom](https://github.com/dkazanc/TomoPhantom) with elsa.

Installation instructions for all dependencies can be found at the [bottom](#prerequisites-and-installation-instructions) of the page

If you are interested in how GMRES is implemented you can also find different python versions, with or without elsa bindings at [Pydes-boop/GMRES](https://github.com/Pydes-boop/GMRES), which also gives a short introduction into the GMRES algorithms and their use-cases.

## Content

### Benchmarking

1.   [matched_comparison.py](benchmarking/matched_comparison.py)
     - compares different matched solvers from elsa against AB-, BA- GMRES
     - fixed phantom size
     - compares scenarios with different max_iterations
     - compares scenarios with different amounts of angles (40, 180, 400) for an arc of 180°
     - compares matched and unmatched versions of AB-, BA- GMRES against CG
     - plots mse against iterations for every scenario
     - plots mse against execution time for every scenario

2.   [angles_comparison.py](benchmarking/angles_comparison.py)
     - currently unused as its weird to plot this data
     - fixed max_iterations
     - fixed phantom size
     - compares different matched solvers from elsa against AB-, BA- GMRES
     - compares scenarios with different amounts of angles starting at 20, up to 420 in steps of +20 for an arc of 180°
     - compares matched and unmatched versions of AB-, BA- GMRES against CG

### Experiments

1.   [tomophantom_elsa_sinograms.py](experiments/tomophantom_elsa_sinograms.py)
     - generates different analytical phantoms using tomophantom and compares them to elsas numerical phantoms
     - compares scenarios with different amounts of angles starting at 20, up to 420 in steps of +20 for an arc of 180°
     - applies noise to tomophantom phantoms
     - recontructs using ABGMRES
     - shows plot of true phantom, elsa sinogram, noisy tomophantom sinogram and ABGMRES reconstruction using noisy tomophantom sinogram

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
```
