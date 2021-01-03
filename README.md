# Video Games Sales Prediction

This project predicts the global sales of the video games with machine learning algorithms and models. 

## Getting Started

### Prerequisites

* Python 3.7
* Intel Math Kernel Library ([MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html))
* C++ 11 or newer
* Tensorflow
* Numpy, Pandas

## Quick Start
Assume you are initially at the root directory.

Open `src/CMakeLists.txt` and change ``SET(MKLROOT path/to/mkl)`` to the directory to your Intel MKL folder. The default directory is `/opt/intel/compilers_and_libraries_2020.2.258/mac/mkl`

Use the script in `video-game-sales-prediction/bin/vgsales.sh` to run the code.

```Shell
> cd ..
> ./bin/vgsales.sh run
```

You can see the plots created in `graphs/`. Use `help` for more information about the Shell script.

``> ./bin/vgsales.sh --help``

