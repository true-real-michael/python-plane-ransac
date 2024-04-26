# Python Plane RANSAC with CUDA

This is an implementation of the RANSAC algorithm for plane detection in point clouds. The algorithm is implemented in Python using Numba.

This implementation can can be used to segment planes from multiple point clouds in parallel using CUDA.
The hypotheses are tested in parallel as well.

## Installation
```bash
pip install git+https://github.com/true-real-michael/python-plane-ransac.git
```

## Usage
See the example in the `examples` folder.