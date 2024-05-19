# Python Plane RANSAC with CUDA

This is an implementation of the RANSAC algorithm for plane detection in point clouds. The algorithm is implemented in Python using Numba.

This implementation can can be used to segment planes from multiple point clouds in parallel using CUDA.
The hypotheses are tested in parallel as well.

## Installation
To run the exmples, clone the repository
```bash
git clone https://github.com/true-real-michael/python-plane-ransac.git
```
To use in your owd code, install via pip
```bash
pip install git+https://github.com/true-real-michael/python-plane-ransac.git
```

## Usage
See the examples for [plane detection](https://github.com/true-real-michael/python-plane-ransac/blob/main/examples/plane_detection.ipynb) and [outlier removal](https://github.com/true-real-michael/python-plane-ransac/blob/main/examples/outlier_removal.ipynb) in the `examples` folder.

If GitHub fails to render the notebooks, you can view the examples via Jupyter NBViewer: [plane detection](https://nbviewer.org/github/true-real-michael/python-plane-ransac/blob/main/examples/plane_detection.ipynb) and [outlier removal](https://nbviewer.org/github/true-real-michael/python-plane-ransac/blob/main/examples/outlier_removal.ipynb).
