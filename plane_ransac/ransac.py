from typing import List

import numpy as np
import numpy.typing as npt
import numba as nb
from numba import cuda

from plane_ransac.util import (
    get_plane_from_points,
    measure_distance,
)

__all__ = ["CudaRansac"]

CUDA_THREADS = 1024


class CudaRansac:
    """
    RANSAC implementation using CUDA.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        hypotheses_number: int = CUDA_THREADS,
        initial_points_number: int = 6,
    ):
        """
        Initialize the RANSAC parameters.
        :param threshold: Distance threshold for points to be considered inliers.
        :param hypotheses_number: Number of RANSAC hypotheses. (<= 1024)
        :param initial_points_number: Number of initial points to use in RANSAC.
        """

        self.__threshold: float = threshold
        self.__threads_per_block = min(hypotheses_number, CUDA_THREADS)
        self.__eval_plane = self.__get_eval_plane_kernel(initial_points_number)
        self.__random_hypotheses_cuda = cuda.to_device(
            np.random.random((self.__threads_per_block, initial_points_number))
        )

    def evaluate_planes(self, point_clouds: List[npt.NDArray]):
        """
        For each point cloud in the list of point clouds
        fits a plane and returns the list of them
        :param point_clouds: List of point clouds to fit the plane to.
        :return: NDArray of planes, where each plane corresponds to a point cloud
        """

        combined_point_cloud = np.concatenate(point_clouds)
        block_sizes = [len(pc) for pc in point_clouds]
        blocks_number = len(point_clouds)

        # create result planes and copy it to the device
        result_planes_cuda = cuda.to_device(
            np.zeros((blocks_number, 4), dtype=np.float32)
        )

        # copy combined_point_cloud, block_sizes and block_start_indices to the device
        point_cloud_cuda = cuda.to_device(combined_point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        # block_start_indices is an array of indices where each cuda block should
        # take data from this combined with block_sizes allows it to quickly
        # find the desired part of the point cloud
        block_start_indices_cuda = cuda.to_device(
            np.cumsum(np.concatenate(([0], block_sizes[:-1])))
        )

        # call plane evaluation kernel
        self.__eval_plane[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__random_hypotheses_cuda,
            self.__threshold,
            result_planes_cuda,
        )

        # copy result planes back to the host
        return result_planes_cuda.copy_to_host()

    def evaluate_point_masks(self, point_clouds: List[npt.NDArray]):
        """
        For each point cloud in the list of point clouds,
        fit a plane and return a mask of inliers.
        :param point_clouds: List of point clouds to fit the plane to.
        """

        combined_point_cloud = np.concatenate(point_clouds)
        block_sizes = [len(pc) for pc in point_clouds]
        blocks_number = len(point_clouds)

        # create result mask and planes array and copy them to the device
        result_mask_cuda = cuda.to_device(
            np.zeros((len(combined_point_cloud)), dtype=np.bool_)
        )
        planes_cuda = cuda.to_device(np.zeros((blocks_number, 4), dtype=np.float32))

        # copy combined_point_cloud, block_sizes and block_start_indices to the device
        point_cloud_cuda = cuda.to_device(combined_point_cloud)
        block_sizes_cuda = cuda.to_device(block_sizes)
        # block_start_indices is an array of indices where each cuda block should
        # take data from this combined with block_sizes allows it to quickly
        # find the desired part of the point cloud
        block_start_indices_cuda = cuda.to_device(
            np.cumsum(np.concatenate(([0], block_sizes[:-1])))
        )

        # call plane evaluation and mask computation kernels
        self.__eval_plane[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            self.__random_hypotheses_cuda,
            self.__threshold,
            planes_cuda,
        )
        self.__compute_mask[blocks_number, self.__threads_per_block](
            point_cloud_cuda,
            block_sizes_cuda,
            block_start_indices_cuda,
            planes_cuda,
            self.__threshold,
            result_mask_cuda,
        )

        # copy result mask back to the host
        combined_result_mask = result_mask_cuda.copy_to_host()
        result_masks = np.split(combined_result_mask, np.cumsum(block_sizes)[:-1])
        return result_masks

    @staticmethod
    @cuda.jit
    def __compute_mask(
        point_cloud: npt.NDArray,
        block_sizes: npt.NDArray,
        block_start_indices: npt.NDArray,
        planes: npt.NDArray,
        threshold: float,
        result_mask: npt.NDArray,
    ):
        thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x
        for i in range(
            block_start_indices[block_id] + thread_id,
            block_start_indices[block_id] + block_sizes[block_id],
            cuda.blockDim.x,
        ):
            if measure_distance(planes[block_id], point_cloud[i]) < threshold:
                result_mask[i] = True

    @staticmethod
    def __get_eval_plane_kernel(initial_points_number):
        @cuda.jit
        def eval_plane(
            point_cloud: npt.NDArray,
            block_sizes: npt.NDArray,
            block_start_indices: npt.NDArray,
            random_hypotheses: npt.NDArray,
            threshold: float,
            planes: npt.NDArray,
        ):
            thread_id, block_id = cuda.threadIdx.x, cuda.blockIdx.x

            if block_sizes[block_id] < initial_points_number:
                return

            # select random points as inliers
            initial_point_indices = cuda.local.array(
                shape=initial_points_number, dtype=nb.size_t
            )
            for i in range(initial_points_number):
                initial_point_indices[i] = nb.int32(
                    random_hypotheses[thread_id][i] * block_sizes[block_id]
                    + block_start_indices[block_id]
                )

            # calculate the plane coefficients
            plane = cuda.local.array(shape=4, dtype=nb.float32)
            plane[0], plane[1], plane[2], plane[3] = get_plane_from_points(
                point_cloud, initial_point_indices
            )

            # for each point in the block check if it is an inlier
            inliers_number_local = 0
            for i in range(block_sizes[block_id]):
                point = point_cloud[block_start_indices[block_id] + i]
                distance = measure_distance(plane, point)
                if distance < threshold:
                    inliers_number_local += 1

            # shared memory to store the maximum number of inliers for all hypotheses
            max_inliers_number = cuda.shared.array(shape=1, dtype=nb.int32)
            # this mutex is needed to make sure that only one thread writes
            # to the `planes` array for a given block
            mutex = cuda.shared.array(shape=1, dtype=nb.int32)
            if thread_id == 0:
                max_inliers_number[0] = 0
                mutex[0] = 0
            cuda.syncthreads()

            # replace the maximum number of inliers if the current number is greater
            cuda.atomic.max(max_inliers_number, 0, inliers_number_local)

            # if this thread has the maximum number of inliers
            # write this thread's plane to the `planes` array for a given block
            cuda.syncthreads()
            if (
                inliers_number_local == max_inliers_number[0]
                and cuda.atomic.compare_and_swap(mutex, 0, 1) == 0
            ):
                for i in range(4):
                    planes[block_id][i] = plane[i]
            cuda.syncthreads()

        return eval_plane
