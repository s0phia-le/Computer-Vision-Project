"""
Intro to Computer Vision (CSCI-431) Final Project.

This project involves detecting and finding motion vectors for cars from 3D LiDAR point cloud data
frames. This is done through the use of multiple pre-processing steps (voxel downsampling, ground 
removal with RANSAC, background subtraction, etc.) and clustering algorithms (DBSCAN) for car position
estimation. Using these estimates, motion vectors for cars can be produced.

Authors: Natalie Zesch, Sydney Tsin, Nicole Orlik, Sophia Le
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange
from collections import deque


@dataclass
class Vehicle:
    # Unique id for each vehicle to track it from frame to frame.
    vehicle_id: int = -1
    # XYZ position
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    # XYZ velocity (difference in position from previous frame)
    mvec_x: float = 0.0
    mvec_y: float = 0.0
    mvec_z: float = 0.0
    # 3D Bounding Box
    bbox_x_min: float = 0.0
    bbox_x_max: float = 1.0
    bbox_y_min: float = 0.0
    bbox_y_max: float = 1.0
    bbox_z_min: float = 0.0
    bbox_z_max: float = 1.0

    @classmethod
    def csv_header(cls):
        return ",".join(cls.__annotations__.keys())

    def csv_row(self):
        return ",".join([str(self.__dict__[field]) for field in self.__annotations__.keys()])


def write_csv_helper(file: Path, vehicles: Iterable[Vehicle]):
    """
    Writes the list of vehicles to a CSV at the given location.

    Args:
        file: the path to the CSV file
        vehicles: the list of vehicles
    """
    with open(file, "w") as f:
        f.write(Vehicle.csv_header() + "\n")
        for v in vehicles:
            f.write(v.csv_row() + "\n")


def load_point_cloud(path_to_cloud: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(path_to_cloud)

# preprocessing point cloud data
def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.10) -> o3d.geometry.PointCloud:
    """
    Preprocess the point cloud by voxel downsampling.
    
    Args:
        pcd: the point cloud
        voxel_size: the size of the voxels to use in voxel downsampling
    Returns:
        the voxel downsampled point cloud
    """

    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def ground_removal(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Use DBSCAN to detect and remove the ground plane from the given point cloud.

    Args:
        pcd: the point cloud
    Returns:
        the point cloud after ground removal
    """

    # finds plane with largest support in the point cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, 
                                         ransac_n=3,
                                         num_iterations=1000) 
    
    # select all points excluding inliers (the ground)
    return pcd.select_by_index(inliers, invert=True)


def remove_high_points(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Removes points above car height from the given point cloud.

    Args:
        pcd: the point cloud
    Returns:
        the point cloud with removed points above car height
    """
    
    points = np.asarray(pcd.points)
    max_height = 2.3
    high_indices = np.where(points[:, 2] > max_height)[0]
    
    low_pcd = pcd.select_by_index(high_indices, invert=True)

    return low_pcd


def background_subtraction(base_pcd: o3d.geometry.PointCloud, pcd: o3d.geometry.PointCloud, dist_thresh: float) -> o3d.geometry.PointCloud:
    """
    Using a base point cloud, perform background subtraction on the current point cloud and remove
    stationary objects.

    Args:
        base_pcd: the base pcd to use as a reference
        pcd: the current point cloud
        dist_thresh: the distance threshold; a distance between points higher than this is considered "non-static"
    Returns:
        the point cloud after background subtraction
    """

    dists = pcd.compute_point_cloud_distance(base_pcd)
    dists = np.asarray(dists)

    # use a distance threshold to create and use a mask of non-static point clouds
    non_static_idxs = np.where(dists > dist_thresh)[0]
    non_static_pcd = pcd.select_by_index(non_static_idxs)
    static_pcd = pcd.select_by_index(non_static_idxs, invert=True)
    non_static_pcd.paint_uniform_color([0, 0, 1.0])
    static_pcd.paint_uniform_color([1.0, 0, 0])

    return non_static_pcd


def estimate_motion(
    curr_vehicles: list[Vehicle],
    prev_vehicles: list[Vehicle],
    next_vehicle_id: int,
    max_match_distance: float = 2.0,
) -> int:
    """
    Estimate vehicle motion by matching current vehicles to previous vehicles
    based on proximity of their positions.

    Args:
        curr_vehicles: the list of vehicles identified in the current frame
        prev_vehicles: the list of vehicles identified in the previous frame
        next_vehicle_id: the next available vehicle id
        max_match_distance: the maximum distance between matched cars where they will still be considered
        the same car
    Returns:
        the next available vehicle id
    """

    # First frame: assign new IDs since there are no previous vehicles
    if not prev_vehicles:
        for v in curr_vehicles:
            v.vehicle_id = next_vehicle_id
            next_vehicle_id += 1

        return next_vehicle_id

    # Calculate distances between current and previous vehicles
    distances = []

    for curr_idx, curr in enumerate(curr_vehicles):
        for prev_idx, prev in enumerate(prev_vehicles):

            dx = curr.position_x - prev.position_x
            dy = curr.position_y - prev.position_y
            dz = curr.position_z - prev.position_z

            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            distances.append((dist, curr_idx, prev_idx))

    # Sort distances from smallest to largest
    distances.sort(key=lambda x: x[0])

    matched_curr = set()
    matched_prev = set()

    # Match vehicles based on closest distance
    for dist, curr_idx, prev_idx in distances:

        # Ignore matches that are too far away
        if dist > max_match_distance:
            break

        # Ensure each vehicle is matched only once
        if curr_idx in matched_curr or prev_idx in matched_prev:
            continue

        curr = curr_vehicles[curr_idx]
        prev = prev_vehicles[prev_idx]

        # Reuse previous vehicle ID
        curr.vehicle_id = prev.vehicle_id

        # Calculate motion vector (position difference)
        curr.mvec_x = curr.position_x - prev.position_x
        curr.mvec_y = curr.position_y - prev.position_y
        curr.mvec_z = curr.position_z - prev.position_z

        matched_curr.add(curr_idx)
        matched_prev.add(prev_idx)

    # Assign new IDs to vehicles that could not be matched
    for curr_idx, curr in enumerate(curr_vehicles):

        if curr_idx not in matched_curr:
            curr.vehicle_id = next_vehicle_id
            next_vehicle_id += 1

            curr.mvec_x = 0.0
            curr.mvec_y = 0.0
            curr.mvec_z = 0.0

    return next_vehicle_id

def clustering(pcd: o3d.geometry.PointCloud, vehicles, eps, min_points):
    """
    Clusters the point cloud using DBSCAN, finds centroids for each cluster,
    and adds bounding boxes around clusters.
    
    Args:
        pcd: the point cloud
        vehicles: the list of vehicles to add clusters to
        eps: density parameter used to find neighbouring points for DBSCAN
        min_points: the minimum number of points needed to form a cluster for DBSCAN
    """
    
    # run the DBSCAN clustering
    with o3d.utility.VerbosityContextManager(
    o3d.utility.VerbosityLevel.Error
    ) as cm:
        labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=False))
        

    # Analyze clusters
    if len(labels) != 0:
        max_label = labels.max()
        bounding_boxes = []

        for i in range(max_label + 1):
            # get indices of points belonging to cluster i
            cluster_indices = np.where(labels == i)[0]
        
            # extract the cluster as its own PointCloud
            cluster_pcd = pcd.select_by_index(cluster_indices)
            
            # get centroid 
            x, y, z = cluster_pcd.get_center()
            
            # get the Axis-Aligned Bounding Box (AABB)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0) 
            bounding_boxes.append(aabb)
            
            # access raw min/max coordinate arrays if needed
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()
            
            # add the vehicle
            vehicles.append(Vehicle(
                vehicle_id=i,
                position_x=x,
                position_y=y,
                position_z=z,
                bbox_x_max= max_bound[0],
                bbox_x_min= min_bound[0],
                bbox_y_max= max_bound[1],
                bbox_y_min= min_bound[1],
                bbox_z_max= max_bound[2],
                bbox_z_min= min_bound[2]
                )
            )    
        

def main(
    data_path: Path,
    output_path: Path = "perception_results",
    start_index: int = 0,
    end_index: int = -1,
):
    """
    Entry point.

    Args:
        data_path: path to the point cloud data
        output_path: path to the results folder location
        start_index: the frame to begin processing on
        end_index: the frame to end processing on
    """
    if end_index < 0:
        end_index = len(list(data_path.glob("*.pcd"))) + end_index

    previous_vehicles = []
    next_vehicle_id = 0

    # Load each point cloud and display it
    queue = deque()
    for frame_number in trange(start_index, end_index + 1, desc="Processing Frames"):
        vehicles: list[Vehicle] = []

        # Loading point cloud
        pcd = load_point_cloud(data_path / f"{frame_number}.pcd")

        pcd = preprocess_point_cloud(pcd, voxel_size=0.10)

        # ground removal (RANSAC)
        pcd = ground_removal(pcd)

        # remove high points (above cars)
        pcd = remove_high_points(pcd)

        # background subtraction
        dist_threshold = 0.003
        if frame_number <= 2:
            queue.append(pcd)
        elif frame_number < 10:
            queue.append(pcd)
            base_pcd = queue[0]
            pcd = background_subtraction(base_pcd, pcd, dist_threshold)
        else:
            queue.append(pcd)
            base_pcd = queue.popleft()
            pcd = background_subtraction(base_pcd, pcd, dist_threshold)

        if frame_number > 2:
            # clustering (DBSCAN + centroid calculation)
            clustering(pcd, vehicles, eps=2, min_points=2)

            # motion estimation
            next_vehicle_id = estimate_motion(vehicles, previous_vehicles, next_vehicle_id, max_match_distance=2.0)

            # Write list of vehicles as a CSV
            write_csv_helper(output_path / f"{frame_number}.csv", vehicles)

            previous_vehicles = vehicles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path",
        type=Path,
        help="Directory containing .pcd files",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("perception_results"),
        help="Directory where .csv outputs will be saved",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Index of first frame",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        type=int,
        default=-1,
        help="Index of last frame (defaults to -1 for last frame)",
    )
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    main(**vars(args))
