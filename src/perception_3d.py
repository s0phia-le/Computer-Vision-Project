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
    # Start with header by inspecting field names of the Vehicle class; if the list of vehicles is
    # empty then we need a new default Vehicle for the header:
    with open(file, "w") as f:
        f.write(Vehicle.csv_header() + "\n")
        for v in vehicles:
            f.write(v.csv_row() + "\n")


def load_point_cloud(path_to_cloud: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(path_to_cloud)

# preprocessing point cloud data
def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.10) -> o3d.geometry.PointCloud:
    """preprocess the point cloud by downsampling and removing outliers."""
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd

def ground_removal(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """segment point cloud to remove plane"""

    # finds plane with largest support in the point cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.4,
                                         ransac_n=3,
                                         num_iterations=1000)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # o3d.visualization.draw_geometries(
    #     [inlier_cloud, outlier_cloud],
    #     zoom=0.8,
    #     front=[-0.4999, -0.1659, -0.8499],
    #     lookat=[2.1813, 2.0619, 2.0999],
    #     up=[0.1204, -0.9852, 0.1215]
    # )
    
    # select all points excluding inliers (the ground)
    return pcd.select_by_index(inliers, invert=True)


def remove_high_points(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    max_height = 2.3
    high_indices = np.where(points[:, 2] > max_height)[0]
    
    low_pcd = pcd.select_by_index(high_indices, invert=True)

    # high_pcd = pcd.select_by_index(high_indices)
    # high_pcd.paint_uniform_color([1.0, 0, 0])
    # o3d.visualization.draw_geometries(
    #     [high_pcd, low_pcd],
    #     zoom=0.8,
    #     front=[-0.4999, -0.1659, -0.8499],
    #     lookat=[2.1813, 2.0619, 2.0999],
    #     up=[0.1204, -0.9852, 0.1215]
    # )

    return low_pcd


def background_subtraction(base_pcd: o3d.geometry.PointCloud, pcd: o3d.geometry.PointCloud, dist_thresh: float) -> o3d.geometry.PointCloud:
    """
    Using a base point cloud, perform background subtraction on the current point cloud and remove
    stationary objects.
    """

    dists = pcd.compute_point_cloud_distance(base_pcd)
    dists = np.asarray(dists)

    # use a distance threshold to create and use a mask of non-static point clouds
    non_static_idxs = np.where(dists > dist_thresh)[0]
    non_static_pcd = pcd.select_by_index(non_static_idxs)
    static_pcd = pcd.select_by_index(non_static_idxs, invert=True)
    non_static_pcd.paint_uniform_color([0, 0, 1.0])
    static_pcd.paint_uniform_color([1.0, 0, 0])

    o3d.visualization.draw_geometries(
        [non_static_pcd, static_pcd],
        zoom=0.8,
        front=[-0.4999, -0.1659, -0.8499],
        lookat=[2.1813, 2.0619, 2.0999],
        up=[0.1204, -0.9852, 0.1215]
    )

    return non_static_pcd


def visualize_cloud(pcd: o3d.geometry.PointCloud):
    """visualizes a given point cloud"""
    o3d.visualization.draw_geometries([pcd],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])

def estimate_motion(curr_vehicles: list[Vehicle], prev_vehicles: list[Vehicle], next_vehicle_id: int, max_match_distance: float = 2.0, frame_dt: float = 1.0) -> int:
    """Estimate motion of vehicles by matching current vehicles to previous vehicles based on proximity."""
    # first frame: assign IDs but velocity unknown
    if not prev_vehicles:
        for v in curr_vehicles:
            v.vehicle_id = next_vehicle_id
            next_vehicle_id += 1

        return next_vehicle_id

    # calc dist between every current and previous vehicle
    distances = []
    for curr_idx, curr in enumerate(curr_vehicles):
        for prev_idx, prev in enumerate(prev_vehicles):
            dx = curr.position_x - prev.position_x
            dy = curr.position_y - prev.position_y
            dz = curr.position_z - prev.position_z

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            distances.append((dist, curr_idx, prev_idx))

    # closest pairs
    distances.sort(key=lambda x: x[0])
    matched_curr = set()
    matched_prev = set()

    # match vehicle
    for dist, curr_idx, prev_idx in distances:
        if dist > max_match_distance:
            break
        if curr_idx in matched_curr or prev_idx in matched_prev:
            continue

        curr = curr_vehicles[curr_idx]
        prev = prev_vehicles[prev_idx]

        # assign ID and calculate velocity
        curr.vehicle_id = prev.vehicle_id
        curr.mvec_x = curr.position_x - prev.position_x 
        curr.mvec_y = curr.position_y - prev.position_y
        curr.mvec_z = curr.position_z - prev.position_z

        matched_curr.add(curr_idx)
        matched_prev.add(prev_idx)

        # assign ids to unmatched vehicles
        for curr_idx, v in enumerate(curr_vehicles):
            if curr_idx not in matched_curr:
                v.vehicle_id = next_vehicle_id
                next_vehicle_id += 1

                v.mvec_x = 0.0
                v.mvec_y = 0.0
                v.mvec_z = 0.0
    return next_vehicle_id

def clustering(pcd: o3d.geometry.PointCloud, eps, min_points):
    """cluster the point cloud using DBSCAN"""
    
    # run the DBSCAN clustering
    with o3d.utility.VerbosityContextManager(
    o3d.utility.VerbosityLevel.Debug
    ) as cm:
        labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))

    # 3. Analyze clusters
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters (excluding noise)")
    
    # visualize clusies 
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Black/zero for noise
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def main(
    data_path: Path,
    output_path: Path = "perception_results",
    start_index: int = 0,
    end_index: int = -1,
    # YOU MAY ADD OTHER ARGUMENTS HERE.
    # IF YOU DO, SUPPLY GOOD DEFAULTS BOTH HERE AND IN THE ARGUMENT PARSER.
):
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

        # YOUR CODE HERE: do whatever to populate the list of vehicles for this frame. You may use
        # any functions available in open3d for background subtraction, road surface detection,
        # clustering, etc.

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
            clustering(pcd, eps=1.5, min_points=6)

            # motion estimation
            next_vehicle_id = estimate_motion(vehicles, previous_vehicles, next_vehicle_id, max_match_distance=2.0, frame_dt=1.0)

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
