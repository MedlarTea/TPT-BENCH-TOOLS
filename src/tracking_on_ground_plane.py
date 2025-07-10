import os
from data_loader.calib.intrinsic_extrinsic_loader import IntrinsicExtrinsicLoader
from data_loader.file_loader import FileLoader
from data_loader.file_writer import FileWriter
from tools.utils import *
from cfg.dataset.cfg import dataset_sensor_frameid_dict
from cfg.dataset.cfg import dataset_rostopic_msg_frameid_dict
from tracking_visualizer import TrackingVisualizer

import json
import cv2
import open3d
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
import argparse

import time

class Tracker:
    def __init__(self, dataset_dir, sequence_name, calib_path, visualize=True):
        self.dataset_dir = dataset_dir
        self.sequence_name = sequence_name
        self.calib_path = calib_path
        self.visualize = visualize

        self.visualizer = TrackingVisualizer("open3d_camera_params.json")

        self.int_ext_loader = IntrinsicExtrinsicLoader(is_print=False)
        self.int_ext_loader.load_calibration(calib_path=self.calib_path, sensor_frameid_dict=dataset_sensor_frameid_dict)
        self.camera = self.int_ext_loader.sensor_collection["theta_camera"]
        self.T_cam_lidar = self.int_ext_loader.tf_graph.get_relative_transform(self.camera.frame_id, 'os_sensor')

        self.img_path = os.path.join(self.dataset_dir, 'panoramic_images', self.sequence_name)
        self.camera_fnames = os.listdir(self.img_path)
        self.camera_timesteps = sorted([fname.split('.')[0] for fname in self.camera_fnames if fname.endswith('.jpg')])

        self.lidar_point_path = os.path.join(self.dataset_dir, 'lidar_points', self.sequence_name)
        self.pcd_fnames = os.listdir(self.lidar_point_path)
        self.pcd_timesteps = sorted([fname.split('.')[0] for fname in self.pcd_fnames if fname.endswith('.pcd')])

        annotation_path = os.path.join(self.dataset_dir, 'GTs', '{}.json'.format(self.sequence_name))
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.annotated_timesteps = sorted(self.annotations.keys())

        self.kalman_filter = None

    def init_kf(self, dt):
        kf = KalmanFilter(dim_x=4, dim_z=2)  # (x,y,vx,vy)
        kf.x = np.zeros(4)
        # constant-velocity model
        kf.F = np.array([[1., 0., dt, 0.], # x   = x0 + dx*dt
                        [0., 1., 0., dt],  # y   = y0 + dy*dt
                        [0., 0., 1., 0.],  # dx  = dx0
                        [0., 0., 0., 1.]])      # dy  = dy0

        kf.H = np.array([[1., 0., 0., 0.], 
                     [0., 1., 0., 0.]]) 

        # Initial state covariance matrix
        pos_sigma0, vel_sigma0 = 0.2, 0.2         # m, m/s
        kf.P = np.diag([pos_sigma0**2, pos_sigma0**2,
                        vel_sigma0**2, vel_sigma0**2])

        # measurement noise covariance matrix
        pos_sigma_meas = 0.5
        kf.R = np.diag([pos_sigma_meas**2, pos_sigma_meas**2])

        # process noise covariance matrix
        # accel_sigma = 0.01                         # m/s²
        # G = np.array([[0.5*self.dt**2], [0.5*self.dt**2], [self.dt], [self.dt]])
        # kf.Q = G @ G.T * accel_sigma**2           # 4×4

        process_sigma = 0.05
        kf.Q = np.diag([process_sigma**2, process_sigma**2, 
                        process_sigma**2, process_sigma**2])  # Process noise covariance matrix
        return kf

    def track(self):
        for lidar_timestep in tqdm(self.pcd_timesteps):
            lidar_timestamp = float(lidar_timestep)
            closest_cam_timestep, min_time_diff = min(
                    [(cam_timestep, abs(lidar_timestamp - float(cam_timestep))) for cam_timestep in self.camera_timesteps],
                    key=lambda x: x[1]
                )
            if closest_cam_timestep not in self.annotated_timesteps:
                print("No annotation for camera image at timestep:", closest_cam_timestep)
                continue

            min_time_diff = min_time_diff/1e9 
            if min_time_diff > 0.2:
                print(f"Warning: No lidar frame found within 0.3 seconds of image at {lidar_timestamp}. Closest camera frame is {closest_cam_timestep} with time difference {min_time_diff:.3f} seconds.")
                continue

            # target bounding box
            is_exist = self.annotations[closest_cam_timestep]["is_exist"]
            bbox = self.annotations[closest_cam_timestep]["bbox"]
            is_behind_occluded = self.annotations[closest_cam_timestep]["is_behind_occluded"] if "is_behind_occluded" in self.annotations[closest_cam_timestep].keys() else False

            if not is_exist or is_behind_occluded:
                continue
            x0, y0, w, h = bbox

            lidar_pcd_fname = os.path.join(self.lidar_point_path, lidar_timestep + '.pcd')
            xyz_points = np.asarray(open3d.io.read_point_cloud(lidar_pcd_fname).points)
            xyz_points_cam = np.matmul(self.T_cam_lidar[:3, :3], xyz_points.T).T + self.T_cam_lidar[:3, 3].T

            valid_flags, projected_pixels = self.camera.project(xyz_points_cam)
            # indices in the bbox
            valid_indices = np.where((valid_flags) & (x0 <= projected_pixels[:, 0]) & (projected_pixels[:, 0] <= x0 + w) & (y0 <= projected_pixels[:, 1]) & (projected_pixels[:, 1] <= y0 + h))
            dist_in_panoramic_image = np.linalg.norm(xyz_points_cam[valid_indices], axis=1)

            median_depth = np.median(dist_in_panoramic_image) if len(dist_in_panoramic_image) > 30 else None
            center_x = x0 + w / 2
            center_y = y0 + h / 2

            if median_depth is None:
                continue
            target_position_cam = self.camera.unproject(center_x, center_y, median_depth)
            target_position_cam_homo = np.append(target_position_cam, 1)
            target_position_lidar =  np.matmul(np.linalg.inv(self.T_cam_lidar), target_position_cam_homo)[:3]

            if self.kalman_filter is None:
                self.kalman_filter = self.init_kf(dt=0.1)
            self.kalman_filter.predict()
            self.kalman_filter.update(target_position_lidar[:2])

            print("Measurement: {:.3f}, {:.3f}\nKalman:{:.3f}, {:.3f}".format(target_position_lidar[0], target_position_lidar[1], self.kalman_filter.x[0], self.kalman_filter.x[1]))

            _, _ = self.visualizer.step(xyz_points, target_position_lidar, return_vis_handle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPT Bench Tracker')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for the dataset')
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sequence_name = args.sequence_name

    if int(sequence_name.strip("'")) <= 2:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/before_241216')
    else:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/after_241216')
    
    tracker = Tracker(dataset_dir, sequence_name, calib_path, visualize=True)
    tracker.track()