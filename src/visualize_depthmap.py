import os
from data_loader.calib.intrinsic_extrinsic_loader import IntrinsicExtrinsicLoader
from data_loader.file_loader import FileLoader
from data_loader.file_writer import FileWriter
from tools.utils import *

from cfg.dataset.cfg import dataset_sensor_frameid_dict
from cfg.dataset.cfg import dataset_rostopic_msg_frameid_dict

import cv2
import open3d
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import argparse

class depthVisualizer:
    def __init__(self, dataset_dir, sequence_name, calib_path, vis_camera_type="theta_camera", frame_index=20, save_dir=None):
        self.dataset_dir = dataset_dir
        self.sequence_name = sequence_name
        self.calib_path = calib_path
        self.vis_camera_type = vis_camera_type
        self.frame_index = frame_index
        self.save_dir = save_dir

        # Load Intrinsics and Extrinsics
        self.int_ext_loader = IntrinsicExtrinsicLoader(is_print=False)
        self.int_ext_loader.load_calibration(calib_path=self.calib_path, sensor_frameid_dict=dataset_sensor_frameid_dict)
        self.camera = self.int_ext_loader.sensor_collection[self.vis_camera_type]
        
        # Load point cloud filenames and camera images
        self.lidar_point_path = os.path.join(self.dataset_dir, 'lidar_points', sequence_name)
        self.pcd_fnames = os.listdir(self.lidar_point_path)
        self.pcd_timesteps = sorted([fname.split('.')[0] for fname in self.pcd_fnames if fname.endswith('.pcd')])

        if self.vis_camera_type == "zed_camera":
            img_path = os.path.join(self.dataset_dir, 'zed_rgb_images', sequence_name)
            camera_fnames = os.listdir(img_path)
        elif self.vis_camera_type == "theta_camera":
            img_path = os.path.join(self.dataset_dir, 'panoramic_images', sequence_name)
            camera_fnames = os.listdir(img_path)
        else:
            raise ValueError("Unsupported camera type. Choose either 'zed_camera' or 'theta_camera'.")
        
        self.camera_timesteps = sorted([fname.split('.')[0] for fname in camera_fnames if fname.endswith('.jpg')])
        
        # Get transformation from camera to lidar
        self.T_cam_lidar = self.int_ext_loader.tf_graph.get_relative_transform(self.camera.frame_id, 'os_sensor')

        self.cmap = plt.get_cmap('jet')
        self.cmap_colors = self.cmap(np.linspace(0, 1, 31)) * 255
    
    def visualize_depthmap(self):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        if self.frame_index == -1:
            for i in tqdm(range(len(self.camera_timesteps))):
                cam_timestep, vis_img = self._process_frame(i)
                if self.save_dir is not None:
                    save_fname = os.path.join(self.save_dir, f"{cam_timestep}.jpg")
                    cv2.imwrite(save_fname, vis_img)
            return None, None
        
        elif self.frame_index < len(self.camera_timesteps):
            return self._process_frame(self.frame_index)
        else:
            raise ValueError("Frame index exceeds the number of available frames.")
    
    def _process_frame(self, frame_index):
        cam_timestep = self.camera_timesteps[frame_index]
        if self.vis_camera_type == "zed_camera":
            cam_img_fname = os.path.join(self.dataset_dir, 'zed_rgb_images', self.sequence_name, cam_timestep + '.jpg')
        elif self.vis_camera_type == "theta_camera":
            cam_img_fname = os.path.join(self.dataset_dir, 'panoramic_images', self.sequence_name, cam_timestep + '.jpg')
        else:
            raise ValueError("Unsupported camera type. Choose either 'zed_camera' or 'theta_camera'.")
        
        cam_img = cv2.imread(cam_img_fname)

        # Find closest lidar frame to the queried camera image
        cam_timestamp = float(cam_timestep)
        closest_lidar_frame, min_time_diff = min(
            [(lidar_timestep, abs(cam_timestamp - float(lidar_timestep))) for lidar_timestep in self.pcd_timesteps],
            key=lambda x: x[1]
        )
        min_time_diff = min_time_diff / 1e9  # Convert to seconds if the timestamps are in microseconds

        if min_time_diff > 0.3:
            raise ValueError(f"Warning: No lidar frame found within 0.3 seconds of {self.vis_camera_type} image at {cam_timestep}. Closest lidar frame is {closest_lidar_frame} with time difference {min_time_diff:.3f} seconds.")
        else:
            lidar_pcd_fname = os.path.join(self.lidar_point_path, closest_lidar_frame + '.pcd')
            xyz_points = np.asarray(open3d.io.read_point_cloud(lidar_pcd_fname).points)
            xyz_points_cam = np.matmul(self.T_cam_lidar[:3, :3], xyz_points.T).T + self.T_cam_lidar[:3, 3].T

            valid_flags, projected_pixels = self.camera.project(xyz_points_cam)
            valid_pixels = projected_pixels[valid_flags]
            valid_points = xyz_points_cam[valid_flags]
            for index in range(len(valid_points)):
                i = int(min(abs(valid_points[index, 2]), 30.0))
                color = (int(self.cmap_colors[i, 0]), int(self.cmap_colors[i, 1]), int(self.cmap_colors[i, 2]))
                cv2.circle(cam_img, (round(valid_pixels[index, 0]), round(valid_pixels[index, 1])), radius=1, color=color, thickness=-1)

            return cam_timestep, cam_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPT Bench Tracker')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for the dataset')
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process')
    parser.add_argument('--frame_index', type=int, default=20, help='Index of the frame to process')
    parser.add_argument('--save_dir', action='store_true', help='Whether to store projected images (default: False)')
    parser.add_argument('--camera_type', type=str, default="theta_camera", help='Name of the sequence to process')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sequence_name = args.sequence_name
    is_save_dir = args.save_dir
    camera_type = args.camera_type
    frame_index = args.frame_index

    if int(sequence_name.strip("'")) <= 2:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/before_241216')
    else:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/after_241216')

    if camera_type == "theta_camera":
        save_dir = os.path.join(dataset_dir, 'theta_projected_images', sequence_name)
    elif camera_type == "zed_camera":
        save_dir = os.path.join(dataset_dir, 'zed_projected_images', sequence_name)
    else:
        raise ValueError("Unsupported camera type. Choose either 'zed_camera' or 'theta_camera'.")

    if is_save_dir:
        os.makedirs(save_dir, exist_ok=True)
        visualizer = depthVisualizer(dataset_dir=dataset_dir, sequence_name=sequence_name, calib_path=calib_path, vis_camera_type=camera_type, frame_index=-1, save_dir=save_dir)
    if not is_save_dir:
        visualizer = depthVisualizer(dataset_dir=dataset_dir, sequence_name=sequence_name, calib_path=calib_path, vis_camera_type=camera_type, frame_index=frame_index, save_dir=None)

    cam_timestep, vis_img = visualizer.visualize_depthmap()

    cv2.imshow("Lidar Projection", vis_img)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

