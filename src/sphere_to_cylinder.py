import os
from data_loader.calib.intrinsic_extrinsic_loader import IntrinsicExtrinsicLoader
from data_loader.file_loader import FileLoader
from data_loader.file_writer import FileWriter
from tools.utils import *
from cfg.dataset.cfg import dataset_sensor_frameid_dict
from cfg.dataset.cfg import dataset_rostopic_msg_frameid_dict

import cv2
import math
import argparse

DTOR = math.pi / 180

def sphere_to_cylinder(dataset_dir, sequence_name, calib_path, frame_index=20, cyl_width=1920, vfov=100*DTOR, lat1=-70*DTOR, lat2=30*DTOR, long1=-150*DTOR, long2=150*DTOR):
    int_ext_loader = IntrinsicExtrinsicLoader(is_print=False)
    int_ext_loader.load_calibration(calib_path=calib_path, sensor_frameid_dict=dataset_sensor_frameid_dict)
    camera = int_ext_loader.sensor_collection["theta_camera"]

    img_path = os.path.join(dataset_dir, 'panoramic_images', sequence_name)
    camera_fnames = os.listdir(img_path)

    camera_timesteps = sorted([fname.split('.')[0] for fname in camera_fnames if fname.endswith('.jpg')])
    if frame_index < len(camera_fnames):
        # load rgb
        cam_timestep = camera_timesteps[frame_index]
        cam_img_fname = os.path.join(img_path, cam_timestep + '.jpg')
        cam_img = cv2.imread(cam_img_fname)

        # Convert spherical image to cylindrical
        cam_img_cylinder = camera.sphere_to_cylinder_fast(cam_img, cyl_width, vfov, lat1, lat2, long1, long2)
        return  cam_img, cam_img_cylinder
    else:
        raise ValueError("Frame index exceeds the number of available frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPT Bench Converter')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for the dataset')
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process')
    parser.add_argument('--frame_index', type=int, required=True, default=20, help='Index of the frame to process')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sequence_name = args.sequence_name
    frame_index = args.frame_index

    if int(sequence_name.strip("'")) <= 2:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/before_241216')
    else:
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_loader/calib/after_241216')

    # Get both original and cylindrical images
    original_image, cyl_image = sphere_to_cylinder(dataset_dir, sequence_name, calib_path, frame_index=frame_index)

    # Show original image
    cv2.imshow('Original Image', original_image)
    cv2.waitKey(0)  # Wait for a key press before showing the next image
    cv2.destroyAllWindows()

    # Show cylindrical image
    cv2.imshow('Cylindrical Image', cyl_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()