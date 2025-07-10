import os
import rosbag
import argparse
from data_loader.ros_msg.pointcloud import PointCloud
from data_loader.ros_msg.image import Image
from data_loader.ros_msg.odometry import Odometry
from data_loader.calib.intrinsic_extrinsic_loader import IntrinsicExtrinsicLoader
from tools.utils import *

from cfg.dataset.cfg import dataset_sensor_frameid_dict
from cfg.dataset.cfg import dataset_rostopic_msg_frameid_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TPT Bench Writer')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for the dataset')
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process')

    parser.add_argument('--zed_rgb', action='store_true', help='Whether to store zed rgb images (default: False)')
    parser.add_argument('--zed_path_odom', action='store_true', help='Whether to zed path odom')
    parser.add_argument('--lidar_points', action='store_true', help='Whether to store lidar points (default: False)')
    
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sequence_name = args.sequence_name

    # sequence_name = '2024-12-24-18-37-36'
    # dataset_path = '/home/hjyeee/Data/Dataset/rpf_dataset/TPT-Bench'


    ##### Set up the rosbag path
    rosbag_path = os.path.join(dataset_dir, 'rosbags', sequence_name + '.bag')

    ##### Set up the message topic list for different platforms
    for key, value in dataset_sensor_frameid_dict.items():
        print('Sensor: {:<30}, Frame_id: {:<15}'.format(key, value[0]))

    print('Finish loading parameters')
    
    ##### Open the rosbag
    input_bag = rosbag.Bag(rosbag_path)
    print('Finish reading bag, start loading messages, and writing messages to data folder')

    ##### Initialize the lidar object of Sensor class
    # Ouster
    if args.lidar_points and 'ouster_points' in dataset_rostopic_msg_frameid_dict.keys():
        print('Loading ouster messages...')
        pointcloud = PointCloud(sensor_type='ouster')
        output_data_path = os.path.join(dataset_dir, 'lidar_points', sequence_name)
        os.makedirs(output_data_path, exist_ok=True)
        num_msg = pointcloud.load_messages_write_to_file(bag=input_bag, output_path=output_data_path, topic=dataset_rostopic_msg_frameid_dict['ouster_points'][0])
        print('     Saved {} Ouster points messages !'.format(num_msg))

    # theta_camera (NOT USED)
    # if 'theta_image' in dataset_rostopic_msg_frameid_dict.keys():
    #     print('Loading theta_camera messages...')
    #     frame_left_image = Image(sensor_type='frame_cam', msg_type=dataset_rostopic_msg_frameid_dict['theta_image'][1])
    #     output_data_path = os.path.join(dataset_path, sequence_name, 'theta_images')
    #     num_msg = frame_left_image.load_messages_write_to_file(bag=input_bag, output_path=output_data_path, topic=dataset_rostopic_msg_frameid_dict['theta_image'][0])
    #     print('     Saving {} theta_camera image messages !'.format(num_msg))

    # zed_camera
    if args.zed_rgb and 'zed_rgb_image' in dataset_rostopic_msg_frameid_dict.keys():
        print('Loading zed_camera messages...')
        frame_left_image = Image(sensor_type='frame_cam', msg_type=dataset_rostopic_msg_frameid_dict['zed_rgb_image'][1])
        output_data_path = os.path.join(dataset_dir, 'zed_rgb_images', sequence_name)
        os.makedirs(output_data_path, exist_ok=True)
        num_msg = frame_left_image.load_messages_write_to_file(bag=input_bag, output_path=output_data_path, topic=dataset_rostopic_msg_frameid_dict['zed_rgb_image'][0])
        print('     Saving {} zed_camera image messages !'.format(num_msg))
    
    # path odom
    if args.zed_path_odom and 'zed_path_odom' in dataset_rostopic_msg_frameid_dict.keys():
        print('Loading zed_odometry messages...')
        odometry = Odometry(traj_type='TUM', msg_type=dataset_rostopic_msg_frameid_dict['zed_path_odom'][1])
        output_data_path = os.path.join(dataset_dir, 'odometry', sequence_name)
        os.makedirs(output_data_path, exist_ok=True)
        num_msg = odometry.load_messages_write_to_file(bag=input_bag, output_path=output_data_path, topic=dataset_rostopic_msg_frameid_dict['zed_path_odom'][0])
        print('     Saving {} zed_odometry messages !'.format(num_msg))

    
    
    
    ##### Close the rosbag
    input_bag.close()
    print('Close the rosbag')

