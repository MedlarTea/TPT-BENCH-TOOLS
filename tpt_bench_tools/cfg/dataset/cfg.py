#!/usr/bin/python3

dataset_sensor_frameid_dict = {
  'ouster': ['os_sensor'],
  'ouster_imu': ['os_imu'],
  'theta_camera': ["theta_camera_optical_frame"],
  'zed_camera':["zed2_left_camera_optical_frame"],
  'zed_imu':["zed2_imu_link"],
  'zed_points':["zed2_left_camera_frame"],
}

# topic, msg, frameid
dataset_rostopic_msg_frameid_dict = {
  'ouster_imu': ['/ouster/imu', 'sensor_msgs/Imu', 'os_imu'],
  'ouster_points': ['/ouster/points', 'sensor_msgs/PointCloud2', 'os_sensor'],
  # 
  'theta_camera_info': ['/camera/color/camera_info', 'sensor_msgs/CameraInfo', 'theta_camera_optical_frame'],
  'theta_image': ['/camera/color/image_raw/compressed', 'sensor_msgs/CompressedImage', 'theta_camera_optical_frame'],
  #
  'zed_rgb_camera_info': ['/zed2/zed_node/rgb/camera_info', 'sensor_msgs/CameraInfo', 'zed2_left_camera_optical_frame'],
  'zed_rgb_image': ['/zed2/zed_node/rgb/image_rect_color/compressed', 'sensor_msgs/CompressedImage', 'zed2_left_camera_optical_frame'],
  'zed_depth_camera_info': ['/zed2/zed_node/depth/camera_info', 'sensor_msgs/CameraInfo', 'zed2_left_camera_optical_frame'],
  'zed_depth_image': ['/zed2/zed_node/depth/depth_registered', 'sensor_msgs/Image', 'zed2_left_camera_optical_frame'],
  'zed_depth_points': ['/zed2/zed_node/point_cloud/cloud_registered', 'sensor_msgs/PointCloud2', 'zed2_left_camera_frame'],
  'zed_imu': ['/zed2/zed_node/imu/data', 'sensor_msgs/Imu', 'zed2_imu_link'],
  'zed_odom': ['/zed2/zed_node/odom', 'nav_msgs/Odometry', 'odom'],
  'zed_path_odom': ['/zed2/zed_node/path_odom', 'nav_msgs/Path', 'odom'],
  'zed_path_map': ['/zed2/zed_node/path_map', 'nav_msgs/Path', 'map'],
  #
  'tf_static': ['/tf_static', 'tf2_msgs/TFMessage', 'none']
}

if __name__ == '__main__':
  for key, value in dataset_sensor_frameid_dict.items():
    print('Sensor: {:<20}, Frame_id: {:<15}'.format(\
      key, value[0]))

  for key, value in dataset_rostopic_msg_frameid_dict.items():
    print('Sensor Involved: {:<30}, ROSTopic: {:<50}, Msg_type: {:<60}, Frame_id: {:<15}'.format(\
      key, value[0], value[1], value[2]))