#!/usr/bin/python3

import cv2
import numpy as np
import math

class CameraPanoramic:
  def __init__(self, frame_id, width, height, dataset_name, camera_name):
    self.frame_id = frame_id
    self.width = width
    self.height = height
    self.dataset_name = dataset_name
    self.camera_name = camera_name
    self.EPS = 1e-4
  
  def __str__(self):
    """Returns a string representation of the Camera object."""
    return ('Camera ({}) with intrinsics at {}:\n'
            'Width: {}, '
            'Height: {}\n').format(self.camera_name, self.frame_id, self.width, self.height)

  def undistort(self, image):
    undistorted_image = cv2.undistort(image, self.K, self.D)
    return undistorted_image
  
  def precompute_lat_lon_map(self, cyl_width, cyl_height, vfov, lat1, lat2, long1, long2):
    x = np.linspace(0, 1, cyl_width)
    y = np.linspace(0, 1, cyl_height)
    xx, yy = np.meshgrid(x, y)

    y0 = (math.tan(lat1) + math.tan(lat2)) / (math.tan(lat1) - math.tan(lat2))
    tanmaxlat = math.tan(0.5 * vfov)
    tanlat1 = math.tan(lat1) / (-1 - y0)
    tanlat2 = math.tan(lat2) / (1 - y0)

    longitude = long1 + xx * (long2 - long1)

    if abs(abs(lat1) - abs(lat2)) < self.EPS:
        latitude = np.arctan(yy * tanmaxlat)
    else:
        latitude = np.where(yy > y0,
                            np.arctan((yy - y0) * tanlat2),
                            np.arctan((yy - y0) * tanlat1))
    return longitude, latitude
  
  def sphere_to_cylinder_fast(self, sphere_image, cyl_width, vfov, lat1, lat2, long1, long2):
    sphereheight, spherewidth = sphere_image.shape[:2]
    cyl_height = int(cyl_width * math.tan(0.5 * vfov) / (0.5 * (long2 - long1)))

    longitude, latitude = self.precompute_lat_lon_map(cyl_width, cyl_height, vfov, lat1, lat2, long1, long2)

    map_x = ((longitude + math.pi) / (2 * math.pi)) * (spherewidth - 1)
    map_y = ((latitude + math.pi / 2) / math.pi) * (sphereheight - 1)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    cyl_image = cv2.remap(sphere_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return cyl_image
  
  def clamp(self, value, min_val, max_val):
    return max(min_val, min(value, max_val))

  def unproject(self, u, v, depth):
      u_normalized = u/self.width
      v_normalized = v/self.height

      # Clamp u_normalized and v_normalized to [0, 1]
      u_normalized = self.clamp(u_normalized, 0.0, 1.0)
      v_normalized = self.clamp(v_normalized, 0.0, 1.0)

      # Calculate theta and phi
      theta = u_normalized * 2 * math.pi - math.pi
      phi = v_normalized * math.pi - math.pi / 2

      # Clamp phi to avoid numerical issues at the poles
      phi = self.clamp(phi, -math.pi / 2 + 1e-6, math.pi / 2 - 1e-6)

      # Compute the 3D coordinates
      x = depth * math.cos(phi) * math.sin(theta)
      y = depth * math.sin(phi)
      z = depth * math.cos(phi) * math.cos(theta)

      return x, y, z
  
  
  def project(self, p_C):
        """
        Project a batch of 3D points to 2D (panoramic image coordinates).
        The input `p_C` should have shape (N, 3).
        The output is an array of shape (N, 2), where each row is a 2D point.
        """
        # Initialize the output array
        u_C = np.zeros((p_C.shape[0], 2))
        
        # Mask for valid points (i.e., no NaN values)
        valid_mask = np.all(np.isnan(p_C), axis=1) == False
        
        # Filter the points that are valid
        valid_points = p_C[valid_mask]
        
        # Calculate theta and phi for valid points
        theta = np.arctan2(valid_points[:, 0], valid_points[:, 2])
        phi = np.arcsin(-valid_points[:, 1] / np.linalg.norm(valid_points, axis=1))
        
        # Calculate u and v (image coordinates) for valid points
        u = (theta + np.pi) / (2 * np.pi) * self.width
        v = (np.pi / 2 - phi) / np.pi * self.height
        
        # Clip the u, v values to be within the image bounds
        u_C[valid_mask, 0] = np.clip(u, 0, self.width - 1)
        u_C[valid_mask, 1] = np.clip(v, 0, self.height - 1)
        
        # Return a tuple: (valid flag array, projected points)
        return valid_mask, u_C
