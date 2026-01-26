import open3d as o3d
import numpy as np


class TrackingVisualizer:
    def __init__(self, view_param):
        if isinstance(view_param, str):
            self.view_param = o3d.io.read_pinhole_camera_parameters(view_param)
        else:
            self.view_param = view_param
        
        print(self.view_param)

        self.window_height = self.view_param.intrinsic.height
        self.window_width = self.view_param.intrinsic.width

        print("[OPEN3D] window size (HxW): ", self.window_height, self.window_width)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            width = self.window_width,
            height = self.window_height,
        )
        
        self.vis_ctrl = self.vis.get_view_control()
        # self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)

        self.vis_ctrl.set_lookat([0, 0, 0]) 
        self.vis_ctrl.set_front([0, 0, 1]) 
        self.vis_ctrl.set_up([0, -1, 0]) 
        self.vis_ctrl.set_zoom(0.8)    

    def create_3d_bbox(self, center, size, color=[1, 0, 0]):
        dx, dy, dz = size
        cx, cy, cz = center
        # 8 corners of the box
        corners = np.array([
            [cx - dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy + dy / 2, cz + dz / 2],
            [cx - dx / 2, cy + dy / 2, cz + dz / 2],
        ])

        # edges between corners
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical lines
        ]

        # one color for all lines
        colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def step(
            self, 
            # img_height, 
            # img_width,
            points=None,
            person_position=None,
            return_vis_handle=False,
            ):
        self.vis.clear_geometries()

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)  # Size for LiDAR coordinate system
        self.vis.add_geometry(coordinate_frame)

        if person_position is not None:
            bbox = self.create_3d_bbox(center=person_position + np.array([0, 0, 0]),
                            size=(0.6, 0.6, 1.7),
                            color=[0, 1, 0])
            self.vis.add_geometry(bbox)

        if points is not None:
            pcd = o3d.geometry.PointCloud()
            points = points[points[:, 2] <= 1.7]  # Filter out points with height greater than 1.7
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(points), 3)))  # Set color to black
            self.vis.add_geometry(pcd)
        
        # Set the view control to focus on the target
        # if person_position is not None:
        #     print(f"Setting camera to look at position: {person_position}")
        #     self.vis_ctrl.set_lookat(person_position)  # Target position (look at the target)
        #     self.vis_ctrl.set_up([0, 0, 1])  # Adjust the up direction for a better view
        #     self.vis_ctrl.set_front([0, -1, 0])  # Adjust the front direction for a clearer view
        #     self.vis_ctrl.set_zoom(1.0)  # Adjust zoom level to ensure the target is clearly visible

        self.vis.poll_events()
        self.vis.update_renderer()

        rendered_image = self.vis.capture_screen_float_buffer(False)
        rendered_image = np.asarray(rendered_image)

        if return_vis_handle:
            return rendered_image, self.vis
        else:
            return rendered_image, None

